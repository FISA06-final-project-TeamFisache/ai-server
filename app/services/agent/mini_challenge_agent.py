from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.schemas.mini_challenge import (
    AdjustRequest,
    AdjustResponse,
    ChallengeType,
    MiniChallengeRequest,
    MiniChallengeResponse,
)
from app.services.agent.llm import ainvoke_structured
from app.services.agent.tools import get_all_prices
from app.services.session import get_session, save_session

logger = logging.getLogger(__name__)

_SubType = Literal["COFFEE", "DELIVERY", "ALCOHOL", "LATE_NIGHT", "LUNCH", "SHOPPING", "TAXI"]

# ── AI 출력 스키마 ────────────────────────────────────────────────────────────

class _MiniChallengeAIOutput(BaseModel):
    title: str
    description: str
    challenge_type: ChallengeType
    target: int
    category: str
    estimated_saving: int
    ticker: str
    challenge_sub_type: _SubType


class _AdjustAIOutput(BaseModel):
    title: str
    description: str
    challenge_type: ChallengeType
    target: int | None
    category: str
    estimated_saving: int
    ticker: str
    challenge_sub_type: _SubType


# ── 시스템 프롬프트 ────────────────────────────────────────────────────────────

_SUB_TYPE_MAP = (
    "COFFEE(카페·음료), "
    "DELIVERY(배달음식 — sender_name에 배달의민족·요기요·쿠팡이츠가 포함된 식비 항목), "
    "ALCOHOL(주류·술자리), LATE_NIGHT(야식), LUNCH(점심·외식), "
    "SHOPPING(쇼핑·의류), TAXI(택시·교통)"
)

_INIT_SYSTEM = (
    "당신은 소비 패턴 분석 전문가입니다.\n"
    "아래 순서대로 미니 챌린지를 제안하세요.\n\n"
    "1단계 — challenge_sub_type 선정:\n"
    f"   소비 금액이 가장 많은 카테고리를 분석해 아래 중 정확히 하나를 선택합니다.\n"
    f"   {_SUB_TYPE_MAP}\n"
    "   ※ DELIVERY: 배달의민족·요기요·쿠팡이츠 등 배달앱이 sender_name에 포함된 항목은 모두 합산해 DELIVERY 소비 총액으로 계산합니다.\n\n"
    "2단계 — 챌린지 설계:\n"
    "   선정한 challenge_sub_type을 기준으로 나머지 필드를 결정합니다.\n"
    "   - challenge_type: 횟수 제한 → COUNT / 금액 제한 → AMOUNT\n"
    "   - target: 목표값 (COUNT → 횟수, AMOUNT → 한달 소비 한도(원))\n"
    "     ※ AMOUNT일 때 target은 반드시 해당 카테고리의 이번 달 소비 총액보다 작아야 합니다.\n"
    "       target ≥ 소비 총액이면 평소대로 써도 챌린지가 통과되므로 행동 변화가 없습니다.\n"
    "   - estimated_saving: 소비 총액 − target (원)\n"
    "   - description: 아래 순서로 2~3문장 작성합니다.\n"
    "     1) 해당 카테고리 이번 달 소비 금액 언급\n"
    "     2) 챌린지 목표(target) 안내\n"
    "     3) 절약 금액(estimated_saving)으로 주식 투자에 가까워진다는 동기 부여\n"
    "     예) '이번 달 배달 음식에 450,000원을 쓰셨어요. 이번 달은 400,000원 이하로 줄여봐요! 절약한 50,000원으로 주식 투자에 한 걸음 더 가까워질 수 있어요.'\n\n"
    "3단계 — ticker 선택:\n"
    "   아래 현재 주가 목록에서 소비 카테고리와 관심 테마에 맞는 종목 1개를 선택합니다."
)

_ADJUST_SYSTEM = (
    "당신은 소비 패턴 분석 전문가입니다.\n"
    "사용자 피드백에 맞게 챌린지를 조정하거나 새 챌린지를 제안합니다.\n\n"
    "1단계 — challenge_sub_type 선정:\n"
    f"   반드시 아래 중 정확히 하나를 선택합니다.\n"
    f"   {_SUB_TYPE_MAP}\n"
    "   ※ DELIVERY: 배달의민족·요기요·쿠팡이츠 등 배달앱이 sender_name에 포함된 항목은 모두 합산해 DELIVERY 소비 총액으로 계산합니다.\n"
    "   - '더 쉽게/어렵게': 직전 챌린지와 같은 sub_type 유지\n"
    "   - '주제를 바꿔주세요': 직전과 다른 sub_type 선택\n"
    "   - previous_proposals에 있는 챌린지 절대 반복 금지\n\n"
    "2단계 — 챌린지 조정:\n"
    "   - '더 쉽게': target을 늘려 감소 폭을 줄입니다.\n"
    "     단, AMOUNT일 때 target은 반드시 해당 카테고리 소비 총액 미만으로 유지해야 합니다.\n"
    "     target ≥ 소비 총액이면 평소대로 소비해도 챌린지가 통과되므로 의미가 없습니다.\n"
    "   - '더 어렵게': target을 줄여 감소 폭을 늘립니다.\n"
    "   - '주제를 바꿔주세요': 소비 데이터 기반 새 챌린지 설계\n"
    "   - description: 아래 순서로 2~3문장 작성합니다.\n"
    "     1) 해당 카테고리 이번 달 소비 금액 언급\n"
    "     2) 챌린지 목표(target) 안내\n"
    "     3) 절약 금액(estimated_saving)으로 주식 투자에 가까워진다는 동기 부여\n"
    "     예) '이번 달 배달 음식에 450,000원을 쓰셨어요. 이번 달은 400,000원 이하로 줄여봐요! 절약한 50,000원으로 주식 투자에 한 걸음 더 가까워질 수 있어요.'\n\n"
    "3단계 — ticker 선택:\n"
    "   아래 현재 주가 목록에서 적합한 종목 1개를 선택합니다."
)


async def _fetch_prices_text() -> str:
    prices = await get_all_prices()
    if not prices:
        return "주가 조회 실패"
    return "\n".join(f"- {ticker}: {price:,}원" for _, ticker, price in prices)


# ── 엔드포인트 핸들러 ─────────────────────────────────────────────────────────

async def propose_mini_challenge(req: MiniChallengeRequest) -> MiniChallengeResponse:
    session = await get_session(req.user_id)

    session["category_expense"] = [
        {"category": c.category, "amount": c.amount, "sender_name": c.sender_name}
        for c in req.category_expense
    ]
    session["stock_themes"] = req.stock_themes

    cats = "\n".join(
        f"- {c.category}({c.sender_name}): {c.amount:,}원"
        for c in sorted(req.category_expense, key=lambda x: x.amount, reverse=True)
    ) or "소비 데이터 없음"
    themes_text = ", ".join(req.stock_themes) if req.stock_themes else "없음"
    prices_text = await _fetch_prices_text()

    context = (
        f"관심 주식 테마: {themes_text}\n\n"
        f"이번 달 소비 패턴:\n{cats}\n\n"
        f"현재 주가:\n{prices_text}"
    )

    try:
        result = await ainvoke_structured(
            [SystemMessage(content=_INIT_SYSTEM), HumanMessage(content=context)],
            _MiniChallengeAIOutput,
        )
    except Exception as e:
        logger.warning("propose_mini_challenge 실패: %s", e)
        result = None

    if result:
        session.setdefault("proposals", []).append({**result.model_dump(), "feedback": ""})
        await save_session(req.user_id, session)
        return MiniChallengeResponse(created_at=datetime.now(timezone.utc), **result.model_dump())

    await save_session(req.user_id, session)
    return _default_challenge_response()


async def adjust_challenge(req: AdjustRequest) -> AdjustResponse:
    session = await get_session(req.user_id)

    cats_raw = session.get("category_expense", [])
    cats = "\n".join(
        f"- {c['category']}({c.get('sender_name', '')}): {c['amount']:,}원"
        for c in sorted(cats_raw, key=lambda x: x["amount"], reverse=True)
    ) or "소비 데이터 없음"

    proposals = session.get("proposals", [])
    prev_text = ""
    if proposals:
        prev_text = "\n\n이미 제안한 챌린지 (반복 금지):\n" + "\n".join(
            f"- [{p['challenge_type']}] {p['title']} (카테고리:{p['category']})"
            for p in proposals
        )

    themes_text = ", ".join(session.get("stock_themes", [])) or "없음"
    prices_text = await _fetch_prices_text()

    context = (
        f"관심 주식 테마: {themes_text}\n\n"
        f"이번 달 소비 패턴:\n{cats}"
        f"{prev_text}\n\n"
        f"사용자 요청: {req.feedback}\n\n"
        f"현재 주가:\n{prices_text}"
    )

    try:
        result = await ainvoke_structured(
            [SystemMessage(content=_ADJUST_SYSTEM), HumanMessage(content=context)],
            _AdjustAIOutput,
        )
    except Exception as e:
        logger.warning("adjust_challenge 실패: %s", e)
        result = None

    if result:
        session.setdefault("proposals", []).append({**result.model_dump(), "feedback": req.feedback})
        await save_session(req.user_id, session)
        return AdjustResponse(created_at=datetime.now(timezone.utc), **result.model_dump())

    await save_session(req.user_id, session)
    return AdjustResponse(
        created_at=datetime.now(timezone.utc),
        title="소비 줄이기",
        challenge_type=ChallengeType.COUNT,
        target=None,
        category="기타",
        description="조금 더 쉬운 목표로 다시 도전해보세요.",
        ticker="005930.KS",
        estimated_saving=0,
        challenge_sub_type="COFFEE",
    )


def get_last_proposal(session: dict) -> dict | None:
    proposals = session.get("proposals", [])
    return proposals[-1] if proposals else None


def _default_challenge_response() -> MiniChallengeResponse:
    return MiniChallengeResponse(
        created_at=datetime.now(timezone.utc),
        title="소비 줄이기",
        description="이번 달 소비를 한 번 줄여보세요.",
        category="기타",
        target=5,
        challenge_type=ChallengeType.COUNT,
        estimated_saving=0,
        ticker="005930.KS",
        challenge_sub_type="COFFEE",
    )
