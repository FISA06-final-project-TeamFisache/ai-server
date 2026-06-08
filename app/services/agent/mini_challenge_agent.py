from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Literal, TypeVar
from uuid import UUID

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from app.schemas.mini_challenge import (
    AdjustRequest,
    AdjustResponse,
    MiniChallengeRequest,
    MiniChallengeResponse,
    NagRequest,
    NagResponse,
)
from app.services.agent.llm import ainvoke_structured, get_llm
from app.services.agent.tools import MINI_CHALLENGE_TOOLS
from app.services.session import get_session, save_session

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_CHALLENGE_SUB_TYPES = [
    "COFFEE", "DELIVERY", "ALCOHOL", "LATE_NIGHT", "LUNCH", "SHOPPING", "TAXI"
]

_SubType = Literal["COFFEE", "DELIVERY", "ALCOHOL", "LATE_NIGHT", "LUNCH", "SHOPPING", "TAXI"]

# ── AI 출력 스키마 ────────────────────────────────────────────────────────────

class _MiniChallengeAIOutput(BaseModel):
    title: str
    description: str
    challenge_type: str
    target: int
    category: str
    estimated_saving: int
    ticker: str
    challenge_sub_type: _SubType


class _AdjustAIOutput(BaseModel):
    title: str
    description: str
    challenge_type: str
    target: int | None
    category: str
    estimated_saving: int
    ticker: str
    challenge_sub_type: _SubType


class _NagAIOutput(BaseModel):
    nag_message: str


# ── Tool 호출 헬퍼 ─────────────────────────────────────────────────────────────

async def _invoke_with_tools(
    messages: list[BaseMessage],
    output_schema: type[T],
) -> T | None:
    """LLM이 MINI_CHALLENGE_TOOLS를 자율 호출하도록 하고, 결과를 포함해 structured output 반환."""
    llm = get_llm().bind_tools(MINI_CHALLENGE_TOOLS)
    response = await llm.ainvoke(messages)

    if response.tool_calls:
        extended: list[BaseMessage] = list(messages) + [response]
        tool_map = {t.name: t for t in MINI_CHALLENGE_TOOLS}
        for tc in response.tool_calls:
            if tc["name"] in tool_map:
                try:
                    result = await tool_map[tc["name"]].ainvoke(tc["args"])
                    extended.append(
                        ToolMessage(
                            content=json.dumps(result, ensure_ascii=False, default=str),
                            tool_call_id=tc["id"],
                        )
                    )
                except Exception as e:
                    logger.warning("tool 실행 실패 (%s): %s", tc["name"], e)
        return await ainvoke_structured(extended, output_schema)

    return await ainvoke_structured(messages, output_schema)


# ── 시스템 프롬프트 ────────────────────────────────────────────────────────────

_INIT_SYSTEM = (
    "당신은 개인 재무 관리 AI 어시스턴트 Pori입니다.\n"
    "사용자의 소비 패턴을 분석해 이번 달 실천 가능한 미니 챌린지를 제안합니다.\n\n"
    "미니 챌린지는 소비 카테고리별 지출 횟수 또는 금액에 제한을 두는 형태입니다 (예: '카페 5번만 가기', '배달 음식 3만원만 쓰기').\n"
    "반드시 get_stock_prices 툴을 호출해 현재 주가를 조회한 뒤, "
    "소비 카테고리와 관심 테마에 가장 어울리는 종목 1개를 ticker로 지정하세요.\n\n"
    "반드시 아래 JSON 형식으로만 최종 응답하세요 (마크다운 코드블록 없이):\n"
    'example: {"title":"카페 5번만 가기","description":"카페 지출이 전체의 20%로 가장 높아요. 이번 달에는 5잔만 마셔봐요!",'
    '"challenge_type":"count","target":5,"category":"카페","estimated_saving":20000,"ticker":"005930.KS",'
    '"challenge_sub_type":"COFFEE"}\n\n'
    "필드 규칙:\n"
    "- challenge_type: 횟수 제한 → count / 금액 제한 → amount\n"
    "- target: 실제 목표값 (count → 목표 횟수, amount → 목표 금액(원))\n"
    "- estimated_saving: 챌린지 완수 시 예상 절약 금액(원)\n"
    "- ticker: get_stock_prices 결과에서 선택\n"
    f"- challenge_sub_type: 반드시 아래 중 하나 — {', '.join(_CHALLENGE_SUB_TYPES)}\n"
    "- 소비 비중 1위 카테고리 기준으로 제안"
)

_ADJUST_SYSTEM = (
    "당신은 개인 재무 관리 AI 어시스턴트 Pori입니다.\n"
    "사용자 피드백에 맞게 챌린지를 조정하거나 새 챌린지를 제안합니다.\n\n"
    "반드시 get_stock_prices 툴을 호출해 현재 주가를 조회한 뒤 ticker를 지정하세요.\n\n"
    "반드시 아래 JSON 형식으로만 최종 응답하세요 (마크다운 코드블록 없이):\n"
    'example: {"title":"카페 7번만 가기","description":"조금 더 쉽게, 7잔으로 줄여봐요!",'
    '"challenge_type":"count","target":7,"category":"카페","estimated_saving":12000,"ticker":"005930.KS",'
    '"challenge_sub_type":"COFFEE"}\n\n'
    "필드 규칙:\n"
    "- feedback=lower: 가장 최근에 제안한 주제, 더 쉬운 버전 (target 늘리기)\n"
    "- feedback=higher: 가장 최근에 제안한 주제, 더 어려운 버전 (target 줄이기)\n"
    "- feedback=different: 완전히 다른 카테고리 기반 새 챌린지\n"
    "- previous_proposals에 있는 챌린지 절대 반복 금지"
)

_NAG_SYSTEM = (
    "당신은 개인 재무 관리 AI 어시스턴트 Pori입니다.\n"
    "사용자의 챌린지 달성률에 맞는 친근한 잔소리성 독려 메시지를 1~2문장으로 생성하세요.\n"
    '반드시 {"nag_message": "string"} JSON 형식으로만 응답하세요.\n\n'
    "달성률별 톤:\n"
    "- 50%: 반쯤 왔다는 안도감 + 아직 갈 길 있다는 가벼운 긴장감\n"
    "- 80%: 거의 다 왔다는 흥분 + 마지막 집중력 촉구\n"
    "- 90%: 눈앞에 있다는 강한 격려 + 포기 금지 강조\n"
    "이모지 1~2개 포함, 챌린지 제목과 카테고리를 자연스럽게 언급"
)


# ── 엔드포인트 핸들러 ─────────────────────────────────────────────────────────

async def propose_mini_challenge(req: MiniChallengeRequest) -> MiniChallengeResponse:
    session = await get_session(req.user_id)

    # 매 요청마다 최신 소비 데이터·테마로 세션 갱신
    session["category_expense"] = [
        {"category": c.category, "amount": c.amount} for c in req.category_expense
    ]
    session["stock_themes"] = req.stock_themes

    cats = "\n".join(
        f"- {c.category}: {c.amount:,}원"
        for c in sorted(req.category_expense, key=lambda x: x.amount, reverse=True)
    ) or "소비 데이터 없음"
    themes_text = ", ".join(req.stock_themes) if req.stock_themes else "없음"
    context = f"관심 주식 테마: {themes_text}\n\n이번 달 소비 패턴:\n{cats}"

    messages: list[BaseMessage] = [SystemMessage(content=_INIT_SYSTEM), HumanMessage(content=context)]
    result = await _invoke_with_tools(messages, _MiniChallengeAIOutput)

    if result:
        proposal = {
            "title": result.title,
            "description": result.description,
            "challenge_type": result.challenge_type,
            "target": result.target,
            "category": result.category,
            "estimated_saving": result.estimated_saving,
            "ticker": result.ticker,
            "challenge_sub_type": result.challenge_sub_type,
            "feedback": "",
        }
        session.setdefault("proposals", []).append(proposal)
        await save_session(req.user_id, session)

        return MiniChallengeResponse(
            created_at=datetime.now(timezone.utc),
            **{k: proposal[k] for k in proposal if k != "feedback"},
        )

    await save_session(req.user_id, session)
    return _default_challenge_response()


async def adjust_challenge(req: AdjustRequest) -> AdjustResponse:
    session = await get_session(req.user_id)

    feedback_map = {
        "lower": "같은 카테고리에서 더 쉬운 버전으로 조정해주세요.",
        "higher": "같은 카테고리에서 더 어려운 버전으로 조정해주세요.",
        "different": "완전히 다른 카테고리 기반으로 새 챌린지를 제안해주세요.",
    }

    cats_raw = session.get("category_expense", [])
    cats = "\n".join(
        f"- {c['category']}: {c['amount']:,}원"
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
    fb_text = f"\n\n사용자 요청: {feedback_map.get(req.feedback, req.feedback)}"
    context = f"관심 주식 테마: {themes_text}\n\n이번 달 소비 패턴:\n{cats}{prev_text}{fb_text}"

    messages: list[BaseMessage] = [SystemMessage(content=_ADJUST_SYSTEM), HumanMessage(content=context)]
    result = await _invoke_with_tools(messages, _AdjustAIOutput)

    if result:
        proposal = {
            "title": result.title,
            "description": result.description,
            "challenge_type": result.challenge_type,
            "target": result.target,
            "category": result.category,
            "estimated_saving": result.estimated_saving,
            "ticker": result.ticker,
            "challenge_sub_type": result.challenge_sub_type,
            "feedback": req.feedback,
        }
        session.setdefault("proposals", []).append(proposal)
        await save_session(req.user_id, session)

        return AdjustResponse(
            created_at=datetime.now(timezone.utc),
            **{k: proposal[k] for k in proposal if k != "feedback"},
        )

    await save_session(req.user_id, session)
    return AdjustResponse(
        created_at=datetime.now(timezone.utc),
        title="소비 줄이기",
        challenge_type="count",
        target=None,
        category="기타",
        description="조금 더 쉬운 목표로 다시 도전해보세요.",
        ticker="005930.KS",
        estimated_saving=0,
        challenge_sub_type="COFFEE",
    )


async def generate_nag(req: NagRequest) -> NagResponse:
    target_str = f"{req.target}{'회' if req.challenge_type == 'count' else '원'}" if req.target else "-"
    current_str = f"{req.current}{'회' if req.challenge_type == 'count' else '원'}"
    context = (
        f"챌린지: {req.title} | 카테고리: {req.category}\n"
        f"목표: {target_str}, 현재: {current_str}, 달성률: {req.progress_pct}%"
    )

    messages: list[BaseMessage] = [SystemMessage(content=_NAG_SYSTEM), HumanMessage(content=context)]
    result = await ainvoke_structured(messages, _NagAIOutput)

    nag_message = result.nag_message if result else "조금만 더 힘내세요! 거의 다 왔어요."
    return NagResponse(created_at=datetime.now(timezone.utc), nag_message=nag_message)


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
        challenge_type="count",
        estimated_saving=0,
        ticker="005930.KS",
        challenge_sub_type="COFFEE",
    )
