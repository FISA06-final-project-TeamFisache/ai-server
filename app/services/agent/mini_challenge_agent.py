from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.schemas.mini_challenge import (
    AdjustRequest,
    AdjustResponse,
    MiniChallengeRequest,
    MiniChallengeResponse,
    NagRequest,
    NagResponse,
)
from app.services.agent.llm import ainvoke_structured

import logging

logger = logging.getLogger(__name__)

_TICKER_MAP = {
    "카페": "005930.KS", "커피": "005930.KS",
    "식비": "035720.KS", "외식": "035720.KS", "배달": "035720.KS",
    "쇼핑": "035420.KS", "온라인": "035420.KS", "패션": "035420.KS",
    "교통": "005380.KS",
    "구독": "360750.KS", "OTT": "360750.KS",
}
_DEFAULT_TICKER = "005930.KS"
_TICKER_HINT = (
    "추천 종목 코드 목록:\n"
    "- 삼성전자: 005930.KS\n- NAVER: 035420.KS\n- 카카오: 035720.KS\n"
    "- 현대차: 005380.KS\n- SK하이닉스: 000660.KS\n- TIGER 미국S&P500: 360750.KS\n"
    "카테고리에 가장 어울리는 종목 1개 선택"
)
_CHALLENGE_SUB_TYPES = [
    "COFFEE",      # 카페
    "DELIVERY",    # 배달 (식비 + 배달앱)
    "ALCOHOL",     # 술 (식비 + 주점)
    "LATE_NIGHT",  # 야식 (식비 + 23:00~04:00)
    "LUNCH",       # 점심 (식비 + 11:00~14:00)
    "SHOPPING",    # 쇼핑 (쇼핑 + 특정 앱)
    "TAXI"         # 택시 (교통 + 택시)
]

# ── 내부 AI 출력 스키마 ───────────────────────────────────────────────────────

class _MiniChallengeAIOutput(BaseModel):
    title: str
    description: str
    challenge_type: str
    target: int
    category: str
    estimated_saving: int
    ticker: str
    challenge_sub_type: str


class _AdjustAIOutput(BaseModel):
    title: str
    description: str
    challenge_type: str
    target: int | None
    category: str
    estimated_saving: int
    ticker: str
    challenge_sub_type: str


class _NagAIOutput(BaseModel):
    nag_message: str


# ── 초기 제안 ─────────────────────────────────────────────────────────────────

_INIT_SYSTEM = (
    "당신은 개인 재무 관리 AI 어시스턴트 Pori입니다.\n"
    "사용자의 소비 패턴을 분석해 이번 달 실천 가능한 미니 챌린지를 제안합니다.\n\n"
    "반드시 아래 JSON 형식으로만 응답하세요 (마크다운 코드블록 없이):\n"
    '{"title":"카페 5번 줄이기","description":"카페 지출이 전체의 20%로 가장 높아요. 5번만 참으면 약 2만원 절약!",'
    '"challenge_type":"count","target":5,"category":"카페","estimated_saving":20000,"ticker":"005930.KS"}\n\n'
    "필드 규칙:\n"
    "- challenge_type: 횟수 줄이기 류 → count / 금액 줄이기 류 → amount\n"
    "- target: 실제 목표값 (count → 목표 횟수, amount → 목표 절약 금액(원))\n"
    "- estimated_saving: 챌린지 완수 시 예상 절약 금액 (원)\n"
    f"- {_TICKER_HINT}\n"
    "- challenge_sub_type: 챌린지 세부 유형. 반드시 아래 값 중 하나로 지정\n"
    f"- {', '.join(_CHALLENGE_SUB_TYPES)}\n"
    "- 소비 비중 1위 카테고리 기준으로 제안"
)

# ── 조정 ─────────────────────────────────────────────────────────────────────

_ADJUST_SYSTEM = (
    "당신은 개인 재무 관리 AI 어시스턴트 Pori입니다.\n"
    "사용자 피드백에 맞게 챌린지를 조정하거나 새 챌린지를 제안합니다.\n\n"
    "반드시 아래 JSON 형식으로만 응답하세요 (마크다운 코드블록 없이):\n"
    '{"title":"카페 3번 줄이기","description":"조금 더 쉽게, 3번만 참아봐요!",'
    '"challenge_type":"count","target":3,"category":"카페","estimated_saving":12000,"ticker":"005930.KS"}\n\n'
    "필드 규칙:\n"
    "- target: 실제 목표값 (count → 목표 횟수, amount → 목표 절약 금액(원))\n"
    "- feedback=lower: 같은 카테고리, 더 쉬운 버전 (target 줄이기)\n"
    "- feedback=higher: 같은 카테고리, 더 어려운 버전 (target 늘리기)\n"
    "- feedback=different: 완전히 다른 카테고리 기반 새 챌린지\n"
    "- previous_proposals에 있는 챌린지 절대 반복 금지\n"
    f"- {_TICKER_HINT}"
)

# ── nag ──────────────────────────────────────────────────────────────────────

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


def _ticker(data_ticker: str, category: str) -> str:
    if data_ticker and ".KS" in data_ticker:
        return data_ticker
    return next((v for k, v in _TICKER_MAP.items() if k in category), _DEFAULT_TICKER)


async def propose_mini_challenge(req: MiniChallengeRequest) -> MiniChallengeResponse:
    cats = "\n".join(
        f"- {c.category}: {c.amount:,}원"
        for c in sorted(req.category_expense, key=lambda x: x.amount, reverse=True)
    ) or "소비 데이터 없음"

    themes_text = ", ".join(req.stock_themes) if req.stock_themes else "없음"
    context = f"관심 주식 테마: {themes_text}\n\n이번 달 소비 패턴:\n{cats}"

    messages = [SystemMessage(content=_INIT_SYSTEM), HumanMessage(content=context)]
    result = await ainvoke_structured(messages, _MiniChallengeAIOutput)

    if result:
        return MiniChallengeResponse(
            created_at=datetime.now(timezone.utc),
            title=result.title,
            description=result.description,
            category=result.category,
            target=result.target,
            challenge_type=result.challenge_type,
            estimated_saving=result.estimated_saving,
            ticker=_ticker(result.ticker, result.category),
            challenge_sub_type=result.challenge_sub_type,
        )

    return MiniChallengeResponse(
        created_at=datetime.now(timezone.utc),
        title="소비 줄이기",
        description="이번 달 소비를 한 번 줄여보세요.",
        category="기타",
        target=5,
        challenge_type="count",
        estimated_saving=0,
        ticker=_DEFAULT_TICKER,
        challenge_sub_type="COFFEE",
    )


async def adjust_challenge(req: AdjustRequest) -> AdjustResponse:
    cats = "\n".join(
        f"- {c.category}: {c.amount:,}원"
        for c in sorted(req.category_expense, key=lambda x: x.amount, reverse=True)
    ) or "소비 데이터 없음"

    prev_text = ""
    if req.previous_proposals:
        prev_text = "\n\n이미 제안한 챌린지 (반복 금지):\n" + "\n".join(
            f"- [{p.challenge_type}] {p.title} (카테고리:{p.category}, 피드백:{p.feedback})"
            for p in req.previous_proposals
        )

    feedback_map = {
        "lower": "같은 카테고리에서 더 쉬운 버전으로 조정해주세요.",
        "higher": "같은 카테고리에서 더 어려운 버전으로 조정해주세요.",
        "different": "완전히 다른 카테고리 기반으로 새 챌린지를 제안해주세요.",
    }
    last_feedback = req.previous_proposals[-1].feedback if req.previous_proposals else ""
    fb_text = f"\n\n사용자 요청: {feedback_map.get(last_feedback, '')}" if last_feedback else ""

    themes_text = ", ".join(req.stock_themes) if req.stock_themes else "없음"
    context = f"관심 주식 테마: {themes_text}\n\n이번 달 소비 패턴:\n{cats}{prev_text}{fb_text}"

    messages = [SystemMessage(content=_ADJUST_SYSTEM), HumanMessage(content=context)]
    result = await ainvoke_structured(messages, _AdjustAIOutput)

    if result:
        return AdjustResponse(
            created_at=datetime.now(timezone.utc),
            title=result.title,
            challenge_type=result.challenge_type,
            target=result.target,
            category=result.category,
            description=result.description,
            ticker=_ticker(result.ticker, result.category),
            estimated_saving=result.estimated_saving,
            challenge_sub_type=result.challenge_sub_type,
        )

    return AdjustResponse(
        created_at=datetime.now(timezone.utc),
        title="소비 줄이기",
        challenge_type="count",
        target=None,
        category="기타",
        description="조금 더 쉬운 목표로 다시 도전해보세요.",
        ticker=_DEFAULT_TICKER,
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

    messages = [SystemMessage(content=_NAG_SYSTEM), HumanMessage(content=context)]
    result = await ainvoke_structured(messages, _NagAIOutput)

    nag_message = result.nag_message if result else "조금만 더 힘내세요! 거의 다 왔어요."
    return NagResponse(created_at=datetime.now(timezone.utc), nag_message=nag_message)
