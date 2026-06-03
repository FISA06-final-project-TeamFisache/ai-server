from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.challenge import ChallengeRequest, ChallengeResponse, _ChallengeAIOutput
from app.services.agent.llm import ainvoke_structured

logger = logging.getLogger(__name__)

_SYSTEM = (
    "사용자의 소비 패턴을 분석해 이번 달 실천 가능한 미니 챌린지를 제안합니다.\n\n"
    "반드시 아래 JSON 형식으로만 응답하세요 (마크다운 코드블록 없이 순수 JSON):\n"
    '{"icon":"☕","title":"카페 5번 줄이기","difficulty":2,"reasoning":"카페 지출이 전체의 20%를 차지해 절약 여지가 가장 커요. 이번 달 5번만 참으면 5만원을 모을 수 있고, 집에서 커피 만드는 습관도 생겨요.","target_count":5,"last_month_count":9,"estimated_saving":50000}\n\n'
    "필드 설명:\n"
    "- target_count: 제목의 숫자와 반드시 일치 (줄이고자 하는 횟수)\n"
    "- last_month_count: 지난달 해당 카테고리 추정 이용 횟수 (소비 금액 ÷ 평균 단가로 추정, 임시값)\n"
    "  예: 카페 월 지출 18만원 → 아메리카노 4천원 기준 약 45회지만 소비 단위가 다를 수 있으니 합리적 범위로 추정\n"
    "  target_count보다 반드시 커야 함 (last_month_count > target_count)\n"
    "- reasoning: 반드시 2문장. 첫 문장 소비 데이터 근거, 두 번째 문장 달성 효과·행동 조언\n\n"
    "feedback 종류:\n"
    "- null: 소비 비중 1위 카테고리 기준 첫 제안, difficulty=2\n"
    "- lower: 현재 제안과 같은 주제에서 더 쉬운 버전 (target_count 감소, difficulty 낮게)\n"
    "- higher: 현재 제안과 같은 주제에서 더 어려운 버전 (target_count 증가, difficulty 높게)\n"
    "- different: 기존 제안과 다른 카테고리 기반 완전히 새로운 챌린지\n\n"
    "공통 규칙:\n"
    "- previous_proposals에 있는 챌린지는 절대 반복하지 않음\n"
    "- target_count는 제목의 숫자와 반드시 일치 (예: '카페 5번 줄이기' → target_count=5)\n"
    "- difficulty와 target_count 기준:\n"
    "  * 1(쉬움): target_count=2~3\n"
    "  * 2(보통): target_count=4~5\n"
    "  * 3(어려움): target_count=7~10\n"
    "- estimated_saving: 해당 카테고리 월 지출액 기준 현실적인 절약 예상액"
)


async def propose_challenge(req: ChallengeRequest) -> ChallengeResponse:
    categories_text = "\n".join(
        f"- {c.categoryName}: {c.expenseAmount:,}원 ({c.percentage}%)"
        for c in sorted(req.consumption_categories, key=lambda x: x.percentage, reverse=True)
    ) or "소비 데이터 없음 — 일반적인 절약 챌린지를 제안해주세요."

    prev_text = ""
    if req.previous_proposals:
        prev_text = "\n\n이미 제안한 챌린지 (반복 금지):\n" + "\n".join(
            f"- {p.get('title', '')} (난이도: {p.get('difficulty', '')})"
            for p in req.previous_proposals
        )

    current_text = ""
    if req.current_proposal and req.feedback in ("lower", "higher"):
        cp = req.current_proposal
        current_text = (
            f"\n\n현재 제안 (같은 주제에서 난이도만 {'낮춰' if req.feedback == 'lower' else '높여'} 주세요):\n"
            f"- 제목: {cp.get('title', '')}\n"
            f"- 난이도: {cp.get('difficulty', '')}\n"
            f"- 달성 횟수: {cp.get('target_count', '')}"
        )

    feedback_text = ""
    if req.feedback == "different":
        feedback_text = "\n\n사용자가 챌린지 주제를 거절했습니다. 다른 소비 카테고리 기반으로 새 챌린지를 제안하세요."

    context = f"이번 달 소비 패턴:\n{categories_text}{prev_text}{current_text}{feedback_text}"

    ai = await ainvoke_structured(
        [SystemMessage(content=_SYSTEM), HumanMessage(content=context)],
        _ChallengeAIOutput,
        temperature=0.7,
        max_tokens=512,
    )

    if ai is None:
        raise ValueError("LLM 응답에서 JSON 파싱 실패 — 모델 응답을 확인하세요")

    last_month = max(ai.target_count + 1, ai.last_month_count)  # target_count보다 반드시 크게

    return ChallengeResponse(
        icon=ai.icon,
        title=ai.title,
        difficulty=ai.difficulty,
        reasoning=ai.reasoning,
        target_count=ai.target_count,
        last_month_count=last_month,
        step_size=round(100 / max(1, ai.target_count)),
        estimated_saving=ai.estimated_saving,
    )
