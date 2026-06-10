import asyncio
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.schemas.portfolio import ProfileRequest, ProfileResponse
from app.services.agent.llm import ainvoke_structured


class _ExpenseOutput(BaseModel):
    reasoning: str = Field(
        default="",
        description="소비 패턴 분석 scratchpad. 가장 비중 높은 항목과 패턴을 1~2문장으로.",
    )
    comment: str = Field(description="사용자에게 전달할 코멘트. 120자 내외, 1~2문장.")


class _InvestOutput(BaseModel):
    reasoning: str = Field(
        default="",
        description="투자 현황 분석 scratchpad. 성향 설명 기준 기대 비중 vs 실제 비중을 1~2문장으로.",
    )
    comment: str = Field(description="사용자에게 전달할 코멘트. 70자 내외, 1문장.")


def _diagnose_expense(request: ProfileRequest) -> dict:
    """카테고리별 지출 비율 계산. LLM 없이 순수 계산."""
    total = sum(e.expense for e in request.category_expense)
    details = sorted(
        [
            {
                "name": e.name,
                "expense": e.expense,
                "ratio": round(e.expense / total * 100, 1) if total > 0 else 0.0,
            }
            for e in request.category_expense
        ],
        key=lambda x: x["expense"],
        reverse=True,
    )
    return {"details": details, "total": total}


def _diagnose_invest(request: ProfileRequest) -> dict:
    """위험/중립/안전 자산 비율 계산. LLM 없이 순수 계산."""
    total = request.assets_safe + request.assets_moderate + request.assets_risky
    if total == 0:
        return {"total": 0, "risk_ratio": 0, "moderate_ratio": 0, "safe_ratio": 0}
    risk_ratio = round(request.assets_risky * 100 / total)
    moderate_ratio = round(request.assets_moderate * 100 / total)
    safe_ratio = 100 - risk_ratio - moderate_ratio
    return {
        "total": total,
        "risk_ratio": risk_ratio,
        "moderate_ratio": moderate_ratio,
        "safe_ratio": safe_ratio,
    }


_EXPENSE_SYSTEM = (
    "당신은 사용자의 소비 패턴을 함께 들여다보는 금융 친구예요.\n"
    "사용자는 화면에서 카테고리별 비율 차트를 이미 보고 있어요.\n"
    "소비 데이터만 분석하세요 — 투자 성향 얘기는 이 섹션에서 금지예요.\n\n"
    "반드시 아래 순서로 생각하세요.\n\n"
    "1단계 — reasoning (먼저 작성)\n"
    "  · 가장 비중 높은 항목은? (30% 이상이면 높은 편)\n"
    "  · 높다면 돌아보게 하는 방향, 보통이라면 패턴 관찰\n"
    "  → 1~2문장으로 reasoning 필드에 정리\n\n"
    "2단계 — comment\n"
    "  · 비율(%)을 직접 언급하며 자연스럽게\n"
    "  · 관찰 + 소비 습관에 대한 가벼운 행동 제안으로 마무리\n"
    "  · 금지: 딱딱한 표현, 비중 높은 항목을 더 하라는 제안\n"
    "  · 말투: '~네요', '~어요' 자연스러운 대화체\n"
    "  · 120자 내외, 줄바꿈 없이 한 흐름으로\n"
    "  · 이모지·특수문자 금지"
)

_INVEST_SYSTEM = (
    "당신은 사용자의 투자 현황을 솔직하게 짚어주는 금융 친구예요.\n"
    "사용자는 화면에서 안전/중립/위험 비율 바를 이미 보고 있어요.\n\n"
    "중요: PorTI 유형 이름(영문)은 무시하고, 반드시 성향 설명 텍스트를 기준으로 판단하세요.\n\n"
    "반드시 아래 순서로 생각하세요.\n\n"
    "1단계 — reasoning (먼저 작성)\n"
    "  · 성향 설명에서 기대되는 투자 방식은?\n"
    "  · 실제 위험자산 비중이 그 성향에 맞는가, 안 맞는가?\n"
    "  → 1~2문장으로 reasoning 필드에 정리\n\n"
    "2단계 — comment\n"
    "  · 성향과 자산 비중이 얼마나 맞는지 관찰로만 전달\n"
    "  · 위험자산 비율(%)을 직접 언급\n"
    "  · 금지: '~늘려보세요', '~줄여보세요', 수익 암시, 딱딱한 한자어\n"
    "  · 금지: 자산 변경 권유 표현\n"
    "  · 말투: '~네요', '~어요', '~편이에요' 자연스러운 대화체\n"
    "  · 70자 내외, 줄바꿈 없이 한 흐름으로\n"
    "  · 이모지·특수문자 금지"
)


async def _generate_expense_comment(expense_diag: dict) -> str:
    category_lines = "\n".join(
        f"  - {c['name']}: {c['expense']:,}원 ({c['ratio']}%)"
        for c in expense_diag["details"]
    ) or "  지출 내역 없음"

    messages = [
        SystemMessage(content=_EXPENSE_SYSTEM),
        HumanMessage(content=(
            f"소비 내역 (월 평균, 비중 높은 순):\n{category_lines}\n"
            f"총 변동지출: {expense_diag['total']:,}원"
        )),
    ]
    result = await ainvoke_structured(messages, _ExpenseOutput)
    if result is None:
        return "소비 패턴을 분석하는 중 오류가 발생했어요."
    return result.comment


async def _generate_invest_comment(porti_comment: str, invest_diag: dict) -> str:
    messages = [
        SystemMessage(content=_INVEST_SYSTEM),
        HumanMessage(content=(
            f"성향 설명: {porti_comment}\n\n"
            f"실제 자산 구성:\n"
            f"  - 안전자산: {invest_diag['safe_ratio']}%\n"
            f"  - 중립자산: {invest_diag['moderate_ratio']}%\n"
            f"  - 위험자산(주식·IRP·ISA): {invest_diag['risk_ratio']}%\n"
            f"  총 자산: {invest_diag['total']:,}원"
        )),
    ]
    result = await ainvoke_structured(messages, _InvestOutput)
    if result is None:
        return "투자 현황을 분석하는 중 오류가 발생했어요."
    return result.comment


async def analyze_profile(request: ProfileRequest) -> ProfileResponse:
    expense_diag = _diagnose_expense(request)
    invest_diag = _diagnose_invest(request)

    expense_comment, invest_comment = await asyncio.gather(
        _generate_expense_comment(expense_diag),
        _generate_invest_comment(request.porti_comment, invest_diag),
    )

    return ProfileResponse(
        created_at=datetime.now(timezone.utc),
        expense_comment=expense_comment,
        invest_comment=invest_comment,
    )
