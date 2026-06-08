import logging
from datetime import datetime, timezone
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.schemas.salary import SalaryRequest, SalaryResponse, PortfolioItem
from app.services.agent.llm import ainvoke_structured
from app.services.agent.tools import normalize_amounts

logger = logging.getLogger(__name__)


class _RebalanceCommentOutput(BaseModel):
    rebalance_comment: str


def _calculate_adjusted(current: list[dict], salary_diff: int) -> list[dict]:
    target = max(0, sum(a["amount"] for a in current) + salary_diff)
    return normalize_amounts(current, "amount", target)


async def _generate_comment(
    salary_diff: int,
    before: list[dict],
    after: list[dict],
    category_expense: list,
) -> str:
    direction = "초과" if salary_diff > 0 else "결손"

    before_map = {a["account_purpose"]: a["amount"] for a in before}
    change_summary = "\n".join(
        f"  - {a['account_purpose']}: {before_map.get(a['account_purpose'], 0):,}원 → {a['amount']:,}원"
        f" ({'+' if a['amount'] - before_map.get(a['account_purpose'], 0) >= 0 else ''}"
        f"{a['amount'] - before_map.get(a['account_purpose'], 0):,}원)"
        for a in after
    )

    expense_summary = "\n".join(
        f"  - {e.name}: {e.expense:,}원" for e in category_expense
    ) if category_expense else "  소비 내역 없음"

    messages = [
        SystemMessage(content=(
            "당신은 월급 배분 안내 전문가입니다.\n"
            "조정 전후 배분 내역을 보고, 사용자에게 변동 내용을 친근하게 설명하는 한 줄 안내 메시지를 작성하세요.\n"
            "실제 조정된 금액을 정확하게 반영해 안내하세요.\n"
            "이모지나 이모티콘은 사용하지 마세요.\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            '{"rebalance_comment":"한 줄 안내 메시지"}'
        )),
        HumanMessage(content=(
            f"월급 {direction}: {abs(salary_diff):,}원\n\n"
            f"이번 달 소비 패턴:\n{expense_summary}\n\n"
            f"조정 결과:\n{change_summary}"
        )),
    ]

    result = await ainvoke_structured(messages, _RebalanceCommentOutput)
    if result:
        return result.rebalance_comment
    return f"월급 {direction} {abs(salary_diff):,}원을 각 항목에 맞게 조정했어요."


async def analyze_salary_rebalance(request: SalaryRequest) -> SalaryResponse:
    current = [
        {"asset_id": str(item.asset_id), "account_purpose": item.account_purpose, "amount": item.amount}
        for item in request.portfolio_items
    ]

    adjusted = _calculate_adjusted(current, request.salary_diff)

    comment = await _generate_comment(
        request.salary_diff, current, adjusted, request.category_expense
    )

    return SalaryResponse(
        created_at=datetime.now(timezone.utc),
        portfolio_items=[
            PortfolioItem(asset_id=UUID(a["asset_id"]), account_purpose=a["account_purpose"], amount=a["amount"])
            for a in adjusted
        ],
        flow_items=request.flow_items,
        rebalance_comment=comment,
    )
