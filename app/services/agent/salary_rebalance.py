import logging
from datetime import datetime, timezone
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.schemas.salary import SalaryRequest, SalaryResponse, PortfolioItem, FlowItem
from app.services.agent.llm import ainvoke_structured

logger = logging.getLogger(__name__)


class _ItemRatio(BaseModel):
    asset_id: str
    ratio: float  # 0.0 ~ 1.0, 전체 합계 = 1.0

class _AllocationOutput(BaseModel):
    item_ratios: list[_ItemRatio]
    rebalance_comment: str

def _apply_ratios(
    portfolio: list[dict],
    flows: list[dict],
    ratio_map: dict[str, float],
    salary_diff: int,
) -> tuple[list[dict], list[dict]]:
    """ratio × salary_diff를 각 항목에 개별 적용. 반올림 오차는 최대 비율 항목에 흡수."""
    all_items = portfolio + flows
    deltas: dict[str, int] = {v["asset_id"]: round(ratio_map.get(v["asset_id"], 0.0) * salary_diff) for v in all_items}

    remainder = salary_diff - sum(deltas.values())
    if remainder != 0 and ratio_map:
        top_id = max(ratio_map, key=lambda k: ratio_map[k])
        deltas[top_id] = deltas.get(top_id, 0) + remainder

    # 결손금: 잔액 초과 차감 방지 — 미처리 결손금은 잔액 있는 항목에 순차 재분배
    if salary_diff < 0:
        amount_by_id = {v["asset_id"]: v["amount"] for v in all_items}
        overflow = 0
        for aid in list(deltas):
            if amount_by_id[aid] + deltas[aid] < 0:
                overflow += -(amount_by_id[aid] + deltas[aid])
                deltas[aid] = -amount_by_id[aid]

        if overflow > 0:
            for v in sorted(all_items, key=lambda x: -(amount_by_id[x["asset_id"]] + deltas.get(x["asset_id"], 0))):
                if overflow <= 0:
                    break
                aid = v["asset_id"]
                available = amount_by_id[aid] + deltas.get(aid, 0)
                if available <= 0:
                    continue
                take = min(overflow, available)
                deltas[aid] -= take
                overflow -= take

    def apply(items: list[dict]) -> list[dict]:
        return [{**v, "amount": max(0, v["amount"] + deltas.get(v["asset_id"], 0))} for v in items]

    return apply(portfolio), apply(flows)


async def analyze_salary_rebalance(request: SalaryRequest) -> SalaryResponse:
    portfolio_current = [
        {"asset_id": str(p.asset_id), "account_purpose": p.account_purpose, "amount": p.amount}
        for p in request.portfolio_items
    ]
    flow_current = [
        {"asset_id": str(f.asset_id), "title": f.title, "term": f.term,
         "summary": f.summary, "amount": f.amount}
        for f in request.flow_items
    ]
    all_items = portfolio_current + flow_current

    expense_summary = "\n".join(
        f"  - {e.name}: {e.expense:,}원" for e in request.category_expense
    ) or "  소비 내역 없음"

    portfolio_desc = "\n".join(
        f"  - [asset_id: {p['asset_id']}] {p['account_purpose']}: {p['amount']:,}원"
        for p in portfolio_current
    ) or "  없음"

    flow_desc = "\n".join(
        f"  - [asset_id: {f['asset_id']}] {f['title']}({f['term']}), {f['summary']}: {f['amount']:,}원"
        for f in flow_current
    ) or "  없음"

    direction = "잉여금" if request.salary_diff > 0 else "결손금"
    all_ids = [v["asset_id"] for v in all_items]

    messages = [
        SystemMessage(content=(
            "당신은 월급 배분 전문가입니다.\n"
            "월급 변동액을 각 항목의 성격과 소비 패턴을 고려해 항목별로 배분하세요.\n\n"
            "판단 기준:\n"
            "- 잉여금(+): 투자 포트폴리오(flow_items) 비중을 높이는 방향을 우선 검토하세요.\n"
            "- 결손금(-): 생활비·비상금 같은 필수 항목보다 저축·투자 항목을 먼저 줄이세요.\n"
            "- account_purpose, title, summary, term을 참고해 맥락에 맞게 판단하세요.\n"
            "- 소비 지출이 많은 달(식비·쇼핑 등 증가)이면 생활비 비율을 높이고 투자 비율을 낮추세요.\n"
            "- 소비가 적은 달이면 투자 비율을 더 높여도 좋습니다.\n"
            "- 특정 카테고리 지출이 급증한 경우 해당 목적의 portfolio_item 비율을 우선 보호하세요.\n\n"
            "출력 규칙:\n"
            "- item_ratios에 제공된 모든 asset_id를 빠짐없이 포함하세요.\n"
            "- ratio는 0.0~1.0 사이이며, 모든 ratio의 합계는 반드시 1.0이어야 합니다.\n"
            "- 변동시키지 않을 항목은 ratio를 0.0으로 설정하세요.\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            '{"item_ratios": [{"asset_id": "...", "ratio": 0.0}, ...], "rebalance_comment": "<항목별 배분 이유 설명>"}\n\n'
            "rebalance_comment 작성 기준:\n"
            "- ratio가 0보다 큰 항목마다 배분 이유를 한 줄씩 자연스러운 문장으로 설명하세요. 예: '장기ETF에 60%를 배분했어요. 소비가 적은 달이라 투자 비중을 높였어요.'\n"
            "- 항목 이름은 title 또는 account_purpose를 사용하세요. asset_id는 쓰지 마세요.\n"
            "- 마지막에 전체 배분 방향을 한 문장으로 요약하세요."
        )),
        HumanMessage(content=(
            f"월급 {direction}: {request.salary_diff:+,}원\n\n"
            f"이번 달 소비 패턴:\n{expense_summary}\n\n"
            f"portfolio_items (생활비·저축·비상금 등):\n{portfolio_desc}\n\n"
            f"flow_items (투자 포트폴리오):\n{flow_desc}\n\n"
            f"배분 대상 asset_id 목록: {all_ids}"
        )),
    ]

    result = await ainvoke_structured(messages, _AllocationOutput)

    valid = (
        result is not None
        and {r.asset_id for r in result.item_ratios} == {v["asset_id"] for v in all_items}
        and all(0.0 <= r.ratio <= 1.0 for r in result.item_ratios)
        and abs(sum(r.ratio for r in result.item_ratios) - 1.0) < 0.05
    )

    if valid:
        total = sum(r.ratio for r in result.item_ratios)
        ratio_map = {r.asset_id: r.ratio / total for r in result.item_ratios}
        comment = result.rebalance_comment
    else:
        # 폴백: 잉여금·결손금 모두 flow(투자) 우선 — 프롬프트 지침과 일치
        targets = flow_current if flow_current else portfolio_current
        if not targets:
            ratio_map = {}
            comment = f"월급 {direction} {abs(request.salary_diff):,}원 — 배분 대상 없음"
        else:
            fallback_ids = {v["asset_id"] for v in targets}
            ratio_map = {
                v["asset_id"]: (1.0 / len(targets) if v["asset_id"] in fallback_ids else 0.0)
                for v in all_items
            }
            comment = f"월급 {direction} {abs(request.salary_diff):,}원을 조정했어요."
        logger.warning("[analyze_salary_rebalance] LLM 응답 불일치 — 폴백 적용")

    adj_portfolio, adj_flows = _apply_ratios(portfolio_current, flow_current, ratio_map, request.salary_diff)

    return SalaryResponse(
        created_at=datetime.now(timezone.utc),
        portfolio_items=[
            PortfolioItem(asset_id=UUID(a["asset_id"]), account_purpose=a["account_purpose"], amount=a["amount"])
            for a in adj_portfolio
        ],
        flow_items=[
            FlowItem(asset_id=UUID(f["asset_id"]), title=f["title"], term=f["term"],
                     summary=f["summary"], amount=f["amount"])
            for f in adj_flows
        ],
        rebalance_comment=comment,
    )
