import json
import re
from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.salary import SalaryRequest, SalaryResponse, PortfolioItem, FlowItem
from app.services.agent.llm import get_llm


class SalaryRebalanceState(TypedDict):
    salary_diff: int
    current_allocations: list[dict]
    adjusted_allocations: list[dict]
    strategy_raw: str
    rebalance_comment: str


def _plan_adjustment(state: SalaryRebalanceState) -> SalaryRebalanceState:
    llm = get_llm(temperature=0.1)

    direction = "초과" if state["salary_diff"] > 0 else "결손"
    alloc_summary = "\n".join(
        f"  - {a['category']}: {a['amount']:,}원"
        for a in state["current_allocations"]
    )

    messages = [
        SystemMessage(content=(
            "당신은 월급 배분 조정 전문가입니다.\n"
            "월급 초과 또는 결손 금액을 기존 배분 비율에 맞게 조정하고 한 줄 코멘트를 작성하세요.\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            '{"rebalance_comment":"한 줄 설명"}'
        )),
        HumanMessage(content=(
            f"월급 {direction}: {abs(state['salary_diff']):,}원\n"
            f"현재 배분:\n{alloc_summary}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "strategy_raw": result.content.strip()}


def _apply_adjustment(state: SalaryRebalanceState) -> SalaryRebalanceState:
    try:
        match = re.search(r"\{.*\}", state["strategy_raw"], re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    allocations = state["current_allocations"]
    total = sum(a["amount"] for a in allocations) or 1

    adjusted = []
    for a in allocations:
        share = a["amount"] / total
        new_amount = max(0, round(a["amount"] + state["salary_diff"] * share))
        adjusted.append({**a, "amount": new_amount})

    direction = "초과" if state["salary_diff"] > 0 else "결손"
    comment = data.get(
        "rebalance_comment",
        f"월급 {direction} {abs(state['salary_diff']):,}원을 기존 비율에 맞게 조정했어요.",
    )
    return {**state, "adjusted_allocations": adjusted, "rebalance_comment": comment}


def _build_graph() -> StateGraph:
    graph = StateGraph(SalaryRebalanceState)
    graph.add_node("plan", _plan_adjustment)
    graph.add_node("apply", _apply_adjustment)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "apply")
    graph.add_edge("apply", END)
    return graph.compile()


_graph = _build_graph()


async def analyze_salary_rebalance(request: SalaryRequest) -> SalaryResponse:
    current_allocations = [
        {"asset_id": str(item.asset_id), "category": item.category, "amount": item.amount}
        for item in request.portfolio_items
    ]

    initial_state: SalaryRebalanceState = {
        "salary_diff": request.salary_diff,
        "current_allocations": current_allocations,
        "adjusted_allocations": [],
        "strategy_raw": "",
        "rebalance_comment": "",
    }

    final_state: SalaryRebalanceState = await _graph.ainvoke(initial_state)

    portfolio_items = [
        PortfolioItem(
            asset_id=UUID(a["asset_id"]),
            category=a["category"],
            amount=a["amount"],
        )
        for a in final_state["adjusted_allocations"]
    ]

    if not portfolio_items:
        portfolio_items = list(request.portfolio_items)

    return SalaryResponse(
        created_at=datetime.now(timezone.utc),
        portfolio_items=portfolio_items,
        flow_items=request.flow_items,
        rebalance_comment=final_state["rebalance_comment"],
    )
