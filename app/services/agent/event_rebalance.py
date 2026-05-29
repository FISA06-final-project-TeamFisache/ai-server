import json
import re
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.event import (
    EventRebalanceRequest,
    EventRebalanceResponse,
    SalaryRebalanceItem,
)
from app.services.agent.llm import get_llm


class EventRebalanceState(TypedDict):
    title: str
    target_amount: int
    months_left: int
    monthly_needed: int
    salary: int
    current_invest_amount: int
    current_allocations: list[dict]
    porti_type: str
    porti_comment: str
    strategy_raw: str
    new_invest_amount: int
    new_allocations: list[dict]
    rebalance_comment: str


def _plan_rebalance(state: EventRebalanceState) -> EventRebalanceState:
    llm = get_llm(temperature=0.1)

    alloc_summary = "\n".join(
        f"  - {a['category']}: {a['amount']:,}원 ({a['account_name']})"
        for a in state["current_allocations"]
    )

    messages = [
        SystemMessage(content=(
            "당신은 이벤트 목표 자금 마련을 위한 월급 배분 전문가입니다.\n"
            "기존 월급 배분을 유지하면서, 목표 달성을 위해 매월 저축 금액을 조정하세요.\n\n"
            "규칙:\n"
            "- invest_amount: 이벤트 목표 포함 전체 투자금\n"
            "- allocations의 amount 합계 + invest_amount ≤ salary\n"
            "- 이벤트 저축은 '목표저축' 카테고리로 추가하거나 기존 항목 증액\n"
            "- 공격적 성향은 투자 비중 높게, 안정형은 저축 비중 높게\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            '{"invest_amount":600000,'
            '"allocations":[{"account_name":"계좌명","asset_id":"UUID","category":"카테고리","amount":300000}],'
            '"rebalance_comment":"한 줄 설명"}'
        )),
        HumanMessage(content=(
            f"이벤트: {state['title']}\n"
            f"목표 금액: {state['target_amount']:,}원\n"
            f"남은 기간: {state['months_left']}개월\n"
            f"매월 필요 저축: {state['monthly_needed']:,}원\n"
            f"급여: {state['salary']:,}원\n"
            f"현재 투자금: {state['current_invest_amount']:,}원\n"
            f"PorTI 유형: {state['porti_type']} — {state['porti_comment']}\n"
            f"현재 월급 배분:\n{alloc_summary}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "strategy_raw": result.content.strip()}


def _parse_plan(state: EventRebalanceState) -> EventRebalanceState:
    try:
        match = re.search(r"\{.*\}", state["strategy_raw"], re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    raw_allocs = data.get("allocations") or []
    allocations = [
        {
            "account_name": a.get("account_name", ""),
            "asset_id": a.get("asset_id", str(state["current_allocations"][0]["asset_id"])
                               if state["current_allocations"] else ""),
            "category": a.get("category", "기타"),
            "amount": int(a.get("amount", 0)),
        }
        for a in raw_allocs
    ]

    if not allocations:
        allocations = state["current_allocations"]

    return {
        **state,
        "new_invest_amount": int(data.get("invest_amount", state["current_invest_amount"])),
        "new_allocations": allocations,
        "rebalance_comment": data.get("rebalance_comment", "이벤트 목표에 맞게 재조정했어요."),
    }


def _build_graph() -> StateGraph:
    graph = StateGraph(EventRebalanceState)
    graph.add_node("plan", _plan_rebalance)
    graph.add_node("parse", _parse_plan)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "parse")
    graph.add_edge("parse", END)
    return graph.compile()


_graph = _build_graph()


async def rebalance_event(request: EventRebalanceRequest) -> EventRebalanceResponse:
    now = datetime.now(timezone.utc)
    months_left = max(1, int((request.deadline - now).days / 30))
    monthly_needed = max(0, request.target_amount // months_left)

    current_allocations = [
        {
            "account_name": item.account_name,
            "asset_id": str(item.asset_id),
            "category": item.category,
            "amount": item.amount,
        }
        for item in request.rebalance.salary_rebalance
    ]

    initial_state: EventRebalanceState = {
        "title": request.title,
        "target_amount": request.target_amount,
        "months_left": months_left,
        "monthly_needed": monthly_needed,
        "salary": request.rebalance.salary,
        "current_invest_amount": request.rebalance.invest_amount,
        "current_allocations": current_allocations,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "strategy_raw": "",
        "new_invest_amount": request.rebalance.invest_amount,
        "new_allocations": current_allocations,
        "rebalance_comment": "",
    }

    final_state: EventRebalanceState = await _graph.ainvoke(initial_state)

    salary_rebalance = []
    asset_map = {str(item.asset_id): item for item in request.rebalance.salary_rebalance}

    for alloc in final_state["new_allocations"]:
        asset_id_str = alloc.get("asset_id", "")
        matched = asset_map.get(asset_id_str)
        asset_id = matched.asset_id if matched else (
            request.rebalance.salary_rebalance[0].asset_id
            if request.rebalance.salary_rebalance else None
        )
        if asset_id is None:
            continue
        salary_rebalance.append(SalaryRebalanceItem(
            account_name=alloc["account_name"],
            asset_id=asset_id,
            category=alloc["category"],
            amount=alloc["amount"],
        ))

    if not salary_rebalance:
        salary_rebalance = list(request.rebalance.salary_rebalance)

    return EventRebalanceResponse(
        created_at=now,
        invest_amount=final_state["new_invest_amount"],
        salary_rebalance=salary_rebalance,
        rebalance_comment=final_state["rebalance_comment"],
    )
