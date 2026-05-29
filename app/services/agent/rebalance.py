import json
import re
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.portfolio import RebalanceRequest, RebalanceResponse, SalaryRebalanceItem
from app.services.agent.llm import get_llm

# 카테고리별 우선 매핑할 계좌 유형 순서
CATEGORY_ASSET_PRIORITY: dict[str, list[str]] = {
    "생활비":  ["CHECKING", "PARKING"],
    "비상금":  ["PARKING", "CMA", "CHECKING"],
    "저축":    ["SAVINGS", "DEPOSIT", "PARKING"],
    "투자":    ["STOCK", "IRP", "ISA"],
}


class RebalanceState(TypedDict):
    salary: int
    fixed_expense: int
    spendable: int
    porti_type: str
    porti_comment: str
    expense_summary: str
    asset_types: str
    strategy_raw: str
    invest_amount: int
    allocations: list[dict]   # [{category, ratio}]


def _plan_rebalance(state: RebalanceState) -> RebalanceState:
    llm = get_llm(temperature=0.1)
    messages = [
        SystemMessage(content=(
            "당신은 월급 배분 전문가입니다. 사용자의 급여와 소비 패턴을 분석해 최적의 월급 배분 계획을 세우세요.\n\n"
            "규칙:\n"
            "- invest_amount: 투자로 별도 운용할 금액 (원 단위 정수)\n"
            "- allocations: 나머지 가처분 소득을 생활비·비상금·저축으로 나누는 비율 (salary 기준 %)\n"
            "- allocations의 ratio 합계 + invest_amount/salary*100 ≤ 100\n"
            "- 공격적 성향일수록 invest_amount 비중 높게, 안정 성향은 낮게\n\n"
            "반드시 아래 JSON 형식으로만 응답하세요:\n"
            '{"invest_amount": 1200000, "allocations": ['
            '{"category": "생활비", "ratio": 40},'
            '{"category": "비상금", "ratio": 10},'
            '{"category": "저축", "ratio": 12}'
            "]}"
        )),
        HumanMessage(content=(
            f"급여: {state['salary']:,}원\n"
            f"고정지출(이미 별도 처리됨): {state['fixed_expense']:,}원\n"
            f"가처분 소득: {state['spendable']:,}원\n"
            f"PorTI 유형: {state['porti_type']} — {state['porti_comment']}\n"
            f"최근 3개월 평균 변동지출: {state['expense_summary']}\n"
            f"보유 계좌 유형: {state['asset_types']}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "strategy_raw": result.content.strip()}


def _parse_plan(state: RebalanceState) -> RebalanceState:
    try:
        match = re.search(r"\{.*\}", state["strategy_raw"], re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    spendable = state["spendable"]
    invest_amount = int(data.get("invest_amount", round(spendable * 0.3)))
    allocations = data.get("allocations", [
        {"category": "생활비", "ratio": 45},
        {"category": "비상금", "ratio": 10},
        {"category": "저축", "ratio": 10},
    ])

    return {**state, "invest_amount": invest_amount, "allocations": allocations}


def _build_graph() -> StateGraph:
    graph = StateGraph(RebalanceState)
    graph.add_node("plan", _plan_rebalance)
    graph.add_node("parse", _parse_plan)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "parse")
    graph.add_edge("parse", END)
    return graph.compile()


_graph = _build_graph()


def _match_asset(category: str, asset_list: list[dict], used_ids: set) -> dict | None:
    priority = CATEGORY_ASSET_PRIORITY.get(category, [])
    for asset_type in priority:
        asset = next(
            (a for a in asset_list if a["asset_type"] == asset_type and a["asset_id"] not in used_ids),
            None,
        )
        if asset:
            return asset
    # 우선순위 매칭 실패 시 미사용 계좌 중 첫 번째
    return next((a for a in asset_list if a["asset_id"] not in used_ids), None)


async def rebalance_salary(request: RebalanceRequest) -> RebalanceResponse:
    spendable = request.salary - request.fixed_expense

    expense_summary = ", ".join(
        f"{e.name} {e.expense:,}원" for e in request.category_expense
    ) or "데이터 없음"

    asset_list = [
        {"asset_id": str(a.asset_id), "asset_type": a.asset_type, "account_name": a.account_name}
        for a in request.assets
    ]
    asset_types = ", ".join(sorted({a["asset_type"] for a in asset_list})) or "없음"

    initial_state: RebalanceState = {
        "salary": request.salary,
        "fixed_expense": request.fixed_expense,
        "spendable": spendable,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "expense_summary": expense_summary,
        "asset_types": asset_types,
        "strategy_raw": "",
        "invest_amount": 0,
        "allocations": [],
    }

    final_state: RebalanceState = await _graph.ainvoke(initial_state)

    salary_rebalance: list[SalaryRebalanceItem] = []
    used_ids: set[str] = set()

    for alloc in final_state["allocations"]:
        category = alloc.get("category", "기타")
        ratio = int(alloc.get("ratio", 0))
        if ratio <= 0:
            continue
        matched = _match_asset(category, asset_list, used_ids)
        if matched:
            used_ids.add(matched["asset_id"])
            salary_rebalance.append(SalaryRebalanceItem(
                asset_id=matched["asset_id"],
                category=category,
                ratio=ratio,
            ))

    return RebalanceResponse(
        created_at=datetime.now(timezone.utc),
        invest_amount=final_state["invest_amount"],
        salary_rebalance=salary_rebalance,
    )
