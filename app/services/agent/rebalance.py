import json
import re
from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.schemas.portfolio import RebalanceRequest, RebalanceResponse, SalaryRebalanceItem
from app.services.agent.llm import get_llm


class RebalanceState(TypedDict):
    salary: int
    fixed_expense: int
    spendable: int
    porti_type: str
    porti_comment: str
    expense_summary: str
    asset_list: list[dict]   # [{asset_id, asset_type, account_name}]
    strategy_raw: str
    invest_amount: int
    allocations: list[dict]  # [{asset_id, category, ratio}]


def _plan_rebalance(state: RebalanceState) -> RebalanceState:
    llm = get_llm(temperature=0.1)

    asset_lines = "\n".join(
        f"  - asset_id: {a['asset_id']}, {a['account_name']} ({a['asset_type']})"
        for a in state["asset_list"]
    ) or "  보유 계좌 없음"

    messages = [
        SystemMessage(content=(
            "당신은 월급 배분 전문가입니다. 사용자의 급여와 소비 패턴을 분석해 최적의 월급 배분 계획을 세우세요.\n\n"
            "규칙:\n"
            "- invest_amount: 투자로 별도 운용할 금액 (원 단위 정수)\n"
            "- allocations: 나머지 가처분 소득을 계좌별로 나누는 비율 (salary 기준 %)\n"
            "- allocations의 ratio 합계 + invest_amount/salary*100 ≤ 100\n"
            "- 각 allocation의 asset_id는 반드시 아래 보유 계좌 목록에 있는 값을 사용하세요\n"
            "- 공격적 성향일수록 invest_amount 비중 높게, 안정 성향은 낮게\n"
            "- 계좌 용도에 맞게 배분하세요 (CHECKING→생활비, SAVINGS→저축, PARKING→비상금 등)\n"
            "- 이모지나 이모티콘은 사용하지 마세요\n\n"
            "반드시 아래 JSON 형식으로만 응답하세요:\n"
            '{"invest_amount": 1200000, "allocations": [\n'
            '  {"asset_id": "<실제 asset_id>", "category": "생활비", "ratio": 40},\n'
            '  {"asset_id": "<실제 asset_id>", "category": "저축", "ratio": 12}\n'
            "]}"
        )),
        HumanMessage(content=(
            f"급여: {state['salary']:,}원\n"
            f"고정지출(이미 별도 처리됨): {state['fixed_expense']:,}원\n"
            f"가처분 소득: {state['spendable']:,}원\n"
            f"PorTI 유형: {state['porti_type']} — {state['porti_comment']}\n"
            f"최근 3개월 평균 변동지출: {state['expense_summary']}\n\n"
            f"보유 계좌 목록:\n{asset_lines}"
        )),
    ]
    result = llm.invoke(messages)
    return {**state, "strategy_raw": result.content.strip()}


def _parse_plan(state: RebalanceState) -> RebalanceState:
    valid_ids = {a["asset_id"] for a in state["asset_list"]}

    try:
        match = re.search(r"\{.*\}", state["strategy_raw"], re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    invest_amount = int(data.get("invest_amount", round(state["spendable"] * 0.3)))

    raw_allocations = data.get("allocations", [])

    # asset_id가 실제 보유 계좌에 있는 것만 통과
    allocations = [
        alloc for alloc in raw_allocations
        if alloc.get("asset_id") in valid_ids and int(alloc.get("ratio", 0)) > 0
    ]

    # LLM이 asset_id를 제대로 못 썼을 경우 폴백: 순서대로 계좌 배정
    if not allocations and raw_allocations and state["asset_list"]:
        for i, alloc in enumerate(raw_allocations):
            if i >= len(state["asset_list"]):
                break
            ratio = int(alloc.get("ratio", 0))
            if ratio <= 0:
                continue
            allocations.append({
                "asset_id": state["asset_list"][i]["asset_id"],
                "category": alloc.get("category", "기타"),
                "ratio": ratio,
            })

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


async def rebalance_salary(request: RebalanceRequest) -> RebalanceResponse:
    spendable = request.salary - request.fixed_expense

    expense_summary = ", ".join(
        f"{e.name} {e.expense:,}원" for e in request.category_expense
    ) or "데이터 없음"

    asset_list = [
        {"asset_id": str(a.asset_id), "asset_type": a.asset_type, "account_name": a.account_name}
        for a in request.assets
    ]

    initial_state: RebalanceState = {
        "salary": request.salary,
        "fixed_expense": request.fixed_expense,
        "spendable": spendable,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "expense_summary": expense_summary,
        "asset_list": asset_list,
        "strategy_raw": "",
        "invest_amount": 0,
        "allocations": [],
    }

    final_state: RebalanceState = await _graph.ainvoke(initial_state)

    salary_rebalance: list[SalaryRebalanceItem] = [
        SalaryRebalanceItem(
            asset_id=UUID(alloc["asset_id"]),
            category=alloc["category"],
            amount=round(request.salary * int(alloc["ratio"]) / 100),
        )
        for alloc in final_state["allocations"]
    ]

    return RebalanceResponse(
        created_at=datetime.now(timezone.utc),
        invest_amount=final_state["invest_amount"],
        salary_rebalance=salary_rebalance,
    )
