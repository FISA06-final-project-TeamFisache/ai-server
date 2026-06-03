from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.schemas.portfolio import RebalanceRequest, RebalanceResponse, SalaryRebalanceItem
from app.services.agent.llm import get_llm, invoke_structured


class _AllocationItem(BaseModel):
    asset_id: str = Field(default="", description="보유 계좌 목록에 있는 실제 asset_id")
    category: str = Field(default="기타", description="배분 용도 (예: 생활비, 저축, 비상금)")
    ratio: int = Field(default=0, description="급여 대비 비율(%)")


class _RebalancePlan(BaseModel):
    invest_amount: int = Field(default=0, description="투자로 별도 운용할 금액(원)")
    allocations: list[_AllocationItem] = Field(default_factory=list, description="계좌별 배분 목록")


class RebalanceState(TypedDict):
    salary: int
    fixed_expense: int
    spendable: int
    porti_type: str
    porti_comment: str
    expense_summary: str
    asset_list: list[dict]
    invest_amount: int
    allocations: list[dict]


_SYSTEM = (
    "당신은 월급 배분 전문가입니다. 급여와 소비 패턴을 분석해 최적의 월급 배분 계획을 세우세요.\n\n"
    "규칙:\n"
    "- invest_amount: 투자로 별도 운용할 금액 (원 단위 정수, 절대 금액)\n"
    "- allocations: 각 계좌에 배분할 비율 목록\n"
    "  - asset_id: 아래 보유 계좌 목록에 있는 실제 UUID 값 (절대 변경 금지)\n"
    "  - category: 계좌 용도를 나타내는 한글 레이블 (예: 생활비, 저축, 비상금, 교통)\n"
    "              ※ CHECKING/SAVINGS/PARKING 같은 영문 코드가 아닌 반드시 한글로 작성\n"
    "  - ratio: 급여 대비 배분 비율(%, 정수)\n"
    "- 같은 asset_id는 한 번만 사용하세요 (중복 배정 금지)\n"
    "- allocations ratio 합계 + invest_amount/salary×100 ≤ 100\n"
    "- 공격적 성향: invest_amount 비중 높게 / 안정형: 낮게\n"
    "- 이모지나 이모티콘은 사용하지 마세요.\n\n"
    "예시 응답 (급여 4,800,000원 / 균형형 성향):\n"
    '{"invest_amount": 1100000, "allocations": [\n'
    '  {"asset_id": "실제-uuid-여기에", "category": "생활비", "ratio": 40},\n'
    '  {"asset_id": "실제-uuid-여기에", "category": "저축",   "ratio": 20},\n'
    '  {"asset_id": "실제-uuid-여기에", "category": "비상금", "ratio": 10}\n'
    "]}"
)


def _plan_rebalance(state: RebalanceState) -> RebalanceState:
    asset_lines = "\n".join(
        f"  - asset_id: {a['asset_id']}, {a['account_name']} ({a['asset_type']})"
        for a in state["asset_list"]
    ) or "  보유 계좌 없음"

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=(
            f"급여: {state['salary']:,}원\n"
            f"고정지출(별도 처리됨): {state['fixed_expense']:,}원\n"
            f"가처분 소득: {state['spendable']:,}원\n"
            f"PorTI 유형: {state['porti_type']} — {state['porti_comment']}\n"
            f"최근 3개월 평균 변동지출: {state['expense_summary']}\n\n"
            f"보유 계좌 목록:\n{asset_lines}"
        )),
    ]
    result = invoke_structured(messages, _RebalancePlan, temperature=0.1)
    if result is None:
        return {
            **state,
            "invest_amount": round(state["spendable"] * 0.25),
            "allocations": [],
        }
    try:
        valid_ids = {a["asset_id"] for a in state["asset_list"]}

        # asset_id 검증: 실제 보유 계좌에 없는 항목 필터 + 중복 제거
        seen_ids: set[str] = set()
        allocations = []
        for a in result.allocations:
            if a.asset_id in valid_ids and a.ratio > 0 and a.asset_id not in seen_ids:
                allocations.append({"asset_id": a.asset_id, "category": a.category, "ratio": a.ratio})
                seen_ids.add(a.asset_id)

        # LLM이 asset_id를 잘못 썼을 때 폴백: 순서대로 계좌 배정
        if not allocations and result.allocations and state["asset_list"]:
            for i, a in enumerate(result.allocations):
                if i >= len(state["asset_list"]):
                    break
                if a.ratio > 0:
                    allocations.append({
                        "asset_id": state["asset_list"][i]["asset_id"],
                        "category": a.category,
                        "ratio": a.ratio,
                    })

        invest_amount = result.invest_amount
        salary = state["salary"]

        # invest_amount가 급여 초과 시 보정
        if invest_amount >= salary:
            invest_amount = round(state["spendable"] * 0.25)

        # ratio 합계 초과 시 비례 축소
        invest_ratio = invest_amount / salary * 100 if salary > 0 else 0
        total_alloc = sum(a["ratio"] for a in allocations)
        max_alloc = 100 - invest_ratio
        if total_alloc > max_alloc and total_alloc > 0:
            scale = max_alloc / total_alloc
            allocations = [
                {**a, "ratio": max(1, round(a["ratio"] * scale))}
                for a in allocations
            ]

        return {**state, "invest_amount": invest_amount, "allocations": allocations}

    except Exception:
        return {
            **state,
            "invest_amount": round(state["spendable"] * 0.25),
            "allocations": [],
        }


def _build_graph() -> StateGraph:
    graph = StateGraph(RebalanceState)
    graph.add_node("plan", _plan_rebalance)
    graph.set_entry_point("plan")
    graph.add_edge("plan", END)
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
        "invest_amount": 0,
        "allocations": [],
    }

    final_state: RebalanceState = await _graph.ainvoke(initial_state)

    salary_rebalance: list[SalaryRebalanceItem] = [
        SalaryRebalanceItem(
            asset_id=UUID(alloc["asset_id"]),
            category=alloc["category"],
            ratio=int(alloc["ratio"]),
        )
        for alloc in final_state["allocations"]
    ]

    return RebalanceResponse(
        created_at=datetime.now(timezone.utc),
        invest_amount=final_state["invest_amount"],
        salary_rebalance=salary_rebalance,
    )
