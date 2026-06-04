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
    account_purpose: str = Field(default="기타", description="배분 용도 한글 레이블 (예: 생활비, 비상금, 용돈)")
    ratio: int = Field(default=0, description="가처분소득 대비 배분 비율(%, 정수)")
    comment: str = Field(default="", description="배분 근거 한 줄 설명")


class _RebalancePlan(BaseModel):
    invest_amount: int = Field(default=0, description="투자로 별도 운용할 절대 금액(원)")
    allocations: list[_AllocationItem] = Field(
        default_factory=list,
        description="나머지 금액을 배분할 계좌 목록 (ratio는 가처분소득 대비 %)",
    )


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
    "당신은 월급 쪼개기 전문가입니다.\n"
    "가처분소득(= 월급 - 고정지출)을 투자금과 생활비 항목들로 나눕니다.\n\n"
    "배분 순서:\n"
    "  1단계: 가처분소득에서 투자 운용금(invest_amount)을 먼저 분리합니다.\n"
    "  2단계: 나머지(가처분소득 - invest_amount)를 생활비·비상금 등 항목별로 계좌에 배분합니다.\n\n"
    "규칙:\n"
    "- invest_amount: 투자 운용 절대 금액(원), 가처분소득을 초과할 수 없음\n"
    "  * 공격형(AGGRESSIVE): 가처분소득의 40~60%\n"
    "  * 균형형(BALANCED):   가처분소득의 20~35%\n"
    "  * 안정형(CONSERVATIVE): 가처분소득의 10~20%\n"
    "- allocations: 나머지 가처분소득을 배분할 계좌 목록\n"
    "  - asset_id: 보유 계좌 목록의 실제 UUID (절대 변경 금지, 중복 사용 금지)\n"
    "  - account_purpose: 한글 용도명 (생활비, 비상금, 용돈, 교통비, 저축 등)\n"
    "                     ※ CHECKING/SAVINGS 같은 영문 코드 금지, 반드시 한글\n"
    "  - ratio: 가처분소득 대비 배분 비율(%, 정수)\n"
    "  - comment: 이 계좌에 이 금액을 배분하는 이유 한 줄\n"
    "- 핵심 제약: invest_amount ÷ 가처분소득 × 100 + sum(ratio) = 100\n"
    "- 이모지, 이모티콘 사용 금지\n\n"
    "예시 (가처분소득 3,600,000원 / 균형형):\n"
    '{"invest_amount": 900000, "allocations": [\n'
    '  {"asset_id": "uuid-A", "account_purpose": "생활비", "ratio": 50, "comment": "식비·생활용품 등 변동 지출 충당"},\n'
    '  {"asset_id": "uuid-B", "account_purpose": "비상금", "ratio": 15, "comment": "예비 지출 및 비상금 적립"},\n'
    '  {"asset_id": "uuid-C", "account_purpose": "용돈",   "ratio": 10, "comment": "개인 용돈 및 여가비"}\n'
    "]}\n"
    "위 예시에서 invest_amount(25%) + 50% + 15% + 10% = 100%"
)


def _plan_rebalance(state: RebalanceState) -> RebalanceState:
    spendable = state["spendable"]
    default_invest = round(spendable * 0.25)

    asset_lines = "\n".join(
        f"  - asset_id: {a['asset_id']}, {a['account_name']} ({a['asset_type']})"
        for a in state["asset_list"]
    ) or "  보유 계좌 없음"

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=(
            f"월급: {state['salary']:,}원\n"
            f"고정지출(이미 별도 처리됨): {state['fixed_expense']:,}원\n"
            f"가처분소득: {spendable:,}원\n\n"
            f"PorTI 유형: {state['porti_type']} — {state['porti_comment']}\n"
            f"최근 변동지출 패턴: {state['expense_summary']}\n\n"
            f"보유 계좌 목록 (allocations의 asset_id는 아래 UUID만 사용):\n{asset_lines}"
        )),
    ]

    result = invoke_structured(messages, _RebalancePlan, temperature=0.1)
    if result is None:
        return {**state, "invest_amount": default_invest, "allocations": []}

    try:
        valid_ids = {a["asset_id"] for a in state["asset_list"]}

        # asset_id 검증 + 중복 제거
        seen_ids: set[str] = set()
        allocations: list[dict] = []
        for a in result.allocations:
            if a.asset_id in valid_ids and a.ratio > 0 and a.asset_id not in seen_ids:
                allocations.append({
                    "asset_id": a.asset_id,
                    "account_purpose": a.account_purpose,
                    "ratio": a.ratio,
                    "comment": a.comment,
                })
                seen_ids.add(a.asset_id)

        # LLM이 asset_id를 잘못 반환했을 때 폴백: 순서대로 계좌 배정
        if not allocations and result.allocations and state["asset_list"]:
            for i, a in enumerate(result.allocations):
                if i >= len(state["asset_list"]):
                    break
                if a.ratio > 0:
                    allocations.append({
                        "asset_id": state["asset_list"][i]["asset_id"],
                        "account_purpose": a.account_purpose,
                        "ratio": a.ratio,
                        "comment": a.comment,
                    })

        invest_amount = result.invest_amount

        # invest_amount가 가처분소득 초과 시 보정
        if invest_amount >= spendable:
            invest_amount = default_invest

        # invest_ratio + sum(ratios) = 100 맞추기
        # invest_ratio를 먼저 확정하고, 남은 %를 allocations에 비례 배분
        invest_ratio = round(invest_amount / spendable * 100) if spendable > 0 else 25
        remaining_ratio = 100 - invest_ratio  # allocations이 가져야 할 %

        total_alloc = sum(a["ratio"] for a in allocations)
        if total_alloc > 0 and total_alloc != remaining_ratio:
            scale = remaining_ratio / total_alloc
            allocations = [
                {**a, "ratio": max(1, round(a["ratio"] * scale))}
                for a in allocations
            ]
            # 반올림 오차 보정: 마지막 항목에 나머지 흡수
            actual_sum = sum(a["ratio"] for a in allocations)
            diff = remaining_ratio - actual_sum
            if diff != 0 and allocations:
                allocations[-1] = {**allocations[-1], "ratio": max(1, allocations[-1]["ratio"] + diff)}

        return {**state, "invest_amount": invest_amount, "allocations": allocations}

    except Exception:
        return {**state, "invest_amount": default_invest, "allocations": []}


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
            account_purpose=alloc["account_purpose"],
            ratio=int(alloc["ratio"]),
            comment=alloc.get("comment", ""),
        )
        for alloc in final_state["allocations"]
    ]

    return RebalanceResponse(
        created_at=datetime.now(timezone.utc),
        invest_amount=final_state["invest_amount"],
        salary_rebalance=salary_rebalance,
    )
