from datetime import datetime, timezone
from typing import TypedDict
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.schemas.event import (
    EventRebalanceRequest,
    EventRebalanceResponse,
    SalaryRebalanceItem,
)
from app.services.agent.llm import invoke_structured
from app.services.agent.tools import monthly_savings_needed


class _AllocationItem(BaseModel):
    account_name: str = Field(default="")
    asset_id: str = Field(default="")
    category: str = Field(default="기타")
    amount: int = Field(default=0)


class _EventRebalancePlan(BaseModel):
    invest_amount: int = Field(default=0, description="이벤트 목표 포함 전체 투자금(원)")
    allocations: list[_AllocationItem] = Field(default_factory=list, description="계좌별 배분 목록")
    rebalance_comment: str = Field(default="이벤트 목표에 맞게 재조정했어요.", description="조정 내용 한 줄 설명")


class EventRebalanceState(TypedDict):
    title: str
    target_amount: int
    months_left: int
    monthly_needed: int
    monthly_needed_accurate: float   # 복리 적용 정확 계산값 (tool 결과)
    salary: int
    current_invest_amount: int
    current_allocations: list[dict]
    porti_type: str
    porti_comment: str
    new_invest_amount: int
    new_allocations: list[dict]
    rebalance_comment: str


_SYSTEM = (
    "당신은 이벤트 목표 자금 마련을 위한 월급 배분 전문가입니다.\n"
    "기존 월급 배분을 최대한 유지하면서, 목표 달성을 위한 저축액을 조정하세요.\n\n"
    "추론 순서:\n"
    "1. 현재 invest_amount에 이벤트 저축을 포함할 수 있는지 확인하세요.\n"
    "2. allocations 합계 + invest_amount ≤ salary 제약을 지키세요.\n"
    "3. 공격적 성향은 invest_amount 비중을 높게, 안정형은 낮게 조정하세요.\n"
    "4. 이벤트 저축은 '목표저축' 카테고리로 추가하거나 기존 항목을 증액하세요.\n\n"
    "예시:\n"
    "급여 4,800,000원 / 현재 투자금 600,000원 / 월 필요 저축 250,000원 (복리 적용)\n"
    "→ invest_amount: 850000 (600000 + 250000)\n"
    "→ allocations: 기존 배분 유지하되 잔여 범위 내 조정\n"
    "→ rebalance_comment: '이벤트 목표를 위해 매월 25만원을 추가로 투자합니다.'"
)


def _plan_rebalance(state: EventRebalanceState) -> EventRebalanceState:
    # Tool로 복리 적용 정확 월 저축액 계산
    accurate = float(state["monthly_needed_accurate"])

    alloc_summary = "\n".join(
        f"  - {a['category']}: {a['amount']:,}원 ({a['account_name']})"
        for a in state["current_allocations"]
    )

    try:
        result = invoke_structured(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=(
                    f"이벤트: {state['title']}\n"
                    f"목표 금액: {state['target_amount']:,}원\n"
                    f"남은 기간: {state['months_left']}개월\n"
                    f"월 필요 저축 (단순): {state['monthly_needed']:,}원\n"
                    f"월 필요 저축 (복리 0.3% 적용 정확값): {accurate:,.0f}원\n"
                    f"급여: {state['salary']:,}원\n"
                    f"현재 투자금: {state['current_invest_amount']:,}원\n"
                    f"PorTI 유형: {state['porti_type']} — {state['porti_comment']}\n"
                    f"현재 월급 배분:\n{alloc_summary}"
                )),
            ],
            _EventRebalancePlan,
            temperature=0.1,
        )
        if result is None:
            raise ValueError("invoke_structured returned None")
        return {
            **state,
            "new_invest_amount": result.invest_amount,
            "new_allocations": [a.model_dump() for a in result.allocations],
            "rebalance_comment": result.rebalance_comment,
        }
    except Exception:
        return {
            **state,
            "new_invest_amount": state["current_invest_amount"],
            "new_allocations": state["current_allocations"],
            "rebalance_comment": "이벤트 목표에 맞게 재조정했어요.",
        }


def _build_graph() -> StateGraph:
    graph = StateGraph(EventRebalanceState)
    graph.add_node("plan", _plan_rebalance)
    graph.set_entry_point("plan")
    graph.add_edge("plan", END)
    return graph.compile()


_graph = _build_graph()


async def rebalance_event(request: EventRebalanceRequest) -> EventRebalanceResponse:
    now = datetime.now(timezone.utc)
    months_left = max(1, int((request.deadline - now).days / 30))

    # Tool 호출: 복리 적용 월 저축액 (단순 나눗셈보다 정확)
    monthly_accurate = monthly_savings_needed.invoke({
        "target": float(request.target_amount),
        "current": 0.0,
        "months": months_left,
        "monthly_rate": 0.003,
    })

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
        "monthly_needed": max(0, request.target_amount // months_left),
        "monthly_needed_accurate": monthly_accurate,
        "salary": request.rebalance.salary,
        "current_invest_amount": request.rebalance.invest_amount,
        "current_allocations": current_allocations,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
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
