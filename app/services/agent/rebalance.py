import logging
from datetime import datetime, timezone
from typing import Literal, TypedDict
from uuid import UUID
from langgraph.graph import END, StateGraph

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from app.schemas.portfolio import RebalanceRequest, RebalanceResponse, SalaryRebalanceItem
from app.services.agent.llm import ainvoke_structured
from app.services.agent.tools import normalize_amounts
from app.services.agent.porti_types import (
    porti_label,
    STABLE_PORTI_TYPES,
    NEUTRAL_PORTI_TYPES,
    INVEST_PORTI_TYPES,
)

# PorTI 성향별 invest_amount 허용 비율 (min, max)
_INVEST_RATIO: dict[str, tuple[float, float]] = {
    **{t: (0.10, 0.25) for t in STABLE_PORTI_TYPES},
    **{t: (0.15, 0.35) for t in NEUTRAL_PORTI_TYPES},
    **{t: (0.25, 0.50) for t in INVEST_PORTI_TYPES},
}

logger = logging.getLogger(__name__)


class _AllocationItem(BaseModel):
    asset_id: str = Field(default="", description="보유 계좌 목록에 있는 실제 asset_id")
    account_purpose: Literal["생활비", "비상금", "용돈", "저축", "여행", "기타"] = Field(
        default="기타", description="배분 용도 — 투자 관련 용어 사용 불가"
    )
    amount: int = Field(default=0, description="가처분소득 중 배분할 금액(원)")
    comment: str = Field(default="", description="배분 근거 한 줄 설명")


class _RebalancePlan(BaseModel):
    reasoning: str = Field(
        default="",
        description=(
            "배분 결정 전 상황 분석. 소비 패턴, 저축 여력, 계좌 상태, "
            "PorTI 성향과의 트레이드오프를 2~3문장으로 정리"
        ),
    )
    invest_amount: int = Field(default=0, description="투자로 별도 운용할 절대 금액(원)")
    allocations: list[_AllocationItem] = Field(
        default_factory=list,
        description="나머지 금액을 배분할 계좌 목록",
    )


class RebalanceState(TypedDict):
    salary: int
    fixed_expense: int
    spendable: int
    porti_type: str
    porti_comment: str
    category_list: list[dict]       # 원본: [{name, expense}]
    category_details: list[dict]    # 계산 후: [{name, expense, ratio}]
    category_total: int             # 변동지출 합계
    savings_capacity: int           # 저축 가능액 = 가처분소득 - 변동지출
    asset_list: list[dict]          # [{asset_id, asset_type, account_name, balance}]
    invest_amount: int
    allocations: list[dict]
    reasoning: str


def _diagnose(state: RebalanceState) -> RebalanceState:
    """소비 패턴 구조화 + 저축 여력 계산. LLM 없이 순수 계산."""
    salary = state["salary"]

    category_details = [
        {
            "name": c["name"],
            "expense": c["expense"],
            "ratio": round(c["expense"] / salary * 100, 1) if salary > 0 else 0.0,
        }
        for c in state["category_list"]
    ]

    category_total = sum(c["expense"] for c in state["category_list"])
    savings_capacity = state["spendable"] - category_total

    return {
        **state,
        "category_details": category_details,
        "category_total": category_total,
        "savings_capacity": savings_capacity,
    }


_SYSTEM = (
    "당신은 사용자의 월급을 맞춤 설계해주는 재무 어드바이저입니다.\n\n"
    "계좌 유형별 배분 가이드:\n"
    "- CHECKING(입출금통장): 생활비, 용돈\n"
    "- PARKING(파킹통장), CMA: 비상금 (유동성 확보 최우선)\n"
    "- DEPOSIT(정기예금): 중기 목돈 저축\n"
    "비상금 우선 규칙:\n"
    "- PARKING 또는 CMA 계좌 잔액이 0원이면 해당 계좌에 비상금을 최우선으로 배분\n"
    "- 비상금 목표: 월 고정지출의 2~3배 수준\n\n"
    "반드시 아래 순서로 생각하고 결정하세요.\n\n"
    "1단계 — reasoning (결정 전 반드시 먼저 작성)\n"
    "  아래 질문에 답하며 이 사람의 상황을 파악하세요:\n"
    "  · 소비 패턴에서 눈에 띄는 항목이 있는가? (소득 대비 비율 기준)\n"
    "  · 저축 가능액이 충분한가, 빠듯한가?\n"
    "  · 잔액 0원인 파킹통장/CMA가 있는가? → 비상금 먼저\n"
    "  · PorTI 성향 vs 현재 재무 상황 — 어떤 트레이드오프가 있는가?\n"
    "  → 이 분석을 reasoning 필드에 2~3문장으로 정리하세요.\n\n"
    "2단계 — 배분 결정\n"
    "  1단계 reasoning을 바탕으로 금액을 결정하세요.\n"
    "  금액은 천원 단위로 입력하세요.\n"
    "  각 계좌 comment는 reasoning에서 나온 실제 근거를 담아 1문장으로 쓰세요.\n\n"
    "출력 규칙:\n"
    "- reasoning: 배분 결정 전 상황 분석 (2~3문장)\n"
    "- invest_amount: 투자 운용금(원), 가처분소득 초과 불가\n"
    "- allocations:\n"
    "  - asset_id: 보유 계좌의 실제 UUID (변경·중복 금지)\n"
    "  - account_purpose: 계좌 유형 가이드에 맞는 한글 용도명\n"
    "  - amount: 배분 금액(원)\n"
    "  - comment: reasoning 기반, 실제 수치 언급 1문장\n"
    "    예) '식비가 소득의 14.7%라 생활비를 넉넉히 잡았어요'\n"
    "    예) '파킹통장 잔액이 0원이라 비상금부터 채워드렸어요'\n"
    "- 핵심 제약: invest_amount + sum(amount) = 가처분소득\n"
    "- 이모지·이모티콘 사용 금지\n"
)


async def _plan_rebalance(state: RebalanceState) -> RebalanceState:
    spendable = state["spendable"]
    min_ratio, max_ratio = _INVEST_RATIO.get(state["porti_type"], (0.15, 0.35))
    min_invest = round(spendable * min_ratio)
    max_invest = round(spendable * max_ratio)
    default_invest = round(spendable * (min_ratio + max_ratio) / 2)
    min_accounts = min(2, len(state["asset_list"]))

    # 소비 패턴: 카테고리별 금액 + 소득 대비 비율
    category_lines = "\n".join(
        f"  - {c['name']}: {c['expense']:,}원 (소득의 {c['ratio']}%)"
        for c in state["category_details"]
    ) or "  소비 내역 없음"

    # 계좌 목록: 잔액 포함
    asset_lines = "\n".join(
        f"  - asset_id: {a['asset_id']}, {a['account_name']} ({a['asset_type']}) 잔액: {a['balance']:,}원"
        for a in state["asset_list"]
    ) or "  보유 계좌 없음"

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=(
            f"월급: {state['salary']:,}원\n"
            f"고정지출(별도 처리됨): {state['fixed_expense']:,}원\n"
            f"가처분소득: {spendable:,}원\n\n"
            f"PorTI 유형: {porti_label(state['porti_type'])} — {state['porti_comment']}\n\n"
            f"소비 패턴 (3개월 월평균):\n{category_lines}\n"
            f"변동지출 합계: {state['category_total']:,}원\n"
            f"저축 가능액: {state['savings_capacity']:,}원 "
            f"(소득의 {round(state['savings_capacity']/state['salary']*100, 1) if state['salary'] > 0 else 0}%)\n\n"
            f"보유 계좌 (allocations의 asset_id는 아래 UUID만 사용):\n{asset_lines}\n\n"
            f"[제약 조건]\n"
            f"- invest_amount 허용 범위: {min_invest:,}원 ~ {max_invest:,}원 (PorTI 성향 기준)\n"
            f"- allocations 최소 계좌 수: {min_accounts}개"
        )),
    ]

    result = await ainvoke_structured(messages, _RebalancePlan)
    if result is None:
        return {**state, "invest_amount": default_invest, "allocations": [], "reasoning": ""}

    try:
        valid_ids = {a["asset_id"] for a in state["asset_list"]}

        # asset_id 검증 + 중복 제거
        seen_ids: set[str] = set()
        allocations: list[dict] = []
        for a in result.allocations:
            if a.asset_id in valid_ids and a.amount > 0 and a.asset_id not in seen_ids:
                allocations.append({
                    "asset_id": a.asset_id,
                    "account_purpose": a.account_purpose,
                    "amount": a.amount,
                    "comment": a.comment,
                })
                seen_ids.add(a.asset_id)

        # LLM이 asset_id를 잘못 반환했을 때 폴백: 순서대로 계좌 배정
        if not allocations and result.allocations and state["asset_list"]:
            for i, a in enumerate(result.allocations):
                if i >= len(state["asset_list"]):
                    break
                if a.amount > 0:
                    allocations.append({
                        "asset_id": state["asset_list"][i]["asset_id"],
                        "account_purpose": a.account_purpose,
                        "amount": a.amount,
                        "comment": a.comment,
                    })

        invest_amount = max(min_invest, min(max_invest, result.invest_amount))

        remaining_amount = spendable - invest_amount

        if allocations and remaining_amount > 0:
            allocations = normalize_amounts(allocations, "amount", remaining_amount)

        return {**state, "invest_amount": invest_amount, "allocations": allocations, "reasoning": result.reasoning}

    except Exception as e:
        logger.warning("배분 계획 처리 실패, 기본값 사용: %s", e)
        return {**state, "invest_amount": default_invest, "allocations": [], "reasoning": ""}


def _build_graph() -> StateGraph:
    graph = StateGraph(RebalanceState)
    graph.add_node("diagnose", _diagnose)
    graph.add_node("plan", _plan_rebalance)
    graph.set_entry_point("diagnose")
    graph.add_edge("diagnose", "plan")
    graph.add_edge("plan", END)
    return graph.compile()


_graph = _build_graph()


async def rebalance_salary(request: RebalanceRequest) -> RebalanceResponse:
    spendable = request.salary - request.fixed_expense

    category_list = [
        {"name": e.name, "expense": e.expense}
        for e in request.category_expense
    ]

    asset_list = [
        {
            "asset_id": str(a.asset_id),
            "asset_type": a.asset_type,
            "account_name": a.account_name,
            "balance": a.balance,
        }
        for a in request.assets
    ]

    initial_state: RebalanceState = {
        "salary": request.salary,
        "fixed_expense": request.fixed_expense,
        "spendable": spendable,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "category_list": category_list,
        "category_details": [],
        "category_total": 0,
        "savings_capacity": 0,
        "asset_list": asset_list,
        "invest_amount": 0,
        "allocations": [],
        "reasoning": "",
    }

    final_state: RebalanceState = await _graph.ainvoke(initial_state)

    salary_rebalance: list[SalaryRebalanceItem] = [
        SalaryRebalanceItem(
            asset_id=UUID(alloc["asset_id"]),
            account_purpose=alloc["account_purpose"],
            amount=int(alloc["amount"]),
            comment=alloc.get("comment", ""),
        )
        for alloc in final_state["allocations"]
    ]

    return RebalanceResponse(
        created_at=datetime.now(timezone.utc),
        invest_amount=final_state["invest_amount"],
        reasoning=final_state.get("reasoning", ""),
        salary_rebalance=salary_rebalance,
    )
