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
from app.services.agent.tools import normalize_to_thousands
from app.services.agent.porti_types import (
    porti_detail,
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


_MAX_REFLECT_ITER = 2


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
    feedback: str       # reflect가 plan에게 보내는 재계획 지시
    iteration: int      # plan→reflect 사이클 횟수
    approved: bool      # reflect가 결과를 승인했는지 여부



_SYSTEM = (
    "당신은 사용자의 월급을 맞춤 설계해주는 재무 어드바이저입니다.\n\n"
    "계좌 유형별 배분 가이드:\n"
    "- CHECKING(입출금통장): 생활비, 여가비\n"
    "- PARKING(파킹통장), CMA: 비상금, 예비비\n"
    "- DEPOSIT(정기예금): 목돈\n\n"
    "배분 우선순위 (순서대로 적용):\n"
    "1. 소비 패턴 기반 생활비 — 카테고리별 지출을 보고 다음 달 변동지출을 커버할 생활비를 CHECKING에 배분\n"
    "2. 비상금 — 월급의 3~6배를 목표로 배분\n"
    "3. 예비비 — 월급의 10% 정도를 예비비 명목으로 저축(DEPOSIT 또는 PARKING)\n"
    "4. 용돈·여행 등 — 소비 패턴을 보고 특수한 영역 있으면 별도 배분\n\n"
    "균형 배분 규칙 (필수 준수):\n"
    "- 보유 계좌가 2개 이상이면 반드시 2가지 이상 다른 용도로 분산 배분\n"
    "- 단일 항목이 가처분소득의 60%를 초과하지 않도록 균형 있게 배분\n"
    "- 각 배분 항목 금액: 최소 50,000원 이상\n"
    "- 모든 금액: 반드시 천원(1,000원) 단위\n\n"
    "1단계 — reasoning (결정 전 반드시 먼저 작성)\n"
    "  소비 패턴과 재무 상황을 분석하세요:\n"
    "  · 어떤 소비 항목이 크고, 생활비로 얼마가 필요한가?\n"
    "  · 보유 중인 계좌가 비상금, 예비비를 보유하고 있는가?\n"
    "  · PorTI 성향에 맞는 투자금 규모는?\n"
    "  → 이 분석을 reasoning에 2~3문장으로 정리\n\n"
    "2단계 — 배분 결정\n"
    "  reasoning을 바탕으로 금액을 결정하세요.\n"
    "  금액은 반드시 천원(1,000원) 단위로만 입력하세요.\n"
    "  각 comment는 reasoning의 실제 근거를 담은 1문장\n\n"
    "출력 규칙:\n"
    "- reasoning: 배분 결정 전 상황 분석 (2~3문장)\n"
    "- invest_amount: 투자 운용금(원), 가처분소득 초과 불가, 천원 단위\n"
    "- allocations:\n"
    "  - asset_id: 보유 계좌의 실제 UUID (변경·중복 금지)\n"
    "  - account_purpose: 계좌 유형 가이드에 맞는 한글 용도명\n"
    "  - amount: 배분 금액(원), 반드시 천원 단위\n"
    "  - comment: 실제 수치 언급 1문장\n"
    "    예) '식비가 소득의 14.7%라 생활비를 넉넉히 잡았어요'\n"
    "    예) '파킹통장 잔액이 0원이라 비상금을 일부 채워드렸어요'\n"
    "    예) '저축 여력이 충분해 정기예금에 저축을 배분했어요'\n"
    "- 핵심 제약: invest_amount + sum(amount) = 가처분소득\n"
    "- 이모지·이모티콘 사용 금지\n"
)


async def _plan_rebalance(state: RebalanceState) -> RebalanceState:
    spendable = state["spendable"]
    min_ratio, max_ratio = _INVEST_RATIO.get(state["porti_type"], (0.15, 0.35))
    min_invest = round(spendable * min_ratio / 1000) * 1000
    max_invest = round(spendable * max_ratio / 1000) * 1000
    default_invest = round(spendable * (min_ratio + max_ratio) / 2 / 1000) * 1000
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

    feedback = state.get("feedback", "").strip()
    feedback_section = (
        f"\n[이전 배분 계획 피드백 — 반드시 반영]\n{feedback}\n"
        if feedback else ""
    )

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=(
            f"월급: {state['salary']:,}원\n"
            f"고정지출(별도 처리됨): {state['fixed_expense']:,}원\n"
            f"가처분소득: {spendable:,}원\n\n"
            f"PorTI 유형: {porti_detail(state['porti_type'])}\n"
            f"사용자 투자 성향 코멘트: {state['porti_comment']}\n\n"
            f"소비 패턴 (3개월 월평균):\n{category_lines}\n"
            f"변동지출 합계: {state['category_total']:,}원\n"
            f"저축 가능액(가처분소득 - 변동지출): {state['savings_capacity']:,}원 "
            f"(소득의 {round(state['savings_capacity']/state['salary']*100, 1) if state['salary'] > 0 else 0}%)\n\n"
            f"보유 계좌 (allocations의 asset_id는 아래 UUID만 사용):\n{asset_lines}\n\n"
            f"[제약 조건]\n"
            f"- invest_amount 허용 범위: {min_invest:,}원 ~ {max_invest:,}원 (PorTI 성향 기준)\n"
            f"- allocations 최소 계좌 수: {min_accounts}개"
            f"{feedback_section}"
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

        invest_amount = max(min_invest, min(max_invest, round(result.invest_amount / 1000) * 1000))

        remaining_amount = spendable - invest_amount

        if allocations and remaining_amount > 0:
            allocations = normalize_to_thousands(allocations, "amount", remaining_amount)

        return {**state, "invest_amount": invest_amount, "allocations": allocations, "reasoning": result.reasoning}

    except Exception as e:
        logger.warning("배분 계획 처리 실패, 기본값 사용: %s", e)
        return {**state, "invest_amount": default_invest, "allocations": [], "reasoning": ""}


class _ReviewedAllocation(BaseModel):
    asset_id: str = Field(description="원래 asset_id 그대로 유지 (변경 금지)")
    account_purpose: Literal["생활비", "비상금", "용돈", "저축", "여행", "기타"] = Field(
        description="배분 용도"
    )
    amount: int = Field(description="조정된 배분 금액(원), 천원 단위")
    comment: str = Field(description="실제 수치(금액 또는 비율)를 포함한 근거 1문장")


class _ReflectOutput(BaseModel):
    approved: bool = Field(
        description=(
            "배분 결과가 모든 기준을 충족하면 True. "
            "근본적 결함(계좌 분산 실패, 투자금 과다 등)이 있으면 False."
        )
    )
    feedback: str = Field(
        default="",
        description="approved=False일 때 재계획에 필요한 구체적 수정 지시. approved=True이면 빈 문자열.",
    )
    reasoning: str = Field(
        description=(
            "최종 배분 결과를 바탕으로 한 2~3문장 설명. "
            "반드시 ① 소비 패턴에서 큰 항목, ② 투자금 설정 이유(PorTI 성향·저축 여력), "
            "③ 계좌 배분 전략 중 2가지 이상을 구체적 수치와 함께 포함"
        )
    )
    allocations: list[_ReviewedAllocation] = Field(
        description="점검 완료된 배분 목록 (asset_id 원본 순서 유지)"
    )


_REFLECT_SYSTEM = (
    "당신은 월급 배분 결과를 검토하는 재무 감수자입니다.\n\n"
    "[1단계] 승인/거부 판단 (approved)\n"
    "아래 기준 중 하나라도 해당하면 approved=False, feedback에 구체적 수정 지시 작성:\n"
    "  - 계좌가 2개 이상인데 배분 항목이 1개뿐인 경우 (분산 배분 실패)\n"
    "  - invest_amount가 가처분소득의 50%를 초과하는 경우\n"
    "  - 계좌 유형과 용도가 명백히 불일치하는 경우 (예: DEPOSIT에 생활비)\n"
    "위 기준에 해당하지 않으면 approved=True.\n\n"
    "[2단계] approved=True일 때만 수행\n"
    "입력으로 제공된 [계획 reasoning]은 참고용입니다. "
    "그 내용을 그대로 쓰지 말고, 최종 배분 결과만을 근거로 새로 작성하세요.\n\n"
    "  A. reasoning 재작성 (최종 배분 결과 기준, 완전 새로 작성):\n"
    "     - 소비 패턴에서 가장 큰 항목과 실제 금액·비율을 언급\n"
    "     - 투자금을 해당 금액으로 설정한 이유를 PorTI 성향과 저축 여력에 연결해 설명\n"
    "     - 각 계좌에 해당 금액을 배분한 핵심 이유를 구체적 수치와 함께 1~2가지 설명\n"
    "     → 2~3문장, '~했어요', '~드렸어요' 부드러운 경어체\n\n"
    "  B. 각 계좌 comment 재작성 (최종 배분 결과 기준, 완전 새로 작성):\n"
    "     - 해당 계좌의 실제 배분 금액과 용도만을 근거로 새로 작성\n"
    "     - 구체적인 수치(금액 또는 비율)를 반드시 포함한 1문장\n"
    "     - 계획 단계 comment를 재활용하지 말 것\n\n"
    "  C. 극단적 배분 교정 (인라인 수정, 재거부 불필요):\n"
    "     - 단일 항목이 배분 가능액의 60%를 초과하면 초과분을 다른 계좌로 분산\n"
    "     - 금액이 50,000원 미만인 항목은 가장 큰 항목에 합산하고 해당 항목 제거\n"
    "     - 조정 후 합계가 배분 가능액과 반드시 일치 (천원 단위)\n\n"
    "[approved=False일 때]\n"
    "  - feedback: 재계획 시 반드시 지켜야 할 구체적 수정 지시 (예: '계좌 2개 모두 사용하고 "
    "CHECKING에 생활비, PARKING에 비상금으로 분산할 것')\n"
    "  - reasoning, allocations: 원래 입력 그대로 반환 (수정 불필요)\n\n"
    "출력 규칙:\n"
    "- asset_id: 원래 값 그대로 유지 (추가·변경·삭제 금지, 단 50,000원 미만 항목 제거는 허용)\n"
    "- 모든 금액: 천원(1,000원) 단위\n"
    "- 이모지·이모티콘 사용 금지\n"
)


async def _reflect(state: RebalanceState) -> RebalanceState:
    if not state["allocations"]:
        return state

    spendable = state["spendable"]
    invest_amount = state["invest_amount"]
    remaining = spendable - invest_amount
    single_cap = round(remaining * 0.6 / 1000) * 1000

    alloc_lines = "\n".join(
        f"  - asset_id: {a['asset_id']}, {a['account_purpose']}: {a['amount']:,}원"
        for a in state["allocations"]
    )
    category_lines = "\n".join(
        f"  - {c['name']}: {c['expense']:,}원 (소득의 {c['ratio']}%)"
        for c in state["category_details"]
    ) or "  소비 내역 없음"

    prior_reasoning = state.get("reasoning", "").strip()
    prior_section = (
        f"\n[계획 reasoning — 참고만 할 것, 그대로 쓰지 말 것]\n{prior_reasoning}\n"
        if prior_reasoning else ""
    )

    messages = [
        SystemMessage(content=_REFLECT_SYSTEM),
        HumanMessage(content=(
            f"PorTI 유형: {porti_detail(state['porti_type'])}\n"
            f"사용자 투자 성향 코멘트: {state['porti_comment']}\n\n"
            f"가처분소득: {spendable:,}원\n"
            f"투자 운용금: {invest_amount:,}원\n"
            f"배분 가능 잔액 (가처분소득 - 투자금): {remaining:,}원\n\n"
            f"[최종 배분 결과 — reasoning과 comment 재작성의 유일한 근거]\n"
            f"{alloc_lines}\n\n"
            f"소비 패턴 (3개월 월평균):\n{category_lines}\n"
            f"변동지출 합계: {state['category_total']:,}원\n"
            f"저축 가능액(가처분소득 - 변동지출): {state['savings_capacity']:,}원\n\n"
            f"[교정 기준]\n"
            f"- 단일 항목 상한: {single_cap:,}원 (배분 가능액의 60%)\n"
            f"- 항목 최소 금액: 50,000원"
            f"{prior_section}"
        )),
    ]

    result = await ainvoke_structured(messages, _ReflectOutput)
    if result is None:
        return {**state, "approved": True}

    next_iter = state.get("iteration", 0) + 1

    # 거부: 재계획 필요 — allocations/reasoning은 그대로 유지
    if not result.approved:
        return {**state, "approved": False, "feedback": result.feedback, "iteration": next_iter}

    try:
        valid_ids = {a["asset_id"] for a in state["asset_list"]}
        seen_ids: set[str] = set()
        reviewed: list[dict] = []

        for a in result.allocations:
            if a.asset_id in valid_ids and a.amount > 0 and a.asset_id not in seen_ids:
                reviewed.append({
                    "asset_id": a.asset_id,
                    "account_purpose": a.account_purpose,
                    "amount": a.amount,
                    "comment": a.comment,
                })
                seen_ids.add(a.asset_id)

        # asset_id 검증 실패 시 원래 배분 유지하되 reasoning만 갱신
        if not reviewed:
            return {**state, "reasoning": result.reasoning, "approved": True, "iteration": next_iter}

        if remaining > 0:
            reviewed = normalize_to_thousands(reviewed, "amount", remaining)

        return {
            **state,
            "reasoning": result.reasoning,
            "allocations": reviewed,
            "approved": True,
            "iteration": next_iter,
        }

    except Exception as e:
        logger.warning("reflect 처리 실패, 원래 결과 유지: %s", e)
        return {
            **state,
            "reasoning": result.reasoning if result else state["reasoning"],
            "approved": True,
            "iteration": next_iter,
        }


def _route_after_reflect(state: RebalanceState) -> str:
    if not state.get("approved", True) and state.get("iteration", 0) < _MAX_REFLECT_ITER:
        return "plan"
    return END


def _build_graph() -> StateGraph:
    graph = StateGraph(RebalanceState)
    graph.add_node("plan", _plan_rebalance)
    graph.add_node("reflect", _reflect)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "reflect")
    graph.add_conditional_edges("reflect", _route_after_reflect, {"plan": "plan", END: END})
    return graph.compile()


_graph = _build_graph()


async def rebalance_salary(request: RebalanceRequest) -> RebalanceResponse:
    salary = request.salary
    spendable = salary - request.fixed_expense

    category_list = [
        {"name": e.name, "expense": e.expense}
        for e in request.category_expense
    ]
    category_details = [
        {
            "name": c["name"],
            "expense": c["expense"],
            "ratio": round(c["expense"] / salary * 100, 1) if salary > 0 else 0.0,
        }
        for c in category_list
    ]
    category_total = sum(c["expense"] for c in category_list)
    savings_capacity = spendable - category_total

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
        "salary": salary,
        "fixed_expense": request.fixed_expense,
        "spendable": spendable,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "category_list": category_list,
        "category_details": category_details,
        "category_total": category_total,
        "savings_capacity": savings_capacity,
        "asset_list": asset_list,
        "invest_amount": 0,
        "allocations": [],
        "reasoning": "",
        "feedback": "",
        "iteration": 0,
        "approved": False,
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
