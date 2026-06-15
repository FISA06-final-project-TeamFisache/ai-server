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

# ── Few-Shot CoT 예시 라이브러리 ──────────────────────────────────────────────

_FEW_SHOT_CHECKING_PARKING = (
    "[참고 예시 — 계좌 2개: 입출금 + 파킹/CMA, PorTI: NEUTRAL(JUDO)]\n"
    "상황: 가처분소득 200만원, 변동지출 75만원(식비·교통·여가), 저축 여력 125만원\n"
    "계좌: A입출금통장(CHECKING, 잔액 45만원) / B파킹통장(PARKING, 잔액 3만원)\n\n"
    "<STEP 1> 계좌 역할 결정\n"
    "  A입출금통장(CHECKING) → 역할: 생활비  이유: 카드결제·이체 자유, 일상 소비 전담\n"
    "  B파킹통장(PARKING)   → 역할: 비상금  이유: 고금리 유동성 계좌, 잔액 3만원으로 거의 비어 우선 충전 필요\n\n"
    "<STEP 2> invest_amount 결정 — PorTI 허용 범위를 반드시 먼저 확인\n"
    "  · PorTI NEUTRAL(JUDO) 허용 범위: 200만원 x 15~35% = 300,000원~700,000원\n"
    "  · 비상금 우선 채워야 하므로 범위 하한에 가깝게 설정\n"
    "  · invest_amount = 300,000원\n\n"
    "<STEP 3> 나머지 배분 (가처분소득 2,000,000원 - invest_amount 300,000원 = 1,700,000원)\n"
    "  · 식비(35만원, 17.5%)가 1위 지출 → 생활비는 변동지출 75만원 + 여유 5만원 = 800,000원\n"
    "  · 파킹 잔액 3만원, 비상금 목표(월급 3배) 훨씬 미달 → 나머지 900,000원 전액 비상금\n\n"
    "예시 결과: invest_amount=300,000 / CHECKING 생활비 800,000원 / PARKING 비상금 900,000원\n"
    "합계 확인: 300,000 + 800,000 + 900,000 = 2,000,000원 ✓"
)

_FEW_SHOT_CHECKING_DEPOSIT = (
    "[참고 예시 — 계좌 2개: 입출금 + 정기예금, PorTI: STABLE(ARCHERY)]\n"
    "상황: 가처분소득 250만원, 변동지출 90만원, 저축 여력 160만원\n"
    "계좌: A입출금통장(CHECKING, 잔액 120만원) / B정기예금(DEPOSIT, 잔액 500만원)\n\n"
    "<STEP 1> 계좌 역할 결정\n"
    "  A입출금통장(CHECKING) → 역할: 생활비  이유: 일상 지출 전담, 수시 입출금 가능\n"
    "  B정기예금(DEPOSIT)   → 역할: 저축    이유: 장기 목돈 적립 전용, 중도해지 불이익 있어 생활비에 사용 부적합\n\n"
    "<STEP 2> invest_amount 결정 — PorTI 허용 범위를 반드시 먼저 확인\n"
    "  · PorTI STABLE(ARCHERY) 허용 범위: 250만원 x 10~25% = 250,000원~625,000원\n"
    "  · 원금 보전 선호 성향 → 범위 최솟값 설정\n"
    "  · invest_amount = 250,000원\n\n"
    "<STEP 3> 나머지 배분 (가처분소득 2,500,000원 - invest_amount 250,000원 = 2,250,000원)\n"
    "  · 변동지출 90만원 + 여유 20만원 = 생활비 110만원\n"
    "  · 나머지 115만원 전액 정기예금 저축\n\n"
    "예시 결과: invest_amount=250,000 / CHECKING 생활비 1,100,000원 / DEPOSIT 저축 1,150,000원\n"
    "합계 확인: 250,000 + 1,100,000 + 1,150,000 = 2,500,000원 ✓"
)

_FEW_SHOT_THREE_ACCOUNTS = (
    "[참고 예시 — 계좌 3개: 입출금 + 파킹 + 정기예금, PorTI: INVEST(CYCLING)]\n"
    "상황: 가처분소득 300만원, 변동지출 110만원(식비·교통·쇼핑 등), 저축 여력 190만원\n"
    "계좌: A입출금(CHECKING, 잔액 80만원) / B파킹(PARKING, 잔액 200만원) / C정기예금(DEPOSIT, 잔액 800만원)\n\n"
    "<STEP 1> 계좌 역할 결정 — 3개 계좌 모두 역할 배정 필수\n"
    "  A입출금(CHECKING) → 역할: 생활비  이유: 일상 소비 및 카드결제 전담\n"
    "  B파킹(PARKING)   → 역할: 예비비  이유: 잔액 200만원으로 비상금은 어느 정도 확보됨 → 이번 달은 예비비 소액 추가\n"
    "  C정기예금(DEPOSIT)→ 역할: 저축    이유: 장기 목돈 적립 전용, 매달 꾸준히 적립\n\n"
    "<STEP 2> invest_amount 결정 — PorTI 허용 범위를 반드시 먼저 확인\n"
    "  · PorTI INVEST(CYCLING) 허용 범위: 300만원 x 25~50% = 750,000원~1,500,000원\n"
    "  · 적극적 수익 추구 성향, 저축 여력 충분 → 범위 내 적극적 설정\n"
    "  · invest_amount = 900,000원 (범위 내 30% 수준)\n\n"
    "<STEP 3> 나머지 배분 (가처분소득 3,000,000원 - invest_amount 900,000원 = 2,100,000원)\n"
    "  · 변동지출 110만원 + 여유 10만원 = 생활비 120만원\n"
    "  · 파킹 잔액 충분 → 예비비 명목 소액(50만원) 배분\n"
    "  · 나머지 40만원 → 정기예금 저축\n\n"
    "예시 결과: invest_amount=900,000 / CHECKING 생활비 1,200,000원 / PARKING 예비비 500,000원 / DEPOSIT 저축 400,000원\n"
    "합계 확인: 900,000 + 1,200,000 + 500,000 + 400,000 = 3,000,000원 ✓"
)

_FEW_SHOT_FOUR_PLUS = (
    "[참고 예시 — 계좌 4개 이상, PorTI: NEUTRAL(RHYTHMIC)]\n"
    "상황: 가처분소득 350만원, 변동지출 130만원, 저축 여력 220만원\n"
    "계좌: A입출금(CHECKING, 잔액 30만원) / B파킹(PARKING, 잔액 0원) / C정기예금(DEPOSIT, 잔액 1,000만원) / D CMA(CMA, 잔액 50만원)\n\n"
    "<STEP 1> 계좌 역할 결정 — 4개 계좌 모두 역할 배정 필수, 어떤 계좌도 빠뜨리지 말 것\n"
    "  A입출금(CHECKING) → 역할: 생활비  이유: 일상 지출 전담\n"
    "  B파킹(PARKING)   → 역할: 비상금  이유: 잔액 0원 → 비상금 목표 미달, 최우선 충전 필요\n"
    "  C정기예금(DEPOSIT)→ 역할: 저축    이유: 장기 목돈 적립\n"
    "  D CMA(CMA)       → 역할: 용돈    이유: 단기 유동성 자금(여행·취미 예비비)\n\n"
    "<STEP 2> invest_amount 결정 — PorTI 허용 범위를 반드시 먼저 확인\n"
    "  · PorTI NEUTRAL(RHYTHMIC) 허용 범위: 350만원 x 15~35% = 525,000원~1,225,000원\n"
    "  · 균형 투자 성향, 저축 여력 충분 → 범위 중간 설정\n"
    "  · invest_amount = 700,000원 (범위 내 20% 수준)\n\n"
    "<STEP 3> 나머지 배분 (가처분소득 3,500,000원 - invest_amount 700,000원 = 2,800,000원)\n"
    "  · 변동지출 130만원 커버 → 생활비 135만원\n"
    "  · 파킹 잔액 0원 → 비상금 최우선 100만원 충전\n"
    "  · 나머지를 저축(35만원)·용돈(10만원)으로 분산, 금액이 작아도 역할이 다르면 배분\n\n"
    "예시 결과: invest_amount=700,000 / CHECKING 생활비 1,350,000원 / PARKING 비상금 1,000,000원 / DEPOSIT 저축 350,000원 / CMA 용돈 100,000원\n"
    "합계 확인: 700,000 + 1,350,000 + 1,000,000 + 350,000 + 100,000 = 3,500,000원 ✓"
)


def _build_few_shot_examples(asset_list: list[dict]) -> str:
    """계좌 조합(유형·수)에 맞는 Few-Shot CoT 예시를 동적으로 선택·반환."""
    if not asset_list or len(asset_list) < 2:
        return ""
    types = {a["asset_type"] for a in asset_list}
    n = len(asset_list)
    if n >= 4:
        example = _FEW_SHOT_FOUR_PLUS
    elif n == 3:
        example = _FEW_SHOT_THREE_ACCOUNTS
    elif types & {"PARKING", "CMA"}:
        example = _FEW_SHOT_CHECKING_PARKING
    elif "DEPOSIT" in types:
        example = _FEW_SHOT_CHECKING_DEPOSIT
    else:
        example = _FEW_SHOT_CHECKING_PARKING
    return (
        "\n\n[참고: 유사 상황 추론 예시 — 아래 사고 방식을 그대로 따르되 이 사용자 데이터에 맞게 적용하세요]\n"
        + example
    )


def _build_account_role_section(asset_list: list[dict]) -> str:
    """배분 전 계좌 역할 사전 분석 CoT 섹션 생성."""
    if len(asset_list) < 2:
        return ""
    lines = "\n".join(
        f"  - {a['account_name']} ({a['asset_type']}, 잔액 {a['balance']:,}원) → 역할: ?  이유: ?"
        for a in asset_list
    )
    return (
        "\n\n[STEP 1: 배분 전 각 계좌의 역할을 먼저 결정하세요 — 모든 계좌에 역할 부여 필수]\n"
        f"{lines}\n"
        "→ 위 역할 결정이 reasoning의 출발점이 되어야 합니다."
    )


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
            "투자 성향과의 트레이드오프를 2~3문장으로 정리. "
            "JUDO·CYCLING 등 PorTI 유형 코드명은 절대 언급하지 말 것"
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
            f"{_build_account_role_section(state['asset_list'])}"
            f"{_build_few_shot_examples(state['asset_list'])}"
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
    "  - 사용 가능 계좌가 3개 이상인데 배분 항목이 1개 이하인 경우 (미활용 계좌 과다)\n"
    "  - 사용 가능 계좌가 4개 이상인데 배분 항목이 2개 이하인 경우 (미활용 계좌 과다)\n"
    "  - invest_amount가 프롬프트 내 'PorTI 허용 범위'를 벗어난 경우\n"
    "  - 계좌 유형과 용도가 명백히 불일치하는 경우 (예: DEPOSIT에 생활비)\n"
    "  - 계좌 용도와 comment가 명백히 불일치하는 경우 (예: comment에 '비상금'이라 쓰여 있는데 account_purpose가 '저축')\n"
    "  - DEPOSIT 계좌가 있는데 저축·목돈 관련 배분이 전혀 없는 경우 (DEPOSIT 계좌 미활용)\n"
    "위 기준에 해당하지 않으면 approved=True.\n\n"
    "[승인/거부 판단 예시 — 경계 케이스 기준]\n"
    "예시 A: 계좌 3개(CHECKING·PARKING·DEPOSIT), 배분 2개\n"
    "  → approved=True  이유: '3개 이상인데 1개 이하' 기준 미해당(배분 2개 ≥ 2개)\n"
    "  → feedback: '' (빈 문자열, 추가 요구 금지)\n\n"
    "예시 B: 계좌 3개(CHECKING·PARKING·DEPOSIT), 배분 1개\n"
    "  → approved=False  이유: '3개 이상인데 1개 이하' 기준 해당\n"
    "  → feedback: 'CHECKING에 생활비 배분 외에 PARKING 또는 DEPOSIT에도 비상금/저축 배분 추가'\n\n"
    "예시 C: 계좌 4개, 배분 2개\n"
    "  → approved=False  이유: '4개 이상인데 2개 이하' 기준 해당\n"
    "  → feedback: '4개 계좌 중 3개 이상 사용. 미사용 계좌에 용도에 맞는 소액이라도 배분할 것'\n\n"
    "예시 D: invest_amount=1,500,000원, PorTI 허용 범위 500,000원~1,200,000원\n"
    "  → approved=False  이유: invest_amount가 허용 범위 초과\n"
    "  → feedback: 'invest_amount를 PorTI 허용 최대값 1,200,000원 이하로 줄이고, 차액 300,000원을 allocations에 추가 배분'\n\n"
    "예시 E: CHECKING 계좌 — account_purpose='비상금', comment='식비와 교통비가 커서 생활비를 넉넉히 배분했어요'\n"
    "  → approved=False  이유: comment는 '생활비 배분'을 설명하는데 account_purpose='비상금'으로 내용이 불일치\n"
    "  → feedback: 'CHECKING 계좌 account_purpose를 \"생활비\"로 수정. comment가 생활비 배분을 설명하고 있어 용도도 \"생활비\"여야 함'\n\n"
    "예시 F: PARKING 계좌 — account_purpose='저축', comment='비상금이 3만원뿐이라 우선 채워드렸어요'\n"
    "  → approved=False  이유: comment는 '비상금 충전'을 설명하는데 account_purpose='저축'으로 내용이 불일치\n"
    "  → feedback: 'PARKING 계좌 account_purpose를 \"비상금\"으로 수정. comment가 비상금 충전을 설명하고 있어 용도도 \"비상금\"이어야 함'\n\n"
    "[2단계] approved=True일 때만 수행\n"
    "입력으로 제공된 [계획 reasoning]은 참고용입니다. "
    "그 내용을 그대로 쓰지 말고, 최종 배분 결과만을 근거로 새로 작성하세요.\n\n"
    "  A. reasoning 재작성 (최종 배분 결과 기준, 완전 새로 작성):\n"
    "     - 소비 패턴에서 가장 큰 항목과 실제 금액·비율을 언급\n"
    "     - 투자금을 해당 금액으로 설정한 이유를 PorTI 성향과 저축 여력에 연결해 설명\n"
    "     - 각 계좌에 해당 금액을 배분한 핵심 이유를 구체적 수치와 함께 1~2가지 설명\n"
    "     → 2~3문장, '~했어요', '~드렸어요' 부드러운 경어체\n"
    "     → JUDO·CYCLING 등 PorTI 유형 코드명은 절대 쓰지 말 것\n\n"
    "  B. 각 계좌 comment 재작성 (최종 배분 결과 기준, 완전 새로 작성):\n"
    "     - 해당 계좌의 실제 배분 금액과 용도만을 근거로 새로 작성\n"
    "     - 구체적인 수치(금액 또는 비율)를 반드시 포함한 1문장\n"
    "     - 계획 단계 comment를 재활용하지 말 것\n\n"
    "  C. 극단적 배분 교정 (인라인 수정, 재거부 불필요):\n"
    "     - 단일 항목이 배분 가능액의 60%를 초과하면 초과분을 다른 계좌로 분산\n"
    "     - 금액이 50,000원 미만인 항목은 가장 큰 항목에 합산하고 해당 항목 제거\n"
    "     - 조정 후 합계가 배분 가능액과 반드시 일치 (천원 단위)\n\n"
    "[approved=False일 때]\n"
    "  - feedback: 재계획 시 반드시 지켜야 할 구체적 수정 지시 "
    "(예: '계좌 2개 모두 사용하고 CHECKING에 생활비, PARKING에 비상금으로 분산할 것')\n"
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
