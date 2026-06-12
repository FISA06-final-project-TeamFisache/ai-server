import logging
import operator
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict
from uuid import UUID

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, model_validator

from app.schemas.portfolio import (
    AssetPortfolioRequest,
    AssetPortfolioResponse,
    GatheringAccount,
    InvestmentPlan,
    PortfolioItem,
)
from app.services.agent.gather_products import GATHER_PRODUCTS
from app.services.agent.llm import ainvoke_structured, get_llm
from app.services.agent.porti_types import porti_label as _porti_label
from app.services.agent.tools import (
    calculate_hrp_weights,
    compound_interest,
    normalize_ratios,
    search_etfs,
)

logger = logging.getLogger(__name__)


# ── 상수 ──────────────────────────────────────────────────────────────────────

_AccountType = Literal[
    "CHECKING", "PARKING", "CMA", "SAVING", "DEPOSIT",
    "ISA", "IRP", "PENSION_SAVINGS",
]

_CAN_INVEST_TYPES: frozenset[str] = frozenset({"ISA", "IRP", "PENSION_SAVINGS"})

# 노후 버킷 — 구조적 가드용
_RETIRE_TYPES: frozenset[str] = frozenset({"IRP", "PENSION_SAVINGS"})

# 노후 흐름 누락 시 주입할 소액 비중(%). FP 원칙: 노후는 작게라도 항상.
# (흐름은 0%로 넣을 수 없어 필요한 단일 상수 — 성향별 표가 아님)
_RETIRE_FLOOR = 10

# 노후 버킷 누락 시 주입할 기본 흐름 (PENSION_SAVINGS, GATHER_PRODUCTS 기반)
_RETIRE_INJECT_TEMPLATE: dict = {
    "flow_type": "노후", "term": "장기", "investment_months": 240,
    "account_type": "PENSION_SAVINGS",
    "invest_strategy": "장기 분산 ETF로 노후 대비",
    "title": "노후 대비 연금",
    "summary": "당장의 목표와 함께, 노후 준비도 소액으로 미리 시작해두는 구성이에요.",
    "reasoning": "지금의 목표가 가장 급하시겠지만, 노후 자금은 일찍 시작할수록 복리와 세액공제 효과가 크게 붙어요. 그래서 부담되지 않는 작은 금액만 연금저축에 미리 담아두시길 권해드려요. 시간이 가장 큰 무기랍니다.",
    "ratio": 0, "gathering_asset_id": None, "has_user_account": False,
    "gathering_account_name": "우리투자증권 연금저축계좌",
    "gathering_account_institution": "우리투자증권",
    "gathering_account_interest_rate": 0.0,
}

# LLM 실패 시 기본 흐름 — gathering_account 정보 포함
_FALLBACK_FLOWS: list[dict] = [
    {
        "flow_type": "단기", "term": "단기", "investment_months": 6,
        "account_type": "DEPOSIT", "invest_strategy": "",
        "title": "단기 유동성", "summary": "단기 유동성 확보", "reasoning": "",
        "ratio": 20, "gathering_asset_id": None, "has_user_account": False,
        "gathering_account_name": "WON플러스예금",
        "gathering_account_institution": "우리은행",
        "gathering_account_interest_rate": 2.15,
    },
    {
        "flow_type": "중기", "term": "중기", "investment_months": 60,
        "account_type": "ISA", "invest_strategy": "지수 추종 ETF로 중기 성장",
        "title": "중기 목표", "summary": "5년 목표 달성", "reasoning": "",
        "ratio": 30, "gathering_asset_id": None, "has_user_account": False,
        "gathering_account_name": "우리투자증권 중개형 ISA",
        "gathering_account_institution": "우리투자증권",
        "gathering_account_interest_rate": 0.0,
    },
    {
        "flow_type": "장기1", "term": "장기", "investment_months": 240,
        "account_type": "PENSION_SAVINGS", "invest_strategy": "장기 분산 ETF",
        "title": "연금저축 노후 대비", "summary": "20년 노후 준비", "reasoning": "",
        "ratio": 25, "gathering_asset_id": None, "has_user_account": False,
        "gathering_account_name": "우리투자증권 연금저축계좌",
        "gathering_account_institution": "우리투자증권",
        "gathering_account_interest_rate": 0.0,
    },
    {
        "flow_type": "장기2", "term": "장기", "investment_months": 240,
        "account_type": "IRP", "invest_strategy": "채권 혼합 ETF",
        "title": "IRP 노후 대비", "summary": "20년 노후 준비", "reasoning": "",
        "ratio": 25, "gathering_asset_id": None, "has_user_account": False,
        "gathering_account_name": "우리투자증권 개인형 IRP",
        "gathering_account_institution": "우리투자증권",
        "gathering_account_interest_rate": 0.0,
    },
]


# ── 스키마 ─────────────────────────────────────────────────────────────────────

class _FlowPlan(BaseModel):
    """Planner LLM 출력: 단일 투자 흐름 — 기간·비중·계좌 선택 포함."""
    flow_type: str
    term: Literal["단기", "중기", "장기"]
    investment_months: int
    account_type: _AccountType
    invest_strategy: str
    title: str
    summary: str
    reasoning: str = ""
    ratio: int
    # 계좌 선택 결과 (유저 계좌 → UUID, 추천 상품 → null)
    gathering_asset_id: str | None = None
    has_user_account: bool = False
    gathering_account_name: str
    gathering_account_institution: str
    gathering_account_interest_rate: float = 0.0


class _FlowPlansOutput(BaseModel):
    """Planner LLM 출력 전체 — account_type 중복 제거 + ratio 합계 100 보정."""
    flows: list[_FlowPlan]

    @model_validator(mode="after")
    def _validate_flows(self) -> "_FlowPlansOutput":
        # account_type 중복 시 ratio 합산 (선순위 흐름 유지)
        seen: dict[str, int] = {}
        deduped: list[_FlowPlan] = []
        for f in self.flows:
            if f.account_type in seen:
                idx = seen[f.account_type]
                deduped[idx] = deduped[idx].model_copy(update={"ratio": deduped[idx].ratio + f.ratio})
            else:
                seen[f.account_type] = len(deduped)
                deduped.append(f)

        # ratio 합계가 100이 아니면 정규화 (반올림 오차는 첫 항목 흡수)
        total = sum(f.ratio for f in deduped)
        if total != 100 and total > 0:
            ratios = [round(f.ratio / total * 100) for f in deduped]
            ratios[0] += 100 - sum(ratios)
            deduped = [f.model_copy(update={"ratio": r}) for f, r in zip(deduped, ratios)]

        self.flows = deduped
        return self


class _AIPortfolioItem(BaseModel):
    """Executor LLM 출력: 단일 ETF — 검색 결과에서 정확히 복사."""
    name: str
    ticker: str
    interest_rate: float = 0.0


class _PortfolioOutput(BaseModel):
    """Executor LLM 출력 전체."""
    portfolio: list[_AIPortfolioItem]


# ── State ─────────────────────────────────────────────────────────────────────

class AssetPortfolioState(TypedDict):
    """메인 그래프 상태."""
    invest_amount: int
    interest: str
    invest_interests: list[str]
    porti_type: str
    porti_comment: str
    asset_list: list[dict]
    flow_plans: list[dict]
    investment_flows: Annotated[list, operator.add]


class FlowState(TypedDict):
    """흐름별 서브그래프 상태 (executor → verifier 공유)."""
    plan: dict
    shared: dict
    portfolio: list[dict]
    investment_flows: Annotated[list, operator.add]


# ── Planner ───────────────────────────────────────────────────────────────────

_SYSTEM_PLANNER = (
    "당신은 개인 자산관리 전문가(FP)입니다.\n\n"
    "[역할]\n"
    "사용자의 투자 성향·인생 목표·보유 계좌를 종합해 투자 흐름을 설계하고,\n"
    "각 흐름에 맞는 모으기 계좌를 선택하세요.\n\n"
    "[입력 데이터 해석]\n"
    "- interest = 사용자의 '인생 목표'입니다 (예: 내집마련, 결혼, 노후, 자동차). 흐름 설계의 핵심 기준입니다.\n"
    "- invest_interests = 투자 관심 자산군·테마입니다 (예: 미국주식, 반도체). ETF 전략 참고용일 뿐, 흐름 구조와 무관합니다.\n\n"
    "[내부 분석 절차 — JSON에 포함하지 않음]\n"
    "  1단계: 투자 성향이 시사하는 위험 허용 수준 파악\n"
    "  2단계: interest(인생 목표)가 요구하는 자금 규모·달성 기간 파악\n"
    "  3단계: 월 투자금 규모로 흐름 수·비중 배분 결정\n\n"
    "[목표 기반 설계 규칙 — FP 필수]\n"
    "1. 목표와 성향이 충돌해도 목표를 포기하지 마세요. 성향이 '단기'여도 목표가 목돈·장기 자금을 요구하면\n"
    "   중기(ISA)·장기(IRP/PENSION_SAVINGS) 흐름을 추가하세요.\n"
    "2. 목표별 계좌 매칭:\n"
    "   - 내집마련·전세·목돈 마련 → ISA + 예적금 중심 (ISA는 비과세·유연해 3~7년 자금에 적합)\n"
    "     ※ IRP·연금저축은 만 55세까지 인출이 묶이므로 주택 자금에는 큰 비중 배분 금지\n"
    "   - 노후·은퇴 → IRP·PENSION_SAVINGS (연말정산 세액공제). 목표가 달라도 노후 대비 소액 흐름은 항상 1개 포함\n"
    "   - 결혼·자동차 등 단기 목돈 → DEPOSIT/SAVING 중심, 여력 있으면 ISA 일부\n"
    "3. 비중은 성향을 존중하되, 활성 목표(interest) 흐름에 가장 큰 비중을 두세요.\n"
    "4. 목표가 비어 있거나('없음') 불명확하면 성향에 따른 기본 배분을 따르세요.\n\n"
    "[흐름 설계 기준]\n"
    "기간 분류:\n"
    "- 단기 (investment_months 3~18): DEPOSIT 또는 PARKING 권장\n"
    "- 중기 (investment_months 24~84): 안정 성향 → SAVING, 투자 성향·목돈 마련(집 등) → ISA\n"
    "- 장기 (investment_months 120~360): 노후·은퇴 목적 → IRP 또는 PENSION_SAVINGS\n\n"
    "계좌 종류:\n"
    "- 적립 전용 (투자 상품 불가): CHECKING, PARKING, CMA, SAVING, DEPOSIT\n"
    "- 투자 상품 가능: ISA, IRP, PENSION_SAVINGS\n\n"
    "[계좌 선택 규칙]\n"
    "1. 유저 보유 계좌에 해당 account_type이 있으면 → 그 계좌 사용\n"
    "   - gathering_asset_id = 해당 asset_id (정확히 복사)\n"
    "   - has_user_account = true\n"
    "   - gathering_account_institution = \"\" (정보 없음)\n"
    "   - 동일 account_type은 흐름 전체에서 1개만 할당\n"
    "2. 보유 계좌 없으면 → 추천 가능 상품에서 product_type 일치 항목 선택\n"
    "   - gathering_asset_id = null\n"
    "   - has_user_account = false\n"
    "   - gathering_account_name·institution·interest_rate = 상품 정보 그대로\n\n"
    "[출력 제약]\n"
    "- ratio 합계 반드시 100\n"
    "- 각 account_type은 흐름 전체에서 최대 1회 사용\n"
    "- invest_strategy: 적립 전용 계좌는 빈 문자열, 투자 가능 계좌는 ETF 전략 1문장\n"
    "- title: 인생 목표·성향 반영한 구체적 목표명 (예: '내집마련 ISA 비과세 시드', '노후 대비 연금')\n"
    "- summary: 카드에 기본으로 보이는 1~2문장. 한눈에 '내 상황에 맞춰 분석됐다'고 느껴지게 개인화하세요.\n"
    "  · 사용자의 실제 인생 목표와 PorTI 성향을 구체적으로 직접 언급 (일반론·템플릿식 표현 금지)\n"
    "  · 이 계좌·기간을 고른 핵심 이유 한 가지를 자연스럽게 포함\n"
    "  · 친근한 상담 말투(~요/~네요)로, AI가 내 상황을 읽고 골라준 듯한 인상\n"
    "- reasoning: 사용자에게 직접 이야기하듯 3~4문장의 개인화된 FP 상담. 아래를 구체적으로 녹이세요:\n"
    "  · 이 인생 목표에 왜 이 계좌·기간(개월 수 언급)·비중이 적합한지\n"
    "  · 선택한 계좌의 실질 혜택(ISA 비과세, 연금 세액공제, 예적금 안정성 등)을 일상어로 풀어서\n"
    "  · PorTI 성향과 어떻게 어울리는지 — 목표와 성향이 충돌하면 그 해소 논리를 부드럽게 설명\n"
    "  · 따뜻하고 신뢰감 있는 말투(~요/~네요/~랍니다)로, 사용자를 안심시키는 한마디로 마무리\n"
    "  딱딱한 한자어·사전식 정의 나열 금지. 사용자의 상황에 맞춰 구체적으로.\n\n"
    "반드시 JSON만 응답."
)


def _planner_messages(state: AssetPortfolioState) -> list:
    """Planner LLM 입력 메시지: 사용자 프로파일 + 보유 계좌 + 추천 상품 목록."""
    assets_text = "\n".join(
        f"  - asset_id={a['asset_id']}, type={a['asset_type']}, "
        f"name={a['account_name']}, balance={a['balance']:,}원"
        for a in state["asset_list"]
    ) or "  없음"

    products_text = "\n".join(
        f"  - product_type={p['product_type']}, name={p['name']}, "
        f"institution={p['institution']}, interest_rate={p['interest_rate'] or 0.0}%"
        for p in GATHER_PRODUCTS
    )

    return [
        SystemMessage(content=_SYSTEM_PLANNER),
        HumanMessage(content=(
            f"PorTI 유형: {_porti_label(state['porti_type'])}\n"
            f"투자 성향 설명: {state['porti_comment']}\n"
            f"인생 목표(interest): {state['interest'] or '없음'}\n"
            f"투자 관심 자산군(invest_interests): {', '.join(state['invest_interests']) or '없음'}\n"
            f"월 투자금: {state['invest_amount']:,}원\n\n"
            f"유저 보유 계좌:\n{assets_text}\n\n"
            f"추천 가능 상품:\n{products_text}"
        )),
    ]


def _build_retire_flow(state: AssetPortfolioState) -> dict:
    """노후 흐름 주입용. 사용자 목표(interest)를 끼워넣어 정적 문구가 일반 템플릿처럼 보이지 않게 한다.
    계좌 id는 이후 재탐색 단계에서 확정하므로 여기선 보유한 노후 계좌 종류만 맞춘다.
    """
    flow = dict(_RETIRE_INJECT_TEMPLATE)

    goal = (state.get("interest") or "").strip()
    if goal:
        flow["summary"] = f"'{goal}' 목표에 집중하면서도, 노후는 작게라도 지금부터 함께 챙겨두는 구성이에요."
        flow["reasoning"] = (
            f"지금은 '{goal}'이 가장 급하시겠지만, 노후 자금은 일찍 시작할수록 복리와 세액공제 효과가 크게 붙어요. "
            f"그래서 '{goal}' 자금에 집중하시되, 부담되지 않는 작은 금액만 연금저축에 함께 담아두시길 권해드려요. "
            "시간이 가장 큰 무기랍니다."
        )

    # 유저가 보유한 노후 계좌 종류(IRP/연금)에 맞춤 — 실제 id는 재탐색 단계가 확정
    for a in state["asset_list"]:
        if a["asset_type"] in _RETIRE_TYPES:
            flow["account_type"] = a["asset_type"]
            break

    return flow


def _apply_structural_guard(flows: list[dict], state: AssetPortfolioState) -> list[dict]:
    """얇은 구조적 가드 — 비중은 LLM에 맡기고 '노후 흐름 존재'만 결정론적으로 보장.

    LLM이 노후(IRP/PENSION_SAVINGS) 흐름을 누락하면 소액(_RETIRE_FLOOR%) 흐름을
    주입하고 그만큼 기존 흐름 비중을 비례 축소해 합계 100을 유지한다.
    이미 노후 흐름이 있거나 흐름이 비어 있으면 LLM 출력을 그대로 둔다.
    """
    if not flows:
        return flows
    if any(f["account_type"] in _RETIRE_TYPES for f in flows):
        return flows

    flows = [dict(f) for f in flows]
    injected = _build_retire_flow(state)

    remaining = 100 - _RETIRE_FLOOR
    total = sum(f["ratio"] for f in flows) or 1
    for f in flows:
        f["ratio"] = round(f["ratio"] / total * remaining)
    injected["ratio"] = _RETIRE_FLOOR
    flows.append(injected)

    # 반올림 오차는 첫 흐름이 흡수해 합계 100 유지
    diff = 100 - sum(f["ratio"] for f in flows)
    if diff:
        flows[0]["ratio"] += diff

    logger.info("구조적 가드: 노후 흐름 누락 → 소액 %d%% 주입", _RETIRE_FLOOR)
    return flows


async def _node_planner(state: AssetPortfolioState) -> dict:
    """Planner 노드: LLM 흐름 설계 + 계좌 선택 → 투자 금액·투자 가능 여부 계산."""
    ai_result = await ainvoke_structured(_planner_messages(state), _FlowPlansOutput)

    raw_flows = (
        [f.model_dump() for f in ai_result.flows]
        if (ai_result and ai_result.flows)
        else _FALLBACK_FLOWS
    )
    raw_flows = _apply_structural_guard(raw_flows, state)

    # gathering_asset_id는 LLM 출력을 신뢰하지 않는다 — 백엔드가 보낸 asset_list가 source of truth.
    # 같은 account_type 계좌가 여러 개면: LLM이 고른 id가 실제 목록에 있고 종류가 맞을 때만 그 선택을 존중,
    # 아니면 잔액이 가장 큰 계좌로 결정론적 선택. 해당 종류 계좌가 없으면 추천 상품 경로로 둔다.
    assets_by_type: dict[str, list[dict]] = {}
    for a in state["asset_list"]:
        assets_by_type.setdefault(a["asset_type"], []).append(a)
    asset_by_id = {a["asset_id"]: a for a in state["asset_list"]}

    for f in raw_flows:
        candidates = assets_by_type.get(f["account_type"], [])
        if candidates:
            chosen = asset_by_id.get(f.get("gathering_asset_id") or "")
            if chosen is None or chosen["asset_type"] != f["account_type"]:
                # LLM 선택이 무효(환각)거나 종류 불일치 → 잔액 최대 계좌로 확정
                chosen = max(candidates, key=lambda a: a["balance"])
            f["gathering_asset_id"] = chosen["asset_id"]
            f["has_user_account"] = True
            f["gathering_account_name"] = chosen["account_name"]
            f["gathering_account_institution"] = ""
        else:
            # 미보유 — 추천 상품 경로 (LLM이 채운 상품 정보 유지)
            f["gathering_asset_id"] = None
            f["has_user_account"] = False

    flow_plans = [
        {
            **f,
            "amount": round(state["invest_amount"] * f["ratio"] / 100),
            "can_invest": f["account_type"] in _CAN_INVEST_TYPES,
            "gathering_account": {
                "name": f["gathering_account_name"],
                "type": f["account_type"],
                "institution": f["gathering_account_institution"],
                "interest_rate": f["gathering_account_interest_rate"],
            },
        }
        for f in raw_flows
    ]

    return {"flow_plans": flow_plans}


# ── Executor ──────────────────────────────────────────────────────────────────

_SYSTEM_EXECUTOR = (
    "포트폴리오 전문가입니다.\n\n"
    "주어진 투자 흐름에 맞는 ETF 포트폴리오를 다음 순서로 설계하세요.\n\n"
    "1단계 — 슬롯 설계 (최대 3개)\n"
    "  이 흐름의 기간·성격에 따라 필요한 역할을 먼저 결정하세요.\n"
    "  예) 단기 안정형 → 코어(단기채권) + 보조(단기자금)\n"
    "      장기 성장형 → 코어(글로벌 주식) + 위성(테마) + 헤지(채권)\n\n"
    "2단계 — 슬롯별 검색\n"
    "  각 슬롯에 맞는 키워드로 search_etfs를 호출하세요.\n"
    "  결과 대부분의 interest_rate가 음수(0 미만)라면:\n"
    "    a) keywords=[] 로 재검색해 수익률 상위 ETF 목록을 받으세요.\n"
    "    b) 재검색 결과도 비어 있으면 첫 검색 결과로 돌아가 interest_rate 가장 높은 항목을 선택하세요.\n\n"
    "3단계 — 선택·출력\n"
    "  검색 결과에서 슬롯별 최적 ETF를 1개씩 선택해 JSON으로 출력하세요.\n"
    "  interest_rate 음수 종목은 양수 대안이 없을 때만 선택하세요.\n\n"
    "제약:\n"
    "- IRP·PENSION_SAVINGS 계좌: 주식·채권·해외 자산군 반드시 분산\n"
    "- name·ticker·interest_rate는 검색 결과의 정확한 값만 사용 (변형 금지)\n"
    '- 최종 출력: {"portfolio":[{"name":"정확한상품명","ticker":"코드","interest_rate":0.0}]}'
)


def _build_executor_ctx(plan: dict, shared: dict) -> str:
    """Executor 에이전트 입력: 흐름 정보 + 사용자 성향·관심사."""
    return (
        f"계좌 유형: {plan['account_type']}\n"
        f"흐름: {plan['flow_type']} ({plan['term']}, {plan['investment_months']}개월)\n"
        f"투자 전략: {plan['invest_strategy'] or '흐름 성격에 맞게 분산 구성'}\n"
        f"PorTI: {_porti_label(shared['porti_type'])} / {shared['porti_comment']}\n"
        f"사용자 관심사: {', '.join(shared['invest_interests']) or '없음'}"
    )


async def _node_executor(flow_state: FlowState) -> dict:
    """Executor 노드: create_agent로 흐름별 ETF 검색·선택. 투자 불가 계좌는 즉시 빈 포트폴리오 반환."""
    plan = flow_state["plan"]
    shared = flow_state["shared"]

    if not plan["can_invest"]:
        return {**flow_state, "portfolio": []}

    agent = create_agent(
        model=get_llm(),
        tools=[search_etfs],
        system_prompt=_SYSTEM_EXECUTOR,
        response_format=_PortfolioOutput,
    )
    portfolio: list[dict] = []
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=_build_executor_ctx(plan, shared))]},
            config={"recursion_limit": 8},
        )
        structured = result.get("structured_response")
        if structured:
            portfolio = [
                {
                    "name": item.name,
                    "ticker": item.ticker,
                    "product_type": "ETF",
                    "interest_rate": item.interest_rate,
                }
                for item in structured.portfolio
            ]
    except Exception as e:
        logger.warning("[%s] Executor 실패: %s", plan["flow_type"], e)

    if not portfolio:
        logger.info("[%s] ETF 선택 실패 → 유사도 검색 폴백", plan["flow_type"])
        fallback_rows = await search_etfs.ainvoke({"keywords": list(shared["invest_interests"])})
        if fallback_rows:
            portfolio = [
                {
                    "name": r["name"],
                    "ticker": r.get("ticker", ""),
                    "product_type": "ETF",
                    "interest_rate": float(r.get("interest_rate") or 0.0),
                }
                for r in fallback_rows[:3]
            ]

    return {**flow_state, "portfolio": portfolio}


# ── Verifier ──────────────────────────────────────────────────────────────────

async def _node_verifier(flow_state: FlowState) -> dict:
    """Verifier 노드: HRP 비중 최적화 → 기대 수익 계산 → investment_flow 최종 조립."""
    plan = flow_state["plan"]
    portfolio = list(flow_state.get("portfolio", []))

    # HRP 비중 최적화 (tool이 내부적으로 균등 배분 폴백을 처리함)
    expected_rr_from_hrp: float | None = None
    if len(portfolio) >= 2:
        tickers = [item["ticker"] for item in portfolio if item.get("ticker")]
        if tickers:
            logger.info("HRP: [%s] | 티커: %s", plan["flow_type"], tickers)
            hrp_result = await calculate_hrp_weights.ainvoke({"tickers": tickers})
            weight_map = hrp_result["weights"]
            portfolio = normalize_ratios([
                {**item, "ratio": weight_map.get(item.get("ticker", ""), 0)}
                for item in portfolio
            ])
            expected_rr_from_hrp = hrp_result.get("expected_annual_return_pct")
    elif len(portfolio) == 1:
        portfolio = [{**portfolio[0], "ratio": 100}]

    # 기대 수익률: HRP portfolio_performance 우선, 없으면 모으기 계좌 금리
    ga = plan["gathering_account"]
    if expected_rr_from_hrp and expected_rr_from_hrp > 0:
        expected_rr = expected_rr_from_hrp
    else:
        expected_rr = float(ga.get("interest_rate", 0.0) or 0.0)
        if not portfolio and expected_rr == 0.0:
            logger.warning("[%s] gathering_account interest_rate=0.0 — DB 확인 필요", plan.get("flow_type", ""))

    compound_result = await compound_interest.ainvoke({
        "monthly_amount": plan["amount"],
        "annual_rate_pct": expected_rr,
        "months": plan["investment_months"],
    })
    expected_amount = compound_result["expected_amount"]
    months = plan["investment_months"]

    return {
        "investment_flows": [{
            "flow_type": plan["flow_type"],
            "title": plan["title"],
            "term": plan["term"],
            "summary": plan["summary"],
            "reasoning": plan.get("reasoning", ""),
            "gathering_id": plan["gathering_asset_id"],
            "gathering_account": ga,
            "amount": plan["amount"],
            "portfolio": [
                {
                    "type": item.get("product_type", "ETF"),
                    "name": item["name"],
                    "ticker": item["ticker"],
                    "ratio": item["ratio"],
                    "interest_rate": float(item.get("interest_rate") or 0.0),
                }
                for item in portfolio
            ],
            "expected_rr_pct": round(expected_rr, 1),
            "investment_months": months,
            "expected_amount": round(expected_amount),
            "rr_comment": (
                f"'{plan['title']}' 목표로 {months}개월 적립 시 "
                f"연 {expected_rr:.1f}% 기준 약 {round(expected_amount):,}원 예상됩니다."
                if expected_rr > 0
                else f"'{plan['title']}' 목표로 {months}개월 적립 시 약 {round(expected_amount):,}원 누적 예상됩니다."
            ),
        }]
    }


# ── LangGraph 조립 ────────────────────────────────────────────────────────────

def _route_flows(state: AssetPortfolioState) -> list[Send]:
    """planner 출력의 flow_plans를 흐름별 Send로 변환해 병렬 실행."""
    shared = {
        "porti_type": state["porti_type"],
        "porti_comment": state["porti_comment"],
        "interest": state["interest"],
        "invest_interests": state["invest_interests"],
    }
    return [
        Send("flow_branch", {
            "plan": plan,
            "shared": shared,
            "portfolio": [],
            "investment_flows": [],
        })
        for plan in state["flow_plans"]
    ]


def _build_graph() -> StateGraph:
    """PEV 패턴 그래프 조립: planner → (executor → verifier) 흐름별 병렬."""
    # 흐름별 서브그래프
    flow_sg = StateGraph(FlowState)
    flow_sg.add_node("executor", _node_executor)
    flow_sg.add_node("verifier", _node_verifier)
    flow_sg.set_entry_point("executor")
    flow_sg.add_edge("executor", "verifier")
    flow_sg.add_edge("verifier", END)

    # 메인 그래프
    graph = StateGraph(AssetPortfolioState)
    graph.add_node("planner", _node_planner)
    graph.add_node("flow_branch", flow_sg.compile())
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", _route_flows, ["flow_branch"])
    graph.add_edge("flow_branch", END)

    return graph.compile()


_graph = _build_graph()


# ── Entry point ───────────────────────────────────────────────────────────────

def _safe_uuid(value) -> UUID | None:
    """gathering_id를 UUID로 안전 파싱. 잘못된 값이면 None (500 방지)."""
    if not value:
        return None
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        logger.warning("유효하지 않은 gathering_id 무시: %r", value)
        return None


async def recommend_asset_portfolio(request: AssetPortfolioRequest) -> AssetPortfolioResponse:
    """외부 진입점: 요청 변환 → PEV 그래프 실행 → 응답 반환."""
    asset_list = [
        {
            "asset_id": str(a.asset_id),
            "asset_type": a.asset_type,
            "account_name": a.account_name,
            "balance": a.balance,
        }
        for a in request.invest_assets
    ]

    final_state: AssetPortfolioState = await _graph.ainvoke({
        "invest_amount": request.invest_amount,
        "interest": request.interest,
        "invest_interests": request.invest_interests,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "asset_list": asset_list,
        "flow_plans": [],
        "investment_flows": [],
    })

    return AssetPortfolioResponse(
        created_at=datetime.now(timezone.utc),
        investment_flows=[
            InvestmentPlan(
                title=f["title"],
                term=f["term"],
                summary=f["summary"],
                reasoning=f.get("reasoning", ""),
                gathering_id=_safe_uuid(f.get("gathering_id")),
                gathering_account=(
                    None if _safe_uuid(f.get("gathering_id")) else GatheringAccount(
                        name=f["gathering_account"].get("name", "자유적금"),
                        type=f["gathering_account"].get("type", "SAVING"),
                        institution=f["gathering_account"].get("institution", ""),
                        interest_rate=float(f["gathering_account"].get("interest_rate", 0.0)),
                    )
                ),
                amount=f["amount"],
                portfolio=[
                    PortfolioItem(
                        type=p["type"],
                        name=p["name"],
                        ticker=p["ticker"],
                        ratio=p["ratio"],
                        interest_rate=p["interest_rate"],
                    )
                    for p in f["portfolio"]
                ],
                expected_rr_pct=f["expected_rr_pct"],
                investment_months=f["investment_months"],
                expected_amount=float(f["expected_amount"]),
                rr_comment=f["rr_comment"],
            )
            for f in final_state["investment_flows"]
        ],
    )
