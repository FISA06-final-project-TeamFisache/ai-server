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
    "당신은 개인 자산관리 전문가입니다.\n\n"
    "[역할]\n"
    "사용자의 투자 성향·관심사·보유 계좌를 종합해 투자 흐름을 설계하고,\n"
    "각 흐름에 맞는 모으기 계좌를 선택하세요.\n\n"
    "[내부 분석 절차 — JSON에 포함하지 않음]\n"
    "  1단계: 투자 성향이 시사하는 위험 허용 수준 파악\n"
    "  2단계: 관심사(interest, invest_interests)가 가리키는 자산군·테마 파악\n"
    "  3단계: 월 투자금 규모로 흐름 수·비중 배분 결정\n\n"
    "[흐름 설계 기준]\n"
    "기간 분류:\n"
    "- 단기 (investment_months 3~18): DEPOSIT 또는 PARKING 권장\n"
    "- 중기 (investment_months 24~84): 안정 성향 → SAVING, 투자 성향 → ISA\n"
    "- 장기 (investment_months 120~360): IRP 또는 PENSION_SAVINGS 권장\n\n"
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
    "- title: 관심사·성향 반영한 구체적 목표명 (예: '미국 빅테크 중기 성장')\n"
    "- summary: 관심사·PorTI 성향 언급 포함 1~2문장\n"
    "- reasoning: 이 흐름의 기간·비중 선택 근거 2~3문장\n\n"
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
            f"관심사: {state['interest']}\n"
            f"투자 관심 분야: {', '.join(state['invest_interests']) or '없음'}\n"
            f"월 투자금: {state['invest_amount']:,}원\n\n"
            f"유저 보유 계좌:\n{assets_text}\n\n"
            f"추천 가능 상품:\n{products_text}"
        )),
    ]


async def _node_planner(state: AssetPortfolioState) -> dict:
    """Planner 노드: LLM 흐름 설계 + 계좌 선택 → 투자 금액·투자 가능 여부 계산."""
    ai_result = await ainvoke_structured(_planner_messages(state), _FlowPlansOutput)

    raw_flows = (
        [f.model_dump() for f in ai_result.flows]
        if (ai_result and ai_result.flows)
        else _FALLBACK_FLOWS
    )

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
                gathering_id=UUID(f["gathering_id"]) if f.get("gathering_id") else None,
                gathering_account=(
                    None if f.get("gathering_id") else GatheringAccount(
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
