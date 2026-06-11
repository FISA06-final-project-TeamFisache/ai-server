import asyncio
import json
import logging
import operator
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from app.schemas.portfolio import (
    AssetPortfolioRequest,
    AssetPortfolioResponse,
    GatheringAccount,
    InvestmentPlan,
    PortfolioItem,
)
from app.services.agent.llm import ainvoke_structured, get_llm
from app.services.agent.tools import (
    normalize_ratios,
    compound_interest,
    calculate_hrp_weights,
    search_etfs,
)
from app.services.agent.porti_types import (
    porti_label as _porti_label,
    STABLE_PORTI_TYPES,
    NEUTRAL_PORTI_TYPES,
    INVEST_PORTI_TYPES,
)
from app.services.agent.gather_products import GATHER_PRODUCTS

logger = logging.getLogger(__name__)

_WOORI_BANK = "우리은행"
_WOORI_INVEST = "우리투자증권"
_CAN_INVEST_TYPES = {"ISA", "IRP", "PENSION_SAVINGS"}

_AccountType = Literal[
    "CHECKING", "PARKING", "CMA", "SAVING", "DEPOSIT",
    "ISA", "IRP", "PENSION_SAVINGS",
]



# ── AI 출력 스키마 ─────────────────────────────────────────────────────────────

class _FlowPlan(BaseModel):
    flow_type: str
    term: Literal["단기", "중기", "장기"]
    investment_months: int
    account_type: _AccountType
    invest_strategy: str
    title: str
    summary: str
    reasoning: str = ""
    ratio: int


class _FlowPlansOutput(BaseModel):
    flows: list[_FlowPlan]


class _AIPortfolioItem(BaseModel):
    name: str
    ticker: str


class _PortfolioOutput(BaseModel):
    portfolio: list[_AIPortfolioItem]


# ── State ─────────────────────────────────────────────────────────────────────

class AssetPortfolioState(TypedDict):
    invest_amount: int
    interest: str
    invest_interests: list[str]
    porti_type: str
    porti_comment: str
    asset_list: list[dict]
    flow_plans: list[dict]
    investment_flows: Annotated[list, operator.add]


class _FlowSubgraphOutput(TypedDict):
    investment_flows: list


class FlowExecuteState(TypedDict):
    plan: dict
    shared: dict
    portfolio: list[dict]
    investment_flows: Annotated[list, operator.add]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_user_asset(asset_by_type: dict, account_type: str, used_ids: set) -> dict | None:
    for a in asset_by_type.get(account_type, []):
        if a["asset_id"] not in used_ids:
            return a
    return None


def _find_best_product(account_type: str, prefer_institution: str) -> dict | None:
    candidates = [p for p in GATHER_PRODUCTS if p["product_type"] == account_type]
    if not candidates:
        return None
    woori = [p for p in candidates if prefer_institution in (p.get("institution") or "")]
    return woori[0] if woori else candidates[0]


def _validate_and_merge(
    raw_items: list[_AIPortfolioItem],
    confirmed_by_name: dict[str, dict],
) -> list[dict]:
    validated: list[dict] = []
    for item in raw_items:
        name = item.name
        if name not in confirmed_by_name:
            logger.warning("상품명 목록 미존재, 제거: %s", name)
            continue
        p = confirmed_by_name[name]
        validated.append({
            "name": name,
            "ticker": p.get("ticker") or item.ticker or "",
            "product_type": p.get("product_type", "ETF"),
            "interest_rate": float(p.get("interest_rate") or 0.0),
        })

    n = len(validated)
    if n > 0:
        base = 100 // n
        rem = 100 - base * n
        for i, item in enumerate(validated):
            item["ratio"] = base + (1 if i < rem else 0)

    return validated



# ── Node: plan_flows ──────────────────────────────────────────────────────────

_FALLBACK_FLOWS: list[_FlowPlan] = [
    _FlowPlan(flow_type="단기", term="단기", investment_months=6,
              account_type="DEPOSIT", invest_strategy="",
              title="단기 유동성", summary="단기 유동성 확보", ratio=20),
    _FlowPlan(flow_type="중기", term="중기", investment_months=60,
              account_type="ISA", invest_strategy="지수 추종 ETF로 중기 성장",
              title="중기 목표", summary="5년 목표 달성", ratio=30),
    _FlowPlan(flow_type="장기1", term="장기", investment_months=240,
              account_type="PENSION_SAVINGS", invest_strategy="장기 분산 ETF",
              title="연금저축 노후 대비", summary="20년 노후 준비", ratio=25),
    _FlowPlan(flow_type="장기2", term="장기", investment_months=240,
              account_type="IRP", invest_strategy="채권 혼합 ETF",
              title="IRP 노후 대비", summary="20년 노후 준비", ratio=25),
]


async def _plan_flows(state: AssetPortfolioState) -> dict:
    asset_by_type: dict[str, list[dict]] = {}
    for a in state["asset_list"]:
        asset_by_type.setdefault(a["asset_type"], []).append(a)

    messages = [
        SystemMessage(content=(
            "당신은 개인 자산관리 전문가입니다.\n"
            "흐름을 설계하기 전에 다음 단계를 내부적으로 분석하세요 (JSON 출력에는 포함하지 않음).\n\n"
            "[내부 분석 절차]\n"
            "  1단계. 사용자 프로파일 파악\n"
            "    · 투자 성향이 시사하는 위험 허용 수준은 무엇인가?\n"
            "    · 관심사(interest, invest_interests)가 어떤 자산군·지역·테마를 가리키는가?\n"
            "    · 월 투자금 규모로 흐름을 몇 개로 나누는 것이 현실적인가?\n\n"
            "  2단계. 흐름 구조 결정\n"
            "    · 단기·중기·장기 각 흐름이 필요한 이유는?\n"
            "    · 각 흐름의 기간(investment_months)을 왜 그 범위로 설정하는가?\n\n"
            "  3단계. 비중 배분 근거\n"
            "    · 투자 성향을 고려할 때 단기:중기:장기 비율을 어떻게 배분하는가?\n"
            "    · 각 흐름의 절대 금액이 해당 목표에 충분한가?\n\n"
            "[흐름 설계 기준]\n"
            "계좌 종류:\n"
            "- 적립 전용 (투자 상품 불가): CHECKING, PARKING, CMA, SAVING, DEPOSIT\n"
            "- 투자 상품 가능: ISA, IRP, PENSION_SAVINGS\n\n"
            "기간 기준:\n"
            "- 단기 (investment_months 3~18): DEPOSIT 또는 PARKING 권장\n"
            "- 중기 (investment_months 24~84): 안정 성향 → SAVING, 투자 성향 → ISA\n"
            "- 장기 (investment_months 120~360): IRP 또는 PENSION_SAVINGS 권장\n\n"
            "필드 기준:\n"
            "- ratio 합계 반드시 100\n"
            "- invest_strategy: 투자 불가 계좌는 빈 문자열, 투자 가능 계좌는 ETF 선택 전략 1문장\n"
            "- 각 account_type은 흐름 전체에서 최대 1회 사용 (중복 금지)\n"
            "- title: 관심사·성향을 반영한 구체적 목표명 (예: '미국 빅테크 중기 성장', '안정형 노후 준비')\n"
            "- summary: 관심사·PorTI 성향 언급 포함 1~2문장\n\n"
            "[흐름별 reasoning 필드]\n"
            "위 내부 분석에서 내린 결론을 이 흐름에 적용해 2~3문장으로 작성하세요.\n"
            "  · 2단계·3단계에서 내린 결론이 이 기간과 비중 선택과 어떻게 연결되는가?\n"
            "  · 사용자의 관심사가 이 흐름 전략에 어떻게 반영됐는가?\n"
            "  계좌명·상품명 언급 없이 순수 전략 관점으로 작성. PorTI 유형명(예: Tortoise, Fox) 언급 금지.\n\n"
            "반드시 JSON만 응답."
        )),
        HumanMessage(content=(
            f"PorTI 유형: {_porti_label(state['porti_type'])}\n"
            f"투자 성향 설명: {state['porti_comment']}\n"
            f"관심사: {state['interest']}\n"
            f"투자 관심 분야: {', '.join(state['invest_interests']) or '없음'}\n"
            f"월 투자금: {state['invest_amount']:,}원"
        )),
    ]
    ai_result = await ainvoke_structured(messages, _FlowPlansOutput)
    raw_flows = ai_result.flows if (ai_result and ai_result.flows) else _FALLBACK_FLOWS

    seen_types: dict[str, int] = {}
    deduped: list[_FlowPlan] = []
    for f in raw_flows:
        if f.account_type in seen_types:
            idx = seen_types[f.account_type]
            merged = deduped[idx].model_copy(update={"ratio": deduped[idx].ratio + f.ratio})
            deduped[idx] = merged
            logger.info("account_type 중복 제거: %s → 비율 %d%%로 합산", f.account_type, merged.ratio)
        else:
            seen_types[f.account_type] = len(deduped)
            deduped.append(f)
    raw_flows = deduped

    raw_ratios = [max(1, f.ratio) for f in raw_flows]
    total = sum(raw_ratios)
    if total != 100:
        normalized = [round(r / total * 100) for r in raw_ratios]
        normalized[0] += 100 - sum(normalized)
    else:
        normalized = raw_ratios

    invest_amount = state["invest_amount"]
    used_ids: set = set()

    flow_plans = []
    for f, ratio in zip(raw_flows, normalized):
        fallback_institution = _WOORI_INVEST if f.account_type in _CAN_INVEST_TYPES else _WOORI_BANK
        user_asset = _find_user_asset(asset_by_type, f.account_type, used_ids)
        if user_asset:
            used_ids.add(user_asset["asset_id"])
        best_product = _find_best_product(f.account_type, fallback_institution)

        gathering_account: dict = {
            "name": (
                best_product["name"] if best_product
                else (user_asset["account_name"] if user_asset else f.account_type)
            ),
            "type": f.account_type,
            "institution": (
                best_product["institution"] if best_product
                else (fallback_institution if not user_asset else "")
            ),
            "interest_rate": float(best_product["interest_rate"] or 0.0) if best_product else 0.0,
        }

        flow_plans.append({
            "flow_type": f.flow_type,
            "term": f.term,
            "investment_months": f.investment_months,
            "account_type": f.account_type,
            "invest_strategy": f.invest_strategy,
            "title": f.title or f"{f.flow_type} 투자 플랜",
            "summary": f.summary,
            "reasoning": f.reasoning,
            "ratio": ratio,
            "amount": round(invest_amount * ratio / 100),
            "can_invest": f.account_type in _CAN_INVEST_TYPES,
            "gathering_asset_id": user_asset["asset_id"] if user_asset else None,
            "has_user_account": user_asset is not None,
            "gathering_account": gathering_account,
        })

    return {"flow_plans": flow_plans}


# ── ETF 검색 + 포트폴리오 구성 (agentic, 슬롯 기반) ──────────────────────────────

_VOLATILITY_CAP: dict[tuple[str, str], float | None] = {
    ("STABLE", "단기"): 10.0,  ("STABLE", "중기"): 15.0,  ("STABLE", "장기"): 20.0,
    ("NEUTRAL", "단기"): 15.0, ("NEUTRAL", "중기"): 25.0, ("NEUTRAL", "장기"): 30.0,
    ("INVEST", "단기"): 25.0,  ("INVEST", "중기"): None,  ("INVEST", "장기"): None,
}


def _resolve_max_volatility(porti_type: str, term: str) -> float | None:
    if porti_type in STABLE_PORTI_TYPES:
        group = "STABLE"
    elif porti_type in NEUTRAL_PORTI_TYPES:
        group = "NEUTRAL"
    else:
        group = "INVEST"
    return _VOLATILITY_CAP.get((group, term))


_SYSTEM_BUILD_PORTFOLIO = (
    "포트폴리오 전문가입니다.\n\n"
    "주어진 투자 흐름에 맞는 포트폴리오를 다음 순서로 설계하세요.\n\n"
    "1단계 — 슬롯 설계 (최대 3개)\n"
    "  이 흐름의 기간·성격에 따라 필요한 역할을 먼저 결정하세요.\n"
    "  예) 단기 안정형 → 코어(단기채권) + 보조(단기자금)\n"
    "      장기 성장형 → 코어(글로벌 주식) + 위성(테마) + 헤지(채권)\n\n"
    "2단계 — 슬롯별 검색\n"
    "  각 슬롯에 맞는 키워드로 search_etfs를 호출하세요.\n\n"
    "3단계 — 선택·출력\n"
    "  검색 결과에서 슬롯별 최적 ETF를 1개씩 선택해 JSON으로 출력하세요.\n\n"
    "제약:\n"
    "- IRP·PENSION_SAVINGS 계좌: 주식·채권·해외 자산군 반드시 분산\n"
    "- name·ticker는 검색 결과의 정확한 값만 사용 (변형 금지)\n"
    '- 최종 출력: {"portfolio":[{"name":"정확한상품명","ticker":"코드"}]}'
)


async def _node_build_portfolio(flow_state: FlowExecuteState) -> FlowExecuteState:
    """슬롯 설계(최대 3개) → 슬롯별 ETF 검색(tool) → 최종 포트폴리오 선택."""
    plan = flow_state["plan"]
    shared = flow_state["shared"]

    if not plan["can_invest"]:
        return {**flow_state, "portfolio": []}

    max_volatility = _resolve_max_volatility(shared["porti_type"], plan["term"])
    base_query_parts = list(shared["invest_interests"])
    all_candidates: dict[str, dict] = {}

    llm = get_llm().bind_tools([search_etfs])

    user_ctx = (
        f"계좌 유형: {plan['account_type']}\n"
        f"흐름: {plan['flow_type']} ({plan['term']}, {plan['investment_months']}개월)\n"
        f"투자 전략: {plan['invest_strategy'] or '흐름 성격에 맞게 분산 구성'}\n"
        f"PorTI: {_porti_label(shared['porti_type'])} / {shared['porti_comment']}\n"
        f"사용자 관심사: {', '.join(shared['invest_interests']) or '없음'}"
    )
    messages: list = [SystemMessage(content=_SYSTEM_BUILD_PORTFOLIO), HumanMessage(content=user_ctx)]

    for _ in range(6):
        response = await llm.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            # max_volatility는 PorTI 규칙 값으로 주입 — LLM이 설정한 값 무시
            args = {**tc["args"], "max_volatility": max_volatility}
            tool_result = await search_etfs.ainvoke(args)
            for item in (tool_result or []):
                all_candidates[item["name"]] = item
            messages.append(ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=tc["id"],
            ))

    result = await ainvoke_structured(
        messages + [HumanMessage(content="검색 결과를 바탕으로 최종 포트폴리오를 JSON으로 출력하세요.")],
        _PortfolioOutput,
    )
    raw_items = result.portfolio if result else []
    portfolio = _validate_and_merge(raw_items, all_candidates)

    if not portfolio:
        logger.info("[%s] 포트폴리오 구성 실패 → 기본 유사도 검색 폴백", plan["flow_type"])
        fallback_rows = await search_etfs.ainvoke({"keywords": base_query_parts, "max_volatility": None})
        if fallback_rows:
            fallback_items = [_AIPortfolioItem(name=r["name"], ticker=r.get("ticker", "")) for r in fallback_rows[:3]]
            portfolio = _validate_and_merge(fallback_items, {r["name"]: r for r in fallback_rows[:3]})

    return {**flow_state, "portfolio": portfolio}


# ── Helpers: expected return ──────────────────────────────────────────────────

def _calc_expected_return(
    portfolio: list[dict],
    plan: dict,
    product_by_name: dict[str, dict],
) -> tuple[float, int]:
    ga = plan["gathering_account"]
    expected_rr = float(ga.get("interest_rate", 0.0) or 0.0)

    if portfolio:
        weighted = sum(
            float(product_by_name.get(item["name"], {}).get("interest_rate") or 0.0)
            * item["ratio"] / 100
            for item in portfolio
        )
        if weighted > 0:
            expected_rr = weighted
        else:
            logger.warning("[%s] 포트폴리오 interest_rate 모두 0.0 — DB 데이터 확인 필요", plan.get("flow_type", ""))
    elif expected_rr == 0.0:
        logger.warning("[%s] gathering_account interest_rate=0.0 — DB 데이터 확인 필요", plan.get("flow_type", ""))

    result = compound_interest.invoke({
        "monthly_amount": plan["amount"],
        "annual_rate_pct": expected_rr,
        "months": plan["investment_months"],
    })
    return expected_rr, result["expected_amount"]


# ── Node: execute_flow (finalize — HRP + narrate + output) ───────────────────

async def execute_flow(flow_state: FlowExecuteState) -> dict:
    """HRP 비중 최적화 → 서사화 → 기대 수익 계산 → 최종 흐름 결과 반환."""
    plan = flow_state["plan"]
    portfolio = list(flow_state.get("portfolio", []))

    if len(portfolio) >= 2:
        tickers = [item["ticker"] for item in portfolio if item.get("ticker")]
        if tickers:
            logger.info("HRP: [%s] 비중 계산 시작 | 티커: %s", plan["flow_type"], tickers)
            equal_ratio_map = {item.get("ticker", ""): item["ratio"] for item in portfolio}

            hrp_result = await calculate_hrp_weights.ainvoke({"tickers": tickers})

            if hrp_result["method"] == "hrp":
                weight_map = hrp_result["weights"]
                portfolio = normalize_ratios([
                    {**item, "ratio": weight_map.get(item.get("ticker", ""), item["ratio"])}
                    for item in portfolio
                ])

    product_by_name: dict[str, dict] = {item["name"]: item for item in portfolio}
    expected_rr, expected_amount = _calc_expected_return(portfolio, plan, product_by_name)

    ga = plan["gathering_account"]
    months = plan["investment_months"]
    amount = plan["amount"]

    return {
        "investment_flows": [{
            "flow_type": plan["flow_type"],
            "title": plan["title"],
            "term": plan["term"],
            "summary": plan["summary"],
            "reasoning": plan.get("reasoning", ""),
            "gathering_id": plan["gathering_asset_id"],
            "gathering_account": ga,
            "amount": amount,
            "portfolio": [
                {
                    "type": product_by_name.get(item["name"], {}).get("product_type", "ETF"),
                    "name": item["name"],
                    "ticker": item["ticker"],
                    "ratio": item["ratio"],
                    "interest_rate": float(product_by_name.get(item["name"], {}).get("interest_rate") or 0.0),
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


# ── LangGraph: flow subgraph + Send fan-out ───────────────────────────────────

def _build_flow_subgraph() -> StateGraph:
    graph = StateGraph(FlowExecuteState, output=_FlowSubgraphOutput)
    graph.add_node("build_portfolio", _node_build_portfolio)
    graph.add_node("execute_flow", execute_flow)
    graph.set_entry_point("build_portfolio")
    graph.add_edge("build_portfolio", "execute_flow")
    graph.add_edge("execute_flow", END)
    return graph.compile()


_flow_subgraph = _build_flow_subgraph()


def _route_flows(state: AssetPortfolioState) -> list[Send]:
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
            "candidates": [],
            "portfolio": [],
            "investment_flows": [],
        })
        for plan in state["flow_plans"]
    ]


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(AssetPortfolioState)
    graph.add_node("plan_flows", _plan_flows)
    graph.add_node("flow_branch", _flow_subgraph)

    graph.set_entry_point("plan_flows")
    graph.add_conditional_edges("plan_flows", _route_flows, ["flow_branch"])
    graph.add_edge("flow_branch", END)

    return graph.compile()


_graph = _build_graph()


# ── Entry point ───────────────────────────────────────────────────────────────

async def recommend_asset_portfolio(request: AssetPortfolioRequest) -> AssetPortfolioResponse:
    asset_list = [
        {
            "asset_id": str(a.asset_id),
            "asset_type": a.asset_type,
            "account_name": a.account_name,
            "balance": a.balance,
        }
        for a in request.invest_assets
    ]

    initial_state: AssetPortfolioState = {
        "invest_amount": request.invest_amount,
        "interest": request.interest,
        "invest_interests": request.invest_interests,
        "porti_type": request.porti_type,
        "porti_comment": request.porti_comment,
        "asset_list": asset_list,
        "flow_plans": [],
        "investment_flows": [],
    }

    final_state: AssetPortfolioState = await _graph.ainvoke(initial_state)

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
