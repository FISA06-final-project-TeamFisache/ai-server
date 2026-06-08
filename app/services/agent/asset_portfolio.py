import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Literal, TypedDict
from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.schemas.portfolio import (
    AssetPortfolioRequest,
    AssetPortfolioResponse,
    GatheringAccount,
    InvestmentPlan,
    PortfolioItem,
)
from app.services.agent.llm import ainvoke_structured
from app.services.agent.tools import normalize_ratios
from app.services.rag.db import get_pool
from app.services.agent.porti_types import porti_label as _porti_label

logger = logging.getLogger(__name__)

_EMBED_MODEL = "text-embedding-3-small"
_INVEST_PRODUCT_TYPES = ["STOCK", "ETF", "BOND"]
_GATHER_PRODUCT_TYPES = ["CHECKING", "PARKING", "CMA", "SAVINGS", "DEPOSIT", "ISA", "IRP", "PENSION_SAVINGS"]
_WOORI_BANK = "우리은행"
_WOORI_INVEST = "우리투자증권"
_MAX_REFLECTION_ROUNDS = 2
_CAN_INVEST_TYPES = {"ISA", "IRP", "PENSION_SAVINGS"}

_AccountType = Literal[
    "CHECKING", "PARKING", "CMA", "SAVING", "DEPOSIT",
    "ISA", "IRP", "PENSION_SAVINGS",
]


def _fmt_mktcap(v) -> str:
    if not v:
        return "-"
    v = int(v)
    if v >= 1_000_000_000_000:
        return f"{v / 1_000_000_000_000:.1f}조"
    if v >= 100_000_000:
        return f"{v // 100_000_000}억"
    return f"{v:,}"


# ── AI 출력 스키마 ─────────────────────────────────────────────────────────────

class _FlowPlan(BaseModel):
    flow_type: str
    term: Literal["단기", "중기", "장기"]
    investment_months: int
    account_type: _AccountType
    invest_strategy: str
    title: str
    summary: str
    ratio: int


class _FlowPlansOutput(BaseModel):
    flows: list[_FlowPlan]


class _AccountCommentOutput(BaseModel):
    comment: str


class _AIPortfolioItem(BaseModel):
    name: str
    ticker: str
    ratio: int
    comment: str


class _PortfolioOutput(BaseModel):
    portfolio: list[_AIPortfolioItem]


class _EvalOutput(BaseModel):
    is_aligned: bool
    feedback: str = ""


# ── State ─────────────────────────────────────────────────────────────────────

class AssetPortfolioState(TypedDict):
    invest_amount: int
    interest: str
    invest_interests: list[str]
    porti_type: str
    porti_comment: str
    asset_list: list[dict]
    asset_by_type: dict[str, list[dict]]
    etf_candidates: list[dict]
    gather_products: list[dict]
    flow_plans: list[dict]
    investment_flows: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _can_invest(account_type: str) -> bool:
    return account_type in _CAN_INVEST_TYPES


def _find_user_asset(asset_by_type: dict, account_type: str, used_ids: set) -> dict | None:
    for a in asset_by_type.get(account_type, []):
        if a["asset_id"] not in used_ids:
            return a
    return None


def _find_best_product(gather_products: list[dict], account_type: str, prefer_institution: str) -> dict | None:
    candidates = [p for p in gather_products if p["product_type"] == account_type]
    if not candidates:
        return None
    woori = [p for p in candidates if prefer_institution in (p.get("institution") or "")]
    return woori[0] if woori else candidates[0]


def _validate_and_merge(
    raw_items: list[_AIPortfolioItem],
    confirmed_by_name: dict[str, dict],
    etf_candidates: list[dict],
) -> list[dict]:
    validated: list[dict] = []
    for item in raw_items:
        name = item.name
        if name not in confirmed_by_name:
            matched = next((n for n in confirmed_by_name if name in n or n in name), None)
            if not matched:
                logger.warning("상품명 목록 미존재, 제거: %s", name)
                continue
            name = matched
        p = confirmed_by_name[name]
        validated.append({
            "name": name,
            "ticker": p.get("ticker") or item.ticker or "",
            "ratio": item.ratio,
            "comment": item.comment,
        })

    if not validated and etf_candidates:
        top = etf_candidates[:3]
        base = 100 // len(top)
        rem = 100 - base * len(top)
        validated = [
            {
                "name": p["name"],
                "ticker": p.get("ticker") or "",
                "ratio": base + (rem if i == 0 else 0),
                "comment": "분산 투자를 위한 기본 추천 상품입니다.",
            }
            for i, p in enumerate(top)
        ]

    return normalize_ratios(validated)


def _candidates_text(etf_candidates: list[dict]) -> str:
    return "\n".join(
        f"- [{p['product_type']}] {p['institution']} | {p['name']} "
        f"| ticker:{p.get('ticker') or ''} | 연 {p['interest_rate'] or '-'}% "
        f"| 시가총액:{_fmt_mktcap(p.get('mktcap'))} | 일평균거래대금:{_fmt_mktcap(p.get('avg_trading_value'))} "
        f"| {(p['description'] or '')[:80]}"
        for p in etf_candidates
    ) or "상품 없음"


# ── Embedding ─────────────────────────────────────────────────────────────────

async def _get_embedding(text: str) -> list[float] | None:
    if not text.strip():
        return None
    try:
        return await OpenAIEmbeddings(model=_EMBED_MODEL).aembed_query(text)
    except Exception as e:
        logger.warning("임베딩 생성 실패: %s", e)
        return None


# ── Node: preprocess ──────────────────────────────────────────────────────────

async def _preprocess(state: AssetPortfolioState) -> AssetPortfolioState:
    asset_by_type: dict[str, list[dict]] = {}
    for a in state["asset_list"]:
        asset_by_type.setdefault(a["asset_type"], []).append(a)

    query_parts = list(state["invest_interests"]) + ([state["interest"]] if state["interest"] else [])
    query_vector = await _get_embedding(" ".join(query_parts))

    pool = await get_pool()
    etf_candidates: list[dict] = []
    gather_products: list[dict] = []

    if pool:
        try:
            if query_vector:
                vec_str = "[" + ",".join(f"{x:.8f}" for x in query_vector) + "]"
                rows = await pool.fetch(
                    "SELECT product_type, institution, name, ticker, interest_rate, description, "
                    "mktcap, avg_trading_value "
                    "FROM products "
                    "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                    "ORDER BY embedding <=> $2::vector "
                    "LIMIT 30",
                    _INVEST_PRODUCT_TYPES, vec_str,
                )
            else:
                rows = await pool.fetch(
                    "SELECT product_type, institution, name, ticker, interest_rate, description, "
                    "mktcap, avg_trading_value "
                    "FROM products "
                    "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                    "ORDER BY interest_rate DESC NULLS LAST "
                    "LIMIT 30",
                    _INVEST_PRODUCT_TYPES,
                )
            etf_candidates = [dict(r) for r in rows]
        except Exception as e:
            logger.warning("투자 상품 조회 실패: %s", e)

        try:
            rows = await pool.fetch(
                "SELECT product_type, institution, name, ticker, interest_rate, description "
                "FROM products "
                "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                "ORDER BY product_type, interest_rate DESC NULLS LAST",
                _GATHER_PRODUCT_TYPES,
            )
            gather_products = [dict(r) for r in rows]
        except Exception as e:
            logger.warning("모으기 상품 조회 실패: %s", e)

    return {
        **state,
        "asset_by_type": asset_by_type,
        "etf_candidates": etf_candidates,
        "gather_products": gather_products,
    }


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


async def _plan_flows(state: AssetPortfolioState) -> AssetPortfolioState:
    messages = [
        SystemMessage(content=(
            "당신은 개인 자산관리 전문가입니다.\n"
            "사용자 투자 성향과 관심사를 반영해 3~5개의 투자 흐름을 설계하세요.\n\n"
            "계좌 종류:\n"
            "- 적립 전용 (투자 상품 불가): CHECKING, PARKING, CMA, SAVING, DEPOSIT\n"
            "- 투자 상품 가능: ISA, IRP, PENSION_SAVINGS\n\n"
            "설계 기준:\n"
            "- 단기 (investment_months 3~18): DEPOSIT 또는 CMA 권장\n"
            "- 중기 (investment_months 24~84): 안정 성향 → SAVING, 투자 성향 → ISA\n"
            "- 장기 (investment_months 120~360): IRP 또는 PENSION_SAVINGS 권장\n"
            "- ratio 합계 반드시 100\n"
            "- invest_strategy: 투자 불가 계좌는 빈 문자열,\n"
            "  투자 가능 계좌는 ETF 선택 시 중점 특성 (예: '채권 추종 ETF로 안정성 확보')\n\n"
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

    # ratio 정규화
    raw_ratios = [max(1, f.ratio) for f in raw_flows]
    total = sum(raw_ratios)
    if total != 100:
        normalized = [round(r / total * 100) for r in raw_ratios]
        normalized[0] += 100 - sum(normalized)
    else:
        normalized = raw_ratios

    # 계좌 배정 (deterministic — 병렬 전에 완료해야 used_ids 추적 가능)
    asset_by_type = state["asset_by_type"]
    gather_products = state["gather_products"]
    invest_amount = state["invest_amount"]
    used_ids: set = set()

    flow_plans = []
    for f, ratio in zip(raw_flows, normalized):
        fallback_institution = _WOORI_INVEST if f.account_type in _CAN_INVEST_TYPES else _WOORI_BANK
        user_asset = _find_user_asset(asset_by_type, f.account_type, used_ids)
        if user_asset:
            used_ids.add(user_asset["asset_id"])
        best_product = _find_best_product(gather_products, f.account_type, fallback_institution)

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
            "ratio": ratio,
            "amount": round(invest_amount * ratio / 100),
            "can_invest": _can_invest(f.account_type),
            "gathering_asset_id": user_asset["asset_id"] if user_asset else None,
            "has_user_account": user_asset is not None,
            "gathering_account": gathering_account,
        })

    return {**state, "flow_plans": flow_plans}


# ── Flow Agent (per-flow worker) ──────────────────────────────────────────────

async def _get_account_comment(plan: dict, shared: dict) -> str:
    has_account = plan["has_user_account"]
    account_desc = (
        f'{plan["account_type"]} | {plan["gathering_account"]["name"]} '
        f'({plan["gathering_account"]["institution"]}) | '
        f'기존 계좌 {"있음" if has_account else "없음 → 신규 개설 필요"}'
    )
    messages = [
        SystemMessage(content=(
            "개인 자산관리 전문가입니다. 투자 흐름의 모으기 계좌 선택 이유를 1~2문장으로 설명하세요.\n"
            "기존 계좌 없음이면 왜 개설해야 하는지, 있으면 왜 이 흐름에 활용하는지.\n"
            "수익보장·단정 표현 금지. 50자 이내.\n"
            'JSON만 응답: {"comment":"..."}'
        )),
        HumanMessage(content=(
            f"흐름: {plan['flow_type']} ({plan['term']}, {plan['investment_months']}개월)\n"
            f"PorTI: {_porti_label(shared['porti_type'])} / {shared['porti_comment']}\n"
            f"계좌: {account_desc}"
        )),
    ]
    result = await ainvoke_structured(messages, _AccountCommentOutput)
    return result.comment if result else ""


async def _select_flow_products(plan: dict, shared: dict) -> list[dict]:
    confirmed_by_name = {p["name"]: p for p in shared["etf_candidates"]}
    messages = [
        SystemMessage(content=(
            "포트폴리오 전문가입니다.\n"
            "[선택 가능 상품 목록]에서만 골라 포트폴리오를 구성하세요.\n\n"
            "규칙:\n"
            "- name: 목록의 정확한 상품명 그대로 (변형·새 이름 생성 절대 금지)\n"
            "- ticker: 목록의 ticker 그대로\n"
            "- ratio 합계 = 100\n"
            f"- 투자 전략 힌트: {plan['invest_strategy'] or '흐름 성격에 맞게 분산 구성'}\n"
            "- comment: ETF 특성(지수 추종·채권 추종·배당·섹터 등)이 이 흐름 전략과\n"
            "  어떻게 맞는지 1문장. 수익보장 표현 금지.\n"
            f"- 계좌 {plan['account_type']}에 적합하게 구성\n"
            "  ISA: 국내 ETF 우선 / IRP·연금저축: 장기 분산\n\n"
            f"[선택 가능 상품 목록]\n{_candidates_text(shared['etf_candidates'])}\n\n"
            'JSON만 응답: {"portfolio":[{"name":"정확한상품명","ticker":"코드","ratio":60,"comment":"이유"}]}'
        )),
        HumanMessage(content=(
            f"흐름: {plan['flow_type']} ({plan['term']}, {plan['investment_months']}개월)\n"
            f"PorTI: {_porti_label(shared['porti_type'])} / {shared['porti_comment']}\n"
            f"관심사: {shared['interest']}\n"
            f"투자 관심 분야: {', '.join(shared['invest_interests']) or '없음'}"
        )),
    ]
    result = await ainvoke_structured(messages, _PortfolioOutput)
    raw_items = result.portfolio if result else []
    return _validate_and_merge(raw_items, confirmed_by_name, shared["etf_candidates"])


async def _eval_flow_products(plan: dict, portfolio: list[dict], shared: dict) -> str | None:
    portfolio_desc = ", ".join(f'{p["name"]}({p["ratio"]}%)' for p in portfolio)
    messages = [
        SystemMessage(content=(
            "포트폴리오 검토 전문가입니다.\n"
            "투자 성향과 포트폴리오가 적절한지 평가하세요.\n\n"
            "기준:\n"
            "- 안전형(SWIMMING·ARCHERY): 채권·배당 ETF 위주, 주식 30% 이하\n"
            "- 중립형(JUDO·RHYTHMIC): 주식·채권 균형, 주식 30~60%\n"
            "- 투자형(FENCING·CYCLING): 주식형 ETF 위주, 주식 60% 이상\n"
            "- IRP·연금저축: 장기 분산 구성 여부\n\n"
            'JSON만 응답: {"is_aligned":true,"feedback":""}'
        )),
        HumanMessage(content=(
            f"흐름: {plan['flow_type']} (계좌:{plan['account_type']})\n"
            f"PorTI: {_porti_label(shared['porti_type'])} / {shared['porti_comment']}\n"
            f"포트폴리오: {portfolio_desc}"
        )),
    ]
    result = await ainvoke_structured(messages, _EvalOutput)
    if not result or result.is_aligned:
        return None
    return result.feedback or "성향에 맞게 포트폴리오를 재구성하세요."


async def _refine_flow_products(plan: dict, portfolio: list[dict], feedback: str, shared: dict) -> list[dict]:
    confirmed_by_name = {p["name"]: p for p in shared["etf_candidates"]}
    messages = [
        SystemMessage(content=(
            "포트폴리오 전문가입니다.\n"
            "[선택 가능 상품 목록]에서만 골라 포트폴리오를 개선하세요.\n"
            "name은 목록의 정확한 상품명 그대로. ratio 합계=100.\n\n"
            f"[개선 피드백]\n{feedback}\n\n"
            f"[선택 가능 상품 목록]\n{_candidates_text(shared['etf_candidates'])}\n\n"
            'JSON만 응답: {"portfolio":[{"name":"정확한상품명","ticker":"코드","ratio":60,"comment":"이유"}]}'
        )),
        HumanMessage(content=(
            f"흐름: {plan['flow_type']} (계좌:{plan['account_type']})\n"
            f"PorTI: {_porti_label(shared['porti_type'])} / {shared['porti_comment']}"
        )),
    ]
    result = await ainvoke_structured(messages, _PortfolioOutput)
    raw_items = result.portfolio if result else []
    refined = _validate_and_merge(raw_items, confirmed_by_name, shared["etf_candidates"])
    return refined if refined else portfolio


async def _run_flow_agent(plan: dict, shared: dict) -> dict:
    account_comment = await _get_account_comment(plan, shared)

    portfolio: list[dict] = []
    if plan["can_invest"]:
        portfolio = await _select_flow_products(plan, shared)
        for _ in range(_MAX_REFLECTION_ROUNDS):
            feedback = await _eval_flow_products(plan, portfolio, shared)
            if not feedback:
                break
            logger.info("refine %s: %s", plan["flow_type"], feedback[:60])
            portfolio = await _refine_flow_products(plan, portfolio, feedback, shared)

    # 수익률 계산 (deterministic)
    product_by_name: dict[str, dict] = {}
    for p in shared["etf_candidates"] + shared["gather_products"]:
        product_by_name.setdefault(p["name"], p)

    ga = plan["gathering_account"]
    ga_rate_raw = float(ga.get("interest_rate", 0.0) or 0.0)
    ga_rate = ga_rate_raw if ga_rate_raw > 0.0 else 2.5

    if portfolio:
        weighted = sum(
            (float(product_by_name.get(item["name"], {}).get("interest_rate") or 0.0) or 2.5)
            * item["ratio"] / 100
            for item in portfolio
        )
        expected_rr = weighted if weighted > 0 else 2.5
    else:
        expected_rr = ga_rate

    months = plan["investment_months"]
    amount = plan["amount"]
    r_m = expected_rr / 100 / 12
    expected_amount = (
        amount * ((math.pow(1 + r_m, months) - 1) / r_m)
        if r_m > 0 else float(amount * months)
    )

    return {
        "flow_type": plan["flow_type"],
        "title": plan["title"],
        "term": plan["term"],
        "summary": plan["summary"],
        "gathering_id": plan["gathering_asset_id"],
        "gathering_account": ga,
        "amount": amount,
        "account_comment": account_comment,
        "portfolio": [
            {
                "type": product_by_name.get(item["name"], {}).get("product_type", "ETF"),
                "name": item["name"],
                "ticker": item["ticker"],
                "ratio": item["ratio"],
                "interest_rate": float(product_by_name.get(item["name"], {}).get("interest_rate") or 0.0),
                "comment": item["comment"],
            }
            for item in portfolio
        ],
        "expected_rr_pct": round(expected_rr, 1),
        "investment_months": months,
        "expected_amount": round(expected_amount),
        "rr_comment": f"연 {expected_rr:.1f}% 기준 {months}개월 적립식 복리 시 약 {round(expected_amount):,}원 예상.",
    }


# ── Node: execute_flows ───────────────────────────────────────────────────────

async def _execute_flows(state: AssetPortfolioState) -> AssetPortfolioState:
    shared = {
        "porti_type": state["porti_type"],
        "porti_comment": state["porti_comment"],
        "interest": state["interest"],
        "invest_interests": state["invest_interests"],
        "etf_candidates": state["etf_candidates"],
        "gather_products": state["gather_products"],
    }
    investment_flows = await asyncio.gather(*[
        _run_flow_agent(plan, shared)
        for plan in state["flow_plans"]
    ])
    return {**state, "investment_flows": list(investment_flows)}


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(AssetPortfolioState)
    graph.add_node("preprocess", _preprocess)
    graph.add_node("plan_flows", _plan_flows)
    graph.add_node("execute_flows", _execute_flows)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "plan_flows")
    graph.add_edge("plan_flows", "execute_flows")
    graph.add_edge("execute_flows", END)

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
        "asset_by_type": {},
        "etf_candidates": [],
        "gather_products": [],
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
                account_comment=f["account_comment"],
                portfolio=[
                    PortfolioItem(
                        type=p["type"],
                        name=p["name"],
                        ticker=p["ticker"],
                        ratio=p["ratio"],
                        interest_rate=p["interest_rate"],
                        comment=p["comment"],
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
