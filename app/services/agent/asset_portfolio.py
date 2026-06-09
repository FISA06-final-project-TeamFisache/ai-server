import asyncio
import logging
import math
import operator
import os
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse
from uuid import UUID

import pandas as pd
import pymysql
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel
from pypfopt.hierarchical_portfolio import HRPOpt

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
from app.services.agent.gather_products import GATHER_PRODUCTS

logger = logging.getLogger(__name__)

_EMBED_MODEL = "text-embedding-3-small"
MYSQL_URL = os.getenv("MYSQL_URL", "")
_WOORI_BANK = "우리은행"
_WOORI_INVEST = "우리투자증권"
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


class _AIPortfolioItem(BaseModel):
    name: str
    ticker: str
    comment: str


class _PortfolioOutput(BaseModel):
    portfolio: list[_AIPortfolioItem]


class _NarratedItem(BaseModel):
    ticker: str
    comment: str


class _NarratorOutput(BaseModel):
    items: list[_NarratedItem]


# ── State ─────────────────────────────────────────────────────────────────────

class AssetPortfolioState(TypedDict):
    invest_amount: int
    interest: str
    invest_interests: list[str]
    porti_type: str
    porti_comment: str
    asset_list: list[dict]
    user_embedding: list[float] | None
    flow_plans: list[dict]
    investment_flows: Annotated[list, operator.add]


class FlowExecuteState(TypedDict):
    plan: dict
    shared: dict


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
            "comment": item.comment,
        })

    # 비중은 균등 배분 — HRP가 이후에 덮어씌움
    n = len(validated)
    if n > 0:
        base = 100 // n
        rem = 100 - base * n
        for i, item in enumerate(validated):
            item["ratio"] = base + (1 if i < rem else 0)

    return validated


def _candidates_text(etf_candidates: list[dict]) -> str:
    return "\n".join(
        f"- [{p['product_type']}] {p['institution']} | {p['name']} "
        f"| ticker:{p.get('ticker') or ''} | 연 {p['interest_rate'] or '-'}% "
        f"| 시가총액:{_fmt_mktcap(p.get('mktcap'))} | 일평균거래대금:{_fmt_mktcap(p.get('avg_trading_value'))} "
        f"| 연변동성:{str(round(float(p['volatility']), 1)) + '%' if p.get('volatility') else '-'} "
        f"| {(p['description'] or '')[:80]}"
        for p in etf_candidates
    ) or "상품 없음"



# ── HRP (Hierarchical Risk Parity) ───────────────────────────────────────────

def _fetch_prices_df(tickers: list[str]) -> pd.DataFrame:
    """MySQL etf_prices에서 선택된 ETF 가격 이력을 pivot DataFrame으로 반환."""
    if not tickers or not MYSQL_URL:
        return pd.DataFrame()
    raw = MYSQL_URL.replace("mysql+pymysql://", "mysql://")
    p = urlparse(raw)
    try:
        conn = pymysql.connect(
            host=p.hostname, port=p.port or 3306,
            user=p.username, password=p.password,
            database=p.path.lstrip("/"), charset="utf8mb4",
        )
    except Exception as e:
        logger.warning("MySQL 연결 실패 (HRP 건너뜀): %s", e)
        return pd.DataFrame()
    try:
        placeholders = ",".join(["%s"] * len(tickers))
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT isu_cd, bas_dt, close_prc FROM etf_prices "
                f"WHERE isu_cd IN ({placeholders}) ORDER BY bas_dt",
                tickers,
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    records = [(str(bas_dt), isu_cd, float(close_prc)) for isu_cd, bas_dt, close_prc in rows]
    df = pd.DataFrame(records, columns=["date", "ticker", "price"])
    prices = df.pivot(index="date", columns="ticker", values="price")
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)
    return prices


def _apply_hrp_weights(
    items: list[dict], prices_df: pd.DataFrame
) -> tuple[list[dict], dict | None]:
    """HRPOpt로 비중 계산. 성공 시 (items, metrics), 실패 시 (items, None) 반환."""
    if len(items) < 2:
        logger.info("HRP: ETF %d개 → 단일 자산, 균등 배분 유지", len(items))
        return items, None
    if prices_df.empty:
        logger.info("HRP: 가격 데이터 없음 (MySQL 조회 실패 등) → 균등 배분 유지")
        return items, None

    tickers = [item.get("ticker", "") for item in items]
    available = [t for t in tickers if t and t in prices_df.columns]

    if len(available) < 2:
        logger.info("HRP: 가격 이력 있는 ETF %d개 미만 → 균등 배분 유지 | 요청 티커: %s", len(available), tickers)
        return items, None

    prices = prices_df[available].dropna()
    if len(prices) < 2:
        logger.info("HRP: 가격 데이터 %d일로 부족 (최소 2일) → 균등 배분 유지", len(prices))
        return items, None

    equal_ratios = {item.get("ticker") or item["name"]: item["ratio"] for item in items}
    logger.info("HRP: 계산 시작 | 데이터 %d일 | 균등비중 %s", len(prices), equal_ratios)

    try:
        returns = prices.pct_change().dropna()
        daily_cov = returns.cov()
        hrp = HRPOpt(returns=returns, cov_matrix=daily_cov)
        weights = hrp.optimize()

        new_items = []
        for item in items:
            w = weights.get(item.get("ticker", ""))
            new_items.append({**item, "ratio": round(w * 100) if w is not None else item["ratio"]})

        result = normalize_ratios(new_items)
        hrp_ratios = {item.get("ticker") or item["name"]: item["ratio"] for item in result}
        logger.info("HRP: 적용 완료 | HRP비중 %s", hrp_ratios)

        corr = returns.corr()
        metrics: dict = {
            "data_days": len(prices),
            "per_ticker": {},
        }
        for t in available:
            daily_var = float(daily_cov.loc[t, t])
            vol_pct = round(math.sqrt(daily_var * 252) * 100, 1)
            others = [c for c in available if c != t]
            avg_corr = round(float(corr.loc[t, others].mean()), 2) if others else 0.0
            metrics["per_ticker"][t] = {
                "vol_pct": vol_pct,
                "avg_corr": avg_corr,
                "equal_ratio": equal_ratios.get(t, 0),
                "hrp_ratio": hrp_ratios.get(t, 0),
            }

        return result, metrics
    except Exception as e:
        logger.warning("HRP 가중치 계산 실패 (균등 배분 유지): %s", e)
        return items, None


# ── Embedding ─────────────────────────────────────────────────────────────────

async def _get_embedding(text: str) -> list[float] | None:
    if not text.strip():
        return None
    try:
        return await OpenAIEmbeddings(model=_EMBED_MODEL).aembed_query(text)
    except Exception as e:
        logger.warning("임베딩 생성 실패: %s", e)
        return None


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
    # asset_by_type은 로컬 계산 (state에 저장 불필요)
    asset_by_type: dict[str, list[dict]] = {}
    for a in state["asset_list"]:
        asset_by_type.setdefault(a["asset_type"], []).append(a)

    # embedding 한 번만 계산 → 모든 흐름이 공유
    query_parts = list(state["invest_interests"]) + ([state["interest"]] if state["interest"] else [])
    user_embedding = await _get_embedding(" ".join(query_parts))

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
            "ratio": ratio,
            "amount": round(invest_amount * ratio / 100),
            "can_invest": f.account_type in _CAN_INVEST_TYPES,
            "gathering_asset_id": user_asset["asset_id"] if user_asset else None,
            "has_user_account": user_asset is not None,
            "gathering_account": gathering_account,
        })

    return {"user_embedding": user_embedding, "flow_plans": flow_plans}


# ── ETF 검색 (흐름별 DB 직접 쿼리) ──────────────────────────────────────────────

async def _search_etfs_db(
    user_embedding: list[float] | None,
    max_volatility: float | None = None,
    min_cagr: float | None = None,
    name_contains: list[str] | None = None,
    limit: int = 15,
) -> list[dict]:
    """pgvector hybrid search: SQL 조건 필터 + 관심사 벡터 유사도 정렬."""
    pool = await get_pool()
    if not pool:
        return []

    conditions = ["deleted_at IS NULL"]
    params: list = []
    idx = 1

    if max_volatility is not None:
        conditions.append(f"volatility <= ${idx}")
        params.append(float(max_volatility))
        idx += 1

    if min_cagr is not None:
        conditions.append(f"interest_rate >= ${idx}")
        params.append(float(min_cagr))
        idx += 1

    if name_contains:
        kw_conds = []
        for kw in name_contains:
            kw_conds.append(f"name LIKE ${idx}")
            params.append(f"%{kw}%")
            idx += 1
        conditions.append(f"({' OR '.join(kw_conds)})")

    where = " AND ".join(conditions)

    if user_embedding:
        vec_str = "[" + ",".join(f"{x:.8f}" for x in user_embedding) + "]"
        query = (
            "SELECT product_type, institution, name, ticker, interest_rate, description, "
            "mktcap, avg_trading_value, volatility "
            f"FROM products WHERE {where} "
            f"ORDER BY embedding <=> ${idx}::vector "
            f"LIMIT ${idx + 1}"
        )
        params.extend([vec_str, limit])
    else:
        query = (
            "SELECT product_type, institution, name, ticker, interest_rate, description, "
            "mktcap, avg_trading_value, volatility "
            f"FROM products WHERE {where} "
            f"ORDER BY interest_rate DESC NULLS LAST "
            f"LIMIT ${idx}"
        )
        params.append(limit)

    try:
        rows = await pool.fetch(query, *params)
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("ETF 검색 실패: %s", e)
        return []


@tool
def _search_etf_schema(
    max_volatility: float | None = None,
    min_cagr: float | None = None,
    name_contains: list[str] | None = None,
    limit: int = 15,
) -> str:
    """DB에서 조건에 맞는 ETF를 검색합니다.
    max_volatility: 최대 연변동성(%). 예) 20.0
    min_cagr: 최소 연평균수익률(%). 예) 5.0
    name_contains: ETF명에 포함돼야 할 키워드 목록. 예) ["채권", "배당"]
    limit: 최대 반환 개수 (기본 15)
    """
    _ = max_volatility, min_cagr, name_contains, limit
    return ""


# ── Flow Agent (per-flow worker) ──────────────────────────────────────────────


async def _select_flow_products(plan: dict, shared: dict) -> tuple[list[dict], list[dict]]:
    """ETF 선택 2-Phase.
    Phase 1: LLM이 search_etfs 기준 결정 → _search_etfs_db로 DB 직접 쿼리
    Phase 2: 필터된 후보에서 structured output으로 최종 선택
    Returns: (portfolio_items, latest_candidates)
    """
    user_embedding = shared.get("user_embedding")
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini-2024-07-18"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    ).bind_tools([_search_etf_schema])

    tool_messages = [
        SystemMessage(content=(
            "포트폴리오 전문가입니다.\n"
            "search_etfs 도구를 사용해 이 흐름에 적합한 ETF를 검색하세요.\n"
            "IRP·PENSION_SAVINGS 계좌인 경우 주식·채권·해외 자산군을 반드시 분산하세요.\n"
            "결과가 충분하면 도구를 더 이상 호출하지 마세요."
        )),
        HumanMessage(content=(
            f"계좌 유형: {plan['account_type']}\n"
            f"흐름: {plan['flow_type']} ({plan['term']}, {plan['investment_months']}개월)\n"
            f"투자 전략: {plan['invest_strategy'] or '흐름 성격에 맞게 분산 구성'}\n"
            f"PorTI: {_porti_label(shared['porti_type'])} / {shared['porti_comment']}\n"
            f"관심사: {shared['interest']}\n"
            f"투자 관심 분야: {', '.join(shared['invest_interests']) or '없음'}"
        )),
    ]

    # 초기 fallback: 필터 없이 관심사 기반 검색
    latest_candidates = await _search_etfs_db(user_embedding, limit=15)

    for _ in range(3):
        response = await llm.ainvoke(tool_messages)
        tool_messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            if tc["name"] == _search_etf_schema.name:
                args = tc["args"]
                new_candidates = await _search_etfs_db(
                    user_embedding=user_embedding,
                    max_volatility=args.get("max_volatility"),
                    min_cagr=args.get("min_cagr"),
                    name_contains=args.get("name_contains"),
                    limit=int(args.get("limit", 15)),
                )
                if new_candidates:
                    latest_candidates = new_candidates
                tool_result = _candidates_text(latest_candidates) if latest_candidates else "조건을 충족하는 ETF 없음"
                tool_messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))

    confirmed_by_name = {p["name"]: p for p in latest_candidates}
    filtered_text = _candidates_text(latest_candidates)

    # Phase 2: 필터된 후보에서 최종 선택 (clean structured output call)
    selection_messages = [
        SystemMessage(content=(
            "포트폴리오 전문가입니다.\n"
            "[선택 가능 상품 목록]에서 이 흐름에 적합한 ETF를 선택하세요.\n\n"
            "규칙:\n"
            "- name: 목록의 정확한 상품명 그대로 (변형·새 이름 생성 절대 금지)\n"
            "- ticker: 목록의 ticker 그대로\n"
            "- IRP·PENSION_SAVINGS 계좌인 경우 주식·채권·해외 자산군을 반드시 분산하세요.\n"
            "- comment: 이 ETF의 수치(변동성·CAGR)와 흐름 전략이 맞는 이유 1문장. 수익 보장 표현 금지.\n"
            'JSON만 응답: {"portfolio":[{"name":"정확한상품명","ticker":"코드","comment":"이유"}]}'
        )),
        HumanMessage(content=(
            f"계좌 유형: {plan['account_type']}\n"
            f"흐름: {plan['flow_type']} ({plan['term']}, {plan['investment_months']}개월)\n"
            f"투자 전략: {plan['invest_strategy'] or '흐름 성격에 맞게 분산 구성'}\n"
            f"PorTI: {_porti_label(shared['porti_type'])} / {shared['porti_comment']}\n"
            f"관심사: {shared['interest']}\n"
            f"투자 관심 분야: {', '.join(shared['invest_interests']) or '없음'}\n\n"
            f"[선택 가능 상품 목록]\n{filtered_text}"
        )),
    ]

    result = await ainvoke_structured(selection_messages, _PortfolioOutput)
    raw_items = result.portfolio if result else []
    portfolio = _validate_and_merge(raw_items, confirmed_by_name)
    return portfolio, latest_candidates


async def _narrate_hrp_result(
    portfolio: list[dict],
    hrp_metrics: dict,
    plan: dict,
) -> list[dict]:
    """HRPOpt 수치(변동성·상관계수·비중 변화)를 LLM이 자연어로 번역."""
    per_ticker = hrp_metrics["per_ticker"]
    data_days = hrp_metrics["data_days"]

    rows = []
    for item in portfolio:
        t = item.get("ticker", "")
        m = per_ticker.get(t, {})
        rows.append(
            f"{t} | {item['name']} | {item['comment']} | "
            f"연변동성 {m.get('vol_pct', '-')}% | "
            f"평균상관 {m.get('avg_corr', '-')} | "
            f"균등{m.get('equal_ratio', '-')}%→HRP{m.get('hrp_ratio', '-')}%"
        )

    messages = [
        SystemMessage(content=(
            "투자 포트폴리오 데이터 해설 전문가입니다.\n"
            "HRP(Hierarchical Risk Parity) 알고리즘이 산출한 수치를 바탕으로\n"
            "각 ETF의 비중 배정 근거를 1~2문장으로 설명하세요.\n\n"
            "규칙:\n"
            "- 아래 수치 데이터만 근거로 사용할 것\n"
            "- 비중 변경 제안·수익 보장 표현 금지\n"
            "- ticker 기준으로 응답\n"
            'JSON만 응답: {"items":[{"ticker":"코드","comment":"설명"}]}'
        )),
        HumanMessage(content=(
            f"흐름: {plan['flow_type']} ({plan['term']}, {plan['investment_months']}개월)\n"
            f"HRP 계산 기간: {data_days}거래일\n\n"
            "ticker | ETF명 | 선택이유 | 연변동성 | 평균상관계수 | 균등→HRP비중\n"
            + "\n".join(rows)
        )),
    ]

    result = await ainvoke_structured(messages, _NarratorOutput)
    if not result or not result.items:
        logger.info("Narrator: 결과 없음, 기존 comment 유지")
        return portfolio

    comment_by_ticker = {ni.ticker: ni.comment for ni in result.items}
    logger.info("Narrator: [%s] comment 업데이트 | 티커: %s", plan["flow_type"], list(comment_by_ticker.keys()))
    return [
        {**item, "comment": comment_by_ticker.get(item.get("ticker", ""), item["comment"])}
        for item in portfolio
    ]


def _calc_expected_return(
    portfolio: list[dict],
    plan: dict,
    product_by_name: dict[str, dict],
) -> tuple[float, int]:
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
    return expected_rr, round(expected_amount)


async def _run_flow_agent(plan: dict, shared: dict) -> dict:
    _ga_name = plan["gathering_account"]["name"]
    account_comment = (
        f"보유 중인 {_ga_name} 계좌를 {plan['account_type']} 흐름에 활용합니다."
        if plan["has_user_account"]
        else f"이 흐름을 위해 {_ga_name} ({plan['account_type']}) 계좌 신규 개설이 필요합니다."
    )

    portfolio: list[dict] = []
    etf_candidates: list[dict] = []
    if plan["can_invest"]:
        portfolio, etf_candidates = await _select_flow_products(plan, shared)

        if len(portfolio) >= 2:
            tickers = [item["ticker"] for item in portfolio if item.get("ticker")]
            if tickers:
                logger.info("HRP: [%s] 가격 이력 조회 시작 | 티커: %s", plan["flow_type"], tickers)
                loop = asyncio.get_running_loop()
                prices_df = await loop.run_in_executor(None, _fetch_prices_df, tickers)
                portfolio, hrp_metrics = _apply_hrp_weights(portfolio, prices_df)

                if hrp_metrics:
                    portfolio = await _narrate_hrp_result(portfolio, hrp_metrics, plan)

    product_by_name: dict[str, dict] = {p["name"]: p for p in etf_candidates}

    expected_rr, expected_amount = _calc_expected_return(portfolio, plan, product_by_name)

    ga = plan["gathering_account"]
    months = plan["investment_months"]
    amount = plan["amount"]

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


# ── LangGraph: Send fan-out ───────────────────────────────────────────────────

def _route_flows(state: AssetPortfolioState) -> list[Send]:
    shared = {
        "porti_type": state["porti_type"],
        "porti_comment": state["porti_comment"],
        "interest": state["interest"],
        "invest_interests": state["invest_interests"],
        "user_embedding": state["user_embedding"],
    }
    return [
        Send("execute_flow", {"plan": plan, "shared": shared})
        for plan in state["flow_plans"]
    ]


async def execute_flow(flow_state: FlowExecuteState) -> dict:
    result = await _run_flow_agent(flow_state["plan"], flow_state["shared"])
    return {"investment_flows": [result]}


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(AssetPortfolioState)
    graph.add_node("plan_flows", _plan_flows)
    graph.add_node("execute_flow", execute_flow)

    graph.set_entry_point("plan_flows")
    graph.add_conditional_edges("plan_flows", _route_flows, ["execute_flow"])
    graph.add_edge("execute_flow", END)

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
        "user_embedding": None,
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
