from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pandas as pd
import pymysql
import yfinance as yf
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pypfopt.hierarchical_portfolio import HRPOpt

from app.services.rag.db import get_pool

logger = logging.getLogger(__name__)

# 관심 종목: (표시 이름, Yahoo Finance 티커)
_WATCHLIST: list[tuple[str, str]] = [
    ("삼성전자", "005930.KS"),
    ("SK하이닉스", "000660.KS"),
    ("현대차", "005380.KS"),
    ("NAVER", "035420.KS"),
    ("카카오", "035720.KS"),
    ("TIGER 미국S&P500", "360750.KS"),
    ("KODEX 200", "069500.KS"),
]

_cache: dict[str, tuple[int, float]] = {}  # ticker -> (price_krw, fetched_at)
_CACHE_TTL = 300  # 5분 캐시
_executor = ThreadPoolExecutor(max_workers=4)


def _fetch_price_sync(ticker: str) -> int:
    return int(yf.Ticker(ticker).fast_info["last_price"])


async def get_all_prices() -> list[tuple[str, str, int]]:
    """워치리스트 종목의 현재가(원)를 반환. 캐시 유효 시 캐시 사용."""
    now = time.time()
    loop = asyncio.get_event_loop()

    stale_tickers: list[tuple[str, str]] = []
    tasks: list[asyncio.Future[int]] = []

    for name, ticker in _WATCHLIST:
        if ticker in _cache and now - _cache[ticker][1] < _CACHE_TTL:
            continue
        stale_tickers.append((name, ticker))
        tasks.append(loop.run_in_executor(_executor, _fetch_price_sync, ticker))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (_, ticker), result in zip(stale_tickers, results):
            if isinstance(result, int):
                _cache[ticker] = (result, now)
            else:
                logger.warning("yfinance 조회 실패 (%s): %s", ticker, result)

    return [
        (name, ticker, _cache[ticker][0])
        for name, ticker in _WATCHLIST
        if ticker in _cache
    ]


def pick_stock(
    prices: list[tuple[str, str, int]],
    saved_amount: int,
) -> tuple[str, str, int, float] | None:
    """절약 금액으로 살 수 있는 가장 적합한 종목 선택.

    Returns (name, ticker, price, shares) 또는 None.
    """
    candidates = [
        (name, ticker, price, saved_amount / price)
        for name, ticker, price in prices
        if price > 0
    ]

    in_range = [(n, t, p, s) for n, t, p, s in candidates if 0.05 <= s <= 10]
    pool = in_range if in_range else candidates
    if not pool:
        return None

    etf_keywords = ("TIGER", "KODEX", "KBSTAR", "ARIRANG")
    individuals = [(n, t, p, s) for n, t, p, s in pool if not any(kw in n for kw in etf_keywords)]
    final_pool = individuals if individuals else pool

    return min(final_pool, key=lambda x: abs(x[3] - 1.0))


@tool
async def get_stock_prices() -> list[dict]:
    """관심 종목의 현재 주가를 조회합니다. 챌린지 추천 종목(ticker) 선택 시 반드시 호출하세요."""
    prices = await get_all_prices()
    return [{"name": n, "ticker": t, "price_krw": p} for n, t, p in prices]


def normalize_ratios(items: list[dict], key: str = "ratio") -> list[dict]:
    """비율 합계가 100이 되도록 정규화. 반올림 오차는 첫 항목에 흡수."""
    if not items:
        return items
    total = sum(v[key] for v in items)
    if total == 0 or total == 100:
        return items
    result = [{**v, key: round(v[key] * 100 / total)} for v in items]
    diff = 100 - sum(v[key] for v in result)
    if diff:
        result[0] = {**result[0], key: result[0][key] + diff}
    return result


def normalize_amounts(items: list[dict], key: str, target: int) -> list[dict]:
    """금액 합계가 target이 되도록 비례 조정. 반올림 오차는 마지막 항목에 흡수."""
    if not items or target <= 0:
        return items
    total_weight = sum(v[key] for v in items) or 1
    scale = target / total_weight
    result = [{**v, key: max(0, round(v[key] * scale))} for v in items]
    diff = target - sum(v[key] for v in result)
    if diff != 0:
        result[-1] = {**result[-1], key: max(0, result[-1][key] + diff)}
    return result

def normalize_to_thousands(items: list[dict], key: str, total: int) -> list[dict]:
    """비례 배분 후 천원 단위 반올림. 합계는 total의 천원 내림값에 맞춤."""
    if not items or total <= 0:
        return items
    total_1000 = (total // 1000) * 1000
    if total_1000 == 0:
        return [{**v, key: 0} for v in items]
    total_weight = sum(v[key] for v in items) or 1
    scale = total_1000 / total_weight
    result = [{**v, key: round(v[key] * scale / 1000) * 1000} for v in items]
    diff = total_1000 - sum(v[key] for v in result)
    if diff != 0:
        max_idx = max(range(len(result)), key=lambda i: result[i][key])
        result[max_idx] = {**result[max_idx], key: max(0, result[max_idx][key] + diff)}
    return result

# ── ETF price fetch (sync, run via executor) ─────────────────────────────────

def _fetch_etf_prices(tickers: list[str]) -> pd.DataFrame:
    """MySQL etf_prices에서 ETF 가격 이력을 pivot DataFrame으로 반환."""
    mysql_url = os.getenv("MYSQL_URL", "")
    if not tickers or not mysql_url:
        return pd.DataFrame()
    raw = mysql_url.replace("mysql+pymysql://", "mysql://")
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


# ── @tool definitions ─────────────────────────────────────────────────────────

@tool
def compound_interest(monthly_amount: int, annual_rate_pct: float, months: int) -> dict:
    """복리 적립식 미래가치를 계산합니다.
    monthly_amount: 월 적립금(원)
    annual_rate_pct: 연이율(%). 예) 5.0
    months: 적립 기간(개월)
    Returns: {"expected_amount": int, "total_principal": int}
    """
    r_m = annual_rate_pct / 100 / 12  # 월이율
    if r_m > 0:
        fv = monthly_amount * ((math.pow(1 + r_m, months) - 1) / r_m)
    else:
        fv = float(monthly_amount * months)
    return {
        "expected_amount": round(fv),
        "total_principal": monthly_amount * months,
    }


@tool
async def calculate_hrp_weights(tickers: list[str]) -> dict:
    """HRP(Hierarchical Risk Parity)로 ETF 포트폴리오 최적 비중을 계산합니다.
    tickers: ETF 티커 코드 목록. 예) ["360750", "069500", "148070"]
    Returns: {
        "weights": {ticker: ratio_pct},
        "method": "hrp" | "equal",
        "data_days": int,
        "metrics": {ticker: {"vol_pct": float, "avg_corr": float}}
    }
    """
    def _equal() -> dict:
        n = len(tickers)
        if n == 0:
            return {"weights": {}, "method": "equal", "data_days": 0, "metrics": {}}
        base = 100 // n
        rem = 100 - base * n
        return {
            "weights": {t: base + (1 if i < rem else 0) for i, t in enumerate(tickers)},
            "method": "equal",
            "data_days": 0,
            "metrics": {},
        }

    if len(tickers) < 2:
        return _equal()

    loop = asyncio.get_running_loop()
    prices_df: pd.DataFrame = await loop.run_in_executor(None, _fetch_etf_prices, tickers)

    available = [t for t in tickers if t in prices_df.columns]
    if len(available) < 2:
        logger.info("HRP: 가격 이력 있는 ETF %d개 미만 → 균등 배분 | 요청: %s", len(available), tickers)
        return _equal()

    prices = prices_df[available].dropna()
    if len(prices) < 2:
        logger.info("HRP: 데이터 %d일로 부족 → 균등 배분", len(prices))
        return _equal()

    try:
        returns = prices.pct_change().dropna()
        daily_cov = returns.cov()
        hrp = HRPOpt(returns=returns, cov_matrix=daily_cov)
        raw_weights = hrp.optimize()

        weight_ints = {t: round(w * 100) for t, w in raw_weights.items()}
        diff = 100 - sum(weight_ints.values())
        if diff and weight_ints:
            weight_ints[next(iter(weight_ints))] += diff

        corr = returns.corr()
        metrics: dict[str, dict] = {}
        for t in available:
            daily_var = float(daily_cov.loc[t, t])
            vol_pct = round(math.sqrt(daily_var * 252) * 100, 1)
            others = [c for c in available if c != t]
            avg_corr = round(float(corr.loc[t, others].mean()), 2) if others else 0.0
            metrics[t] = {"vol_pct": vol_pct, "avg_corr": avg_corr}

        logger.info("HRP 완료 | %d일 | %s", len(prices), weight_ints)
        return {"weights": weight_ints, "method": "hrp", "data_days": len(prices), "metrics": metrics}

    except Exception as e:
        logger.warning("HRP 계산 실패 → 균등 배분: %s", e)
        return _equal()


# ── ETF 벡터 검색 ─────────────────────────────────────────────────────────────

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


async def _get_embedding(text: str) -> list[float] | None:
    if not text.strip():
        return None
    try:
        return await _embeddings.aembed_query(text)
    except Exception as e:
        logger.warning("임베딩 생성 실패: %s", e)
        return None


async def _search_etfs_db(
    query_parts: list[str],
    min_trading_value: int = 100_000_000,
    limit: int = 15,
) -> list[dict]:
    """pgvector hybrid search: 거래대금 필터 + 관심사 벡터 유사도 정렬."""
    pool = await get_pool()
    if not pool:
        return []

    user_embedding = await _get_embedding(" ".join(query_parts)) if query_parts else None

    conditions = ["deleted_at IS NULL"]
    params: list = []
    idx = 1

    if min_trading_value > 0:
        conditions.append(f"avg_trading_value >= ${idx}")
        params.append(min_trading_value)
        idx += 1

    where = " AND ".join(conditions)
    select_cols = (
        "product_type, institution, name, ticker, interest_rate, description, "
        "avg_trading_value, acc_trdvol, idx_ind_nm, close_prc, nav, volatility"
    )

    if user_embedding:
        vec_str = "[" + ",".join(f"{x:.8f}" for x in user_embedding) + "]"
        query = (
            f"SELECT {select_cols} FROM products WHERE {where} "
            f"ORDER BY embedding <=> ${idx}::vector "
            f"LIMIT ${idx + 1}"
        )
        params.extend([vec_str, limit])
    else:
        query = (
            f"SELECT {select_cols} FROM products WHERE {where} "
            f"ORDER BY interest_rate DESC NULLS LAST "
            f"LIMIT ${idx}"
        )
        params.append(limit)

    try:
        rows = await pool.fetch(query, *params)
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("ETF DB 검색 실패: %s", e)
        return []


@tool
async def search_etfs(
    keywords: list[str],
) -> list[dict]:
    """포트폴리오 슬롯에 맞는 ETF를 검색합니다.
    keywords: 자산군·테마 키워드 목록. 예) ["글로벌 주식", "채권", "S&P500"]
    """
    results = await _search_etfs_db(keywords, limit=10)
    return [
        {
            "name": r["name"],
            "ticker": r.get("ticker", ""),
            "idx_ind_nm": r.get("idx_ind_nm") or "",
            "volatility": round(float(r["volatility"]), 1) if r.get("volatility") else None,
            "description": (r.get("description") or "")[:80],
            "product_type": r.get("product_type", "ETF"),
            "interest_rate": float(r.get("interest_rate") or 0.0),
        }
        for r in results
    ]
