from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf
from langchain_core.tools import tool

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


MINI_CHALLENGE_TOOLS = [get_stock_prices]
