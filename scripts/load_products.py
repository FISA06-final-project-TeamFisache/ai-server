"""
MySQL etf_prices 에서 interest_rate(실제 보유기간 연환산 수익률)·volatility 를 계산해
PostgreSQL products 테이블을 갱신하는 스크립트.

load_etf_prices.py 실행 후 호출하세요 (pipeline.py 가 자동으로 연결).

실행:
  python scripts/load_products.py
"""

import asyncio
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

import asyncpg
import pymysql
from dotenv import load_dotenv

load_dotenv()

DB_URL    = os.getenv("DB_URL", "")
MYSQL_URL = os.getenv("MYSQL_URL", "")


def _parse_mysql_url(url: str) -> dict:
    raw = url.replace("mysql+pymysql://", "mysql://")
    p   = urlparse(raw)
    return {
        "host": p.hostname, "port": p.port or 3306,
        "user": p.username, "password": p.password,
        "database": p.path.lstrip("/"), "charset": "utf8mb4",
    }


async def _fetch_tickers_from_pg() -> list[str]:
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch(
            "SELECT ticker FROM products WHERE product_type='ETF' AND ticker IS NOT NULL"
        )
        return [r["ticker"] for r in rows]
    finally:
        await conn.close()


def _calculate_metrics(tickers: list[str]) -> dict[str, dict]:
    """MySQL 가격 이력 → interest_rate(연환산 수익률), volatility(연환산 변동성) 계산."""
    if not tickers or not MYSQL_URL:
        return {}

    try:
        conn = pymysql.connect(**_parse_mysql_url(MYSQL_URL))
    except Exception as e:
        print(f"  MySQL 연결 실패: {e}")
        return {}

    records_by_ticker: dict[str, list[tuple]] = defaultdict(list)
    try:
        placeholders = ",".join(["%s"] * len(tickers))
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT isu_cd, bas_dt, close_prc FROM etf_prices "
                f"WHERE isu_cd IN ({placeholders}) ORDER BY isu_cd, bas_dt",
                tickers,
            )
            for isu_cd, bas_dt, close_prc in cur.fetchall():
                records_by_ticker[isu_cd].append((bas_dt, float(close_prc)))
    finally:
        conn.close()

    result: dict[str, dict] = {}
    for ticker, records in records_by_ticker.items():
        if len(records) < 5:
            continue

        dates  = [d for d, _ in records]
        prices = [p for _, p in records]

        # interest_rate: 가장 이전 가격 대비 최신 가격 연환산 수익률(CAGR)
        interest_rate = None
        if prices[0] > 0 and prices[-1] > 0:
            years = (dates[-1] - dates[0]).days / 365.25
            if years > 0:
                interest_rate = round(((prices[-1] / prices[0]) ** (1 / years) - 1) * 100, 2)

        # volatility: 일간 로그수익률 연환산 표준편차
        log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        n       = len(log_returns)
        mean    = sum(log_returns) / n
        var     = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
        volatility = round(math.sqrt(var) * math.sqrt(252) * 100, 2)

        result[ticker] = {"interest_rate": interest_rate, "volatility": volatility}

    return result


async def _update_pg(metrics: dict[str, dict]) -> None:
    if not metrics:
        return
    now  = datetime.now(timezone.utc).replace(tzinfo=None)
    conn = await asyncpg.connect(DB_URL)
    try:
        for ticker, m in metrics.items():
            await conn.execute(
                "UPDATE products SET interest_rate=$1, volatility=$2, updated_at=$3 WHERE ticker=$4",
                m["interest_rate"], m["volatility"], now, ticker,
            )
        print(f"  {len(metrics)}개 종목 업데이트 완료")
    finally:
        await conn.close()


async def async_main() -> None:
    print("PostgreSQL에서 ETF 티커 조회 중...")
    tickers = await _fetch_tickers_from_pg()
    print(f"  {len(tickers)}개 티커\n")
    if not tickers:
        print("  티커 없음, 종료")
        return

    print("MySQL 가격 이력 → interest_rate · volatility 계산 중...")
    metrics = _calculate_metrics(tickers)
    ir_cnt  = sum(1 for m in metrics.values() if m.get("interest_rate") is not None)
    print(f"  {len(metrics)}개 계산 완료 (interest_rate 산출: {ir_cnt}개)\n")

    print("PostgreSQL 업데이트 중...")
    await _update_pg(metrics)
    print("\nload_products.py 완료")


if __name__ == "__main__":
    asyncio.run(async_main())
