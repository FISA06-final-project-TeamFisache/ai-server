"""
1. KRX API (today 기준) → ETF 후보 목록
2. MySQL GROUP BY → 가격 이력 존재 티커만 필터
3. 필터된 종목 → PostgreSQL products upsert (임베딩 포함)
4. MySQL 가격 이력 → interest_rate · volatility 계산 → PostgreSQL 갱신

실행:
  python scripts/load_products.py                    # 오늘 날짜 기준
  python scripts/load_products.py --today 20260609
"""

import argparse
import asyncio
import math
import os
import sys
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import asyncpg
import pymysql
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from load_etf_prices import select_products

load_dotenv()

DB_URL          = os.getenv("DB_URL", "")
MYSQL_URL       = os.getenv("MYSQL_URL", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DIM      = 1536

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def _parse_mysql_url(url: str) -> dict:
    raw = url.replace("mysql+pymysql://", "mysql://")
    p   = urlparse(raw)
    return {
        "host": p.hostname, "port": p.port or 3306,
        "user": p.username, "password": p.password,
        "database": p.path.lstrip("/"), "charset": "utf8mb4",
    }


# ── MySQL: 가격 이력 존재 티커 조회 ───────────────────────────────────────────

def _fetch_mysql_tickers() -> set[str]:
    """MySQL etf_prices에 가격 이력이 존재하는 티커 집합."""
    if not MYSQL_URL:
        return set()
    try:
        conn = pymysql.connect(**_parse_mysql_url(MYSQL_URL))
    except Exception as e:
        print(f"  MySQL 연결 실패: {e}")
        return set()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT isu_cd FROM etf_prices GROUP BY isu_cd")
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


# ── PostgreSQL: products upsert ────────────────────────────────────────────────

_UPSERT_PRODUCT = """
INSERT INTO products (
    id, product_type, institution, name,
    ticker, description, embedding,
    avg_trading_value, acc_trdvol, idx_ind_nm, close_prc, nav,
    created_at, updated_at
)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$13)
ON CONFLICT (name, institution) DO UPDATE SET
    ticker            = EXCLUDED.ticker,
    description       = EXCLUDED.description,
    embedding         = EXCLUDED.embedding,
    avg_trading_value = EXCLUDED.avg_trading_value,
    acc_trdvol        = EXCLUDED.acc_trdvol,
    idx_ind_nm        = EXCLUDED.idx_ind_nm,
    close_prc         = EXCLUDED.close_prc,
    nav               = EXCLUDED.nav,
    updated_at        = EXCLUDED.updated_at
"""


def _embed(text: str) -> list[float]:
    return openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding


async def _ensure_pg_schema(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    new_cols = [
        ("avg_trading_value", "bigint"),
        ("volatility",        "double precision"),
        ("acc_trdvol",        "bigint"),
        ("idx_ind_nm",        "text"),
        ("close_prc",         "double precision"),
        ("nav",               "double precision"),
    ]
    for col, coltype in new_cols:
        exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns WHERE table_name='products' AND column_name=$1", col
        )
        if not exists:
            await conn.execute(f"ALTER TABLE products ADD COLUMN {col} {coltype}")
            print(f"  컬럼 추가: {col} {coltype}")

    emb_exists = await conn.fetchval(
        "SELECT 1 FROM information_schema.columns WHERE table_name='products' AND column_name='embedding'"
    )
    if not emb_exists:
        await conn.execute(f"ALTER TABLE products ADD COLUMN embedding vector({VECTOR_DIM})")
        print(f"  embedding vector({VECTOR_DIM}) 추가")
    else:
        try:
            cur_type = await conn.fetchval(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid='products'::regclass AND attname='embedding' AND attisdropped=false"
            )
            if cur_type and cur_type != f"vector({VECTOR_DIM})":
                print(f"  임베딩 차원 불일치 ({cur_type} → vector({VECTOR_DIM})), 재생성...")
                await conn.execute("ALTER TABLE products DROP COLUMN embedding CASCADE")
                await conn.execute(f"ALTER TABLE products ADD COLUMN embedding vector({VECTOR_DIM})")
        except Exception as e:
            print(f"  차원 검증 예외 (무시): {e}")

    constraint_exists = await conn.fetchval(
        "SELECT 1 FROM information_schema.table_constraints "
        "WHERE table_name='products' AND constraint_type='UNIQUE' "
        "AND constraint_name='uq_products_name_institution'"
    )
    if not constraint_exists:
        await conn.execute(
            "ALTER TABLE products ADD CONSTRAINT uq_products_name_institution UNIQUE (name, institution)"
        )
        print("  UNIQUE(name, institution) 제약 추가")


async def upsert_products_to_pg(products: list[dict]) -> None:
    now  = datetime.now(timezone.utc).replace(tzinfo=None)
    conn = await asyncpg.connect(DB_URL)
    try:
        await _ensure_pg_schema(conn)
        total = len(products)
        for i, p in enumerate(products, 1):
            print(f"[{i}/{total}] 임베딩: {p['name'][:40]}")
            embed_text = p["name"]
            if p.get("idx_ind_nm"):
                embed_text += f" {p['idx_ind_nm']}"
            vec_str = "[" + ",".join(str(v) for v in _embed(embed_text)) + "]"
            await conn.execute(
                _UPSERT_PRODUCT,
                str(uuid.uuid4()),
                p["product_type"], p["institution"], p["name"],
                p["ticker"], p["description"], vec_str,
                p.get("avg_trading_value"), p.get("acc_trdvol"),
                p.get("idx_ind_nm"), p.get("close_prc"), p.get("nav"),
                now,
            )
        print(f"  PostgreSQL upsert 완료: {total}건\n")
    finally:
        await conn.close()


# ── MySQL → metrics 계산 ───────────────────────────────────────────────────────

def _calculate_metrics(tickers: list[str]) -> dict[str, dict]:
    """MySQL 가격 이력 → interest_rate(연환산 수익률), volatility(연환산 변동성) 계산.

    기준일: MySQL etf_prices 전체의 MAX(bas_dt)
    기간: 기준일로부터 3년 전 이후 데이터만 사용
    """
    if not tickers or not MYSQL_URL:
        return {}

    try:
        conn = pymysql.connect(**_parse_mysql_url(MYSQL_URL))
    except Exception as e:
        print(f"  MySQL 연결 실패: {e}")
        return {}

    records_by_ticker: dict[str, list[tuple]] = defaultdict(list)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(bas_dt) FROM etf_prices")
            row = cur.fetchone()
            max_date: date = row[0] if row and row[0] else date.today()

        cutoff_date = max_date - timedelta(days=round(365.25 * 3))
        print(f"  기준일: {max_date}  /  3년 전 기준: {cutoff_date}")

        placeholders = ",".join(["%s"] * len(tickers))
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT isu_cd, bas_dt, close_prc FROM etf_prices "
                f"WHERE isu_cd IN ({placeholders}) AND bas_dt >= %s ORDER BY isu_cd, bas_dt",
                (*tickers, cutoff_date),
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

        interest_rate = None
        if prices[0] > 0 and prices[-1] > 0:
            years = (dates[-1] - dates[0]).days / 365.25
            if years > 0:
                interest_rate = round(((prices[-1] / prices[0]) ** (1 / years) - 1) * 100, 2)

        log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        n       = len(log_returns)
        mean    = sum(log_returns) / n
        var     = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
        volatility = round(math.sqrt(var) * math.sqrt(252) * 100, 2)

        result[ticker] = {"interest_rate": interest_rate, "volatility": volatility}

    return result


# ── PostgreSQL: metrics 갱신 ───────────────────────────────────────────────────

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


# ── 진입점 ────────────────────────────────────────────────────────────────────

async def async_main(today: date) -> None:
    print("=== [1/3] KRX ETF 종목 조회 ===\n")
    products = select_products(today)
    if not products:
        print("  KRX 조회 결과 없음, 종료")
        return
    print(f"  KRX 후보: {len(products)}개\n")

    print("=== [2/3] MySQL 교차 필터 ===\n")
    mysql_tickers = _fetch_mysql_tickers()
    print(f"  MySQL 보유 티커: {len(mysql_tickers)}개")

    filtered = [p for p in products if p["ticker"] in mysql_tickers]
    print(f"  적재 대상 (교집합): {len(filtered)}개\n")

    if not filtered:
        print("  적재할 종목 없음, 종료")
        return

    print("=== [3/3] PostgreSQL 적재 + 지표 갱신 ===\n")
    await upsert_products_to_pg(filtered)

    tickers = [p["ticker"] for p in filtered]
    print("MySQL 가격 이력 → interest_rate · volatility 계산 중...")
    metrics = _calculate_metrics(tickers)
    ir_cnt  = sum(1 for m in metrics.values() if m.get("interest_rate") is not None)
    print(f"  {len(metrics)}개 계산 완료 (interest_rate 산출: {ir_cnt}개)\n")

    print("PostgreSQL 업데이트 중...")
    await _update_pg(metrics)
    print("\nload_products.py 완료")


def _parse_date(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF 메타데이터 + 지표 → PostgreSQL")
    parser.add_argument("--today", type=str, metavar="YYYYMMDD", help="기준일 (기본: 오늘)")
    args = parser.parse_args()

    today = _parse_date(args.today) if args.today else date.today()
    asyncio.run(async_main(today))


if __name__ == "__main__":
    main()
