"""
products 테이블에 ETF 상품 데이터를 적재하는 스크립트.

적재 대상:
  - 전체 상장 ETF (KRX Open API, 최근 3년 연평균 수익률(CAGR) 계산 포함, 이름 임베딩)

실행 전 준비:
  pip install asyncpg pgvector requests openai python-dotenv pymysql

실행:
  python scripts/load_products.py
"""

import asyncio
import math
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

import asyncpg
import pymysql
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_URL = os.getenv("DB_URL", "")
MYSQL_URL = os.getenv("MYSQL_URL", "")
KRX_API_KEY = os.getenv("KRX_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DIM = 1536

openai_client = OpenAI(api_key=OPENAI_API_KEY)

KRX_ETF_URL = "https://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"


# ---------------------------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def _get_three_years_ago(dt: datetime) -> datetime:
    """주어진 날짜로부터 정확히 3년 전 날짜를 반환 (윤년 2월 29일 처리)"""
    try:
        return dt.replace(year=dt.year - 3)
    except ValueError:
        # 2월 29일인 경우 윤년 보정을 위해 28일로 처리
        return dt.replace(year=dt.year - 3, day=28)


# ---------------------------------------------------------------------------
# KRX ETF 수집 (KRX Open API 사용 & 3년 CAGR 계산)
# ---------------------------------------------------------------------------

def _fetch_krx_etf(date_str: str) -> list[dict]:
    resp = requests.get(
        url=KRX_ETF_URL,
        headers={"AUTH_KEY": KRX_API_KEY},
        params={"basDd": date_str},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("OutBlock_1", [])


def _nearest_trading_date(base: datetime, max_lookback: int = 14) -> tuple[str, list[dict]]:
    """base 날짜부터 최대 max_lookback일 전까지 ETF 데이터가 있는 가장 가까운 거래일 반환."""
    for delta in range(0, max_lookback):
        d = (base - timedelta(days=delta)).strftime("%Y%m%d")
        rows = _fetch_krx_etf(d)
        if rows:
            return d, rows
    return "", []


def collect_krx_etf() -> list[dict]:
    # KST 기준 날짜 사용 (UTC+9)
    today = datetime.now(timezone.utc) + timedelta(hours=9)

    # 기준일(가장 가까운 거래일) 데이터
    base_date_str, rows_today = _nearest_trading_date(today)
    if not rows_today:
        print("  ETF 기준일 데이터 없음")
        return []

    # 3년 전 가장 가까운 거래일 데이터
    three_years_ago = _get_three_years_ago(today)
    _, rows_3y = _nearest_trading_date(three_years_ago)

    # 3년 전 종가 맵: ISU_CD → 종가
    price_3y_map: dict[str, float] = {}
    for r in rows_3y:
        code = r.get("ISU_CD", "")
        try:
            price = float(r.get("TDD_CLSPRC", "0").replace(",", ""))
            if price > 0:
                price_3y_map[code] = price
        except ValueError:
            pass

    def _parse_int(val: str) -> int | None:
        try:
            v = int(val.replace(",", ""))
            return v if v > 0 else None
        except (ValueError, AttributeError):
            return None

    products = []
    for row in rows_today:
        code = row.get("ISU_CD", "")
        name = row.get("ISU_NM", "")
        if not name:
            continue

        # 레버리지 및 인버스 상품 제외
        if "레버리지" in name or "인버스" in name:
            continue

        # 3년 전에도 거래된 종목만 포함
        close_3y = price_3y_map.get(code)
        if close_3y is None:
            continue

        try:
            close_today = float(row.get("TDD_CLSPRC", "0").replace(",", ""))
        except ValueError:
            close_today = 0.0

        avg_trading_value = _parse_int(row.get("ACC_TRDVAL", "0"))
        mktcap = _parse_int(row.get("MKTCAP", "0"))

        # 일평균 거래대금 1억 원 미만 ETF 제외 (청산위험)
        if avg_trading_value is not None and avg_trading_value < 100_000_000:
            continue

        # 연평균 수익률(CAGR) 계산: (현재가 / 과거가)^(1/연수) - 1
        interest_rate = None
        if close_3y > 0 and close_today > 0:
            cagr = ((close_today / close_3y) ** (1 / 3.0)) - 1
            interest_rate = round(cagr * 100, 2)

        brand = name.split()[0] if name else ""

        desc_parts = [
            f"ETF명: {name}",
            f"종목코드: {code}",
            f"종가: {row.get('TDD_CLSPRC', '')}원",
        ]
        if avg_trading_value:
            desc_parts.append(f"거래대금: {avg_trading_value:,}원")
        if mktcap:
            desc_parts.append(f"시가총액: {mktcap:,}원")
        if interest_rate is not None:
            desc_parts.append(f"최근 3년 연평균 수익률(CAGR): {interest_rate}%")
        desc_parts.append(f"운용사: {brand}")

        products.append({
            "product_type": "ETF",
            "institution": brand,
            "name": name,
            "ticker": code,
            "interest_rate": interest_rate,
            "avg_trading_value": avg_trading_value,
            "mktcap": mktcap,
            "description": " | ".join(desc_parts),
        })

    # 시가총액 상위 100개로 제한
    products.sort(key=lambda x: x.get("mktcap") or 0, reverse=True)
    return products[:100]


# ---------------------------------------------------------------------------
# MySQL volatility 계산
# ---------------------------------------------------------------------------

def fetch_volatility_from_mysql(tickers: list[str]) -> dict[str, float]:
    """MySQL etf_prices 테이블에서 일간 로그수익률로 연환산 변동성(%) 계산."""
    if not tickers or not MYSQL_URL:
        return {}

    raw = MYSQL_URL.replace("mysql+pymysql://", "mysql://")
    p = urlparse(raw)
    try:
        conn = pymysql.connect(
            host=p.hostname,
            port=p.port or 3306,
            user=p.username,
            password=p.password,
            database=p.path.lstrip("/"),
            charset="utf8mb4",
        )
    except Exception as e:
        print(f"  MySQL 연결 실패 (volatility 계산 건너뜀): {e}")
        return {}

    prices_by_ticker: dict[str, list[float]] = defaultdict(list)
    try:
        placeholders = ",".join(["%s"] * len(tickers))
        query = (
            f"SELECT isu_cd, bas_dt, close_prc FROM etf_prices "
            f"WHERE isu_cd IN ({placeholders}) ORDER BY isu_cd, bas_dt"
        )
        with conn.cursor() as cur:
            cur.execute(query, tickers)
            for isu_cd, _bas_dt, close_prc in cur.fetchall():
                prices_by_ticker[isu_cd].append(float(close_prc))
    finally:
        conn.close()

    result: dict[str, float] = {}
    for ticker, prices in prices_by_ticker.items():
        if len(prices) < 5:
            continue
        log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        n = len(log_returns)
        mean = sum(log_returns) / n
        variance = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
        annual_vol = round(math.sqrt(variance) * math.sqrt(252) * 100, 2)
        result[ticker] = annual_vol

    return result


# ---------------------------------------------------------------------------
# DB 적재
# ---------------------------------------------------------------------------

UPSERT_SQL = """
INSERT INTO products (
    id, product_type, institution, name,
    ticker, interest_rate, description, embedding,
    mktcap, avg_trading_value, volatility,
    created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $12)
ON CONFLICT (name, institution)
DO UPDATE SET
    ticker             = EXCLUDED.ticker,
    interest_rate      = EXCLUDED.interest_rate,
    description        = EXCLUDED.description,
    embedding          = EXCLUDED.embedding,
    mktcap             = EXCLUDED.mktcap,
    avg_trading_value  = EXCLUDED.avg_trading_value,
    volatility         = EXCLUDED.volatility,
    updated_at         = EXCLUDED.updated_at
"""


async def ensure_schema(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    for col, coltype in [("mktcap", "bigint"), ("avg_trading_value", "bigint"), ("volatility", "double precision")]:
        exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'products' AND column_name = $1",
            col,
        )
        if not exists:
            await conn.execute(f"ALTER TABLE products ADD COLUMN {col} {coltype}")
            print(f"  {col} {coltype} 컬럼 추가 완료")

    col_exists = await conn.fetchval("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'products' AND column_name = 'embedding'
    """)

    if not col_exists:
        await conn.execute(f"ALTER TABLE products ADD COLUMN embedding vector({VECTOR_DIM})")
        print(f"  embedding vector({VECTOR_DIM}) 컬럼 추가 완료")
    else:
        # DB의 설정된 벡터 타입 문자열(예: 'vector(1536)')을 확인합니다.
        try:
            current_type = await conn.fetchval("""
                SELECT format_type(atttypid, atttypmod)
                FROM pg_attribute
                WHERE attrelid = 'products'::regclass AND attname = 'embedding' AND attisdropped = false
            """)
            expected_type = f"vector({VECTOR_DIM})"
            if current_type and current_type != expected_type:
                print(f"⚠️ DB 임베딩 차원 불일치 감지 (현재 DB: {current_type}, 모델 차원: {expected_type})")
                print("기존 embedding 컬럼을 삭제(인덱스 포함)하고 새 차원으로 재생성합니다...")
                # CASCADE를 추가하여 기존 벡터 인덱스가 있어도 강제로 삭제되도록 합니다.
                await conn.execute("ALTER TABLE products DROP COLUMN embedding CASCADE")
                await conn.execute(f"ALTER TABLE products ADD COLUMN embedding vector({VECTOR_DIM})")
                print(f"  embedding vector({VECTOR_DIM}) 컬럼 재생성 완료")
        except Exception as e:
            print(f"  차원 검증 중 예외 발생 (무시됨): {e}")

    constraint_exists = await conn.fetchval("""
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = 'products'
          AND constraint_type = 'UNIQUE'
          AND constraint_name = 'uq_products_name_institution'
    """)
    if not constraint_exists:
        await conn.execute(
            "ALTER TABLE products ADD CONSTRAINT uq_products_name_institution "
            "UNIQUE (name, institution)"
        )
        print("  UNIQUE(name, institution) 제약 추가 완료")


async def load(products: list[dict]) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conn: asyncpg.Connection = await asyncpg.connect(DB_URL)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await ensure_schema(conn)

        total = len(products)
        for i, p in enumerate(products, 1):
            print(f"[{i}/{total}] 임베딩 생성: {p['name'][:40]}")
            vector = embed(p["name"])
            vector_str = "[" + ",".join(str(v) for v in vector) + "]"

            await conn.execute(
                UPSERT_SQL,
                str(uuid.uuid4()),
                p["product_type"],
                p["institution"],
                p["name"],
                p.get("ticker"),
                p["interest_rate"],
                p["description"],
                vector_str,
                p.get("mktcap"),
                p.get("avg_trading_value"),
                p.get("volatility"),
                now,
            )
            print(f"  → 적재 완료")
    finally:
        await conn.close()


async def main() -> None:
    print("=== ETF 상품 데이터 수집 시작 ===\n")

    print("KRX ETF 수집 중...")
    try:
        etfs = collect_krx_etf()
        print(f"  → {len(etfs)}건\n")
    except Exception as e:
        print(f"  → 실패: {e}\n")
        return

    if not etfs:
        print("수집된 ETF가 없어 종료합니다.")
        return

    print("MySQL에서 volatility 계산 중...")
    tickers = [e["ticker"] for e in etfs if e.get("ticker")]
    vol_map = fetch_volatility_from_mysql(tickers)
    for e in etfs:
        e["volatility"] = vol_map.get(e.get("ticker"))
    found = sum(1 for e in etfs if e.get("volatility") is not None)
    print(f"  → {found}/{len(etfs)}건 volatility 산출\n")

    print(f"총 {len(etfs)}건 → DB 적재 시작\n")
    await load(etfs)
    print("\n=== 완료 ===")


if __name__ == "__main__":
    asyncio.run(main())
