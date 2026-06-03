"""
products 테이블에 금융 상품 데이터를 적재하는 스크립트.

적재 대상:
  - 우리은행 정기예금 (FSS API)
  - 우리은행 적금     (FSS API)
  - 전체 상장 ETF     (KRX Open API, 최근 1년 수익률 계산 포함)
  - 국채전문유통시장 채권 (KRX API, 최근 1년 수익률 계산 포함)
  - 우리투자증권 ISA  (정적 데이터 하드코딩)
  - 우리투자증권 IRP  (정적 데이터 하드코딩)
  - 우리투자증권 연금저축계좌 (정적 데이터 하드코딩)

실행 전 준비:
  pip install asyncpg pgvector requests openai python-dotenv pandas

실행:
  python scripts/load_products.py
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone, timedelta

import asyncpg
import requests
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL", "")
FSS_API_KEY = os.getenv("FSS_API_KEY", "")
KRX_API_KEY = os.getenv("KRX_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")
VECTOR_DIM = 2048

FSS_BASE = "https://finlife.fss.or.kr/finlifeapi"
KRX_ETF_URL  = "https://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"
KRX_BOND_URL = "https://data-dbg.krx.co.kr/svc/apis/bon/kts_bydd_trd"

# 우리투자증권 ISA/IRP/연금저축 상품 (하드코딩)
STATIC_PRODUCTS: list[dict] = [
    {
        "product_type": "STOCK",
        "institution": "우리투자증권",
        "name": "우리투자증권 중개형 ISA",
        "interest_rate": None,
        "description": (
            "개인종합자산관리계좌(ISA). "
            "가입대상: 19세 이상 또는 근로소득이 있는 15세 이상 거주자. "
            "납입한도: 연 2,000만 원, 총 1억 원. "
            "의무가입기간 3년. 국내 상장주식, 펀드, 채권, ETF/ETN, RP 등 투자 가능. "
            "일반형은 200만 원까지, 서민형 등은 400만 원까지 비과세 혜택이 있으며 "
            "초과분은 9.9% 분리과세."
        ),
    },
    {
        "product_type": "IRP",
        "institution": "우리투자증권",
        "name": "우리투자증권 개인형 IRP",
        "interest_rate": None,
        "description": (
            "개인형 퇴직연금(IRP). "
            "퇴직금 또는 본인 추가 납입금을 운용하며, 연간 납입한도는 연금저축 포함 1,800만 원. "
            "ETF, 펀드, 채권, 예금 등 운용 가능. "
            "만 55세 이후 연금 수령 가능하며, 연금 수령 시 연금소득세가 적용됨. "
            "세액공제 한도와 중도해지 과세는 관련 세법 기준을 따름."
        ),
    },
    {
        "product_type": "PENSION_SAVINGS",
        "institution": "우리투자증권",
        "name": "우리투자증권 연금저축계좌",
        "interest_rate": None,
        "description": (
            "연금저축계좌. 소득세법에서 정한 연금 수령 요건에 따라 자금을 인출하는 경우 "
            "연금소득으로 과세되는 상품. 가입기간 5년 이상, 만 55세 이후 연금 개시 가능. "
            "연간 납입한도는 퇴직연금 등과 합산하여 1,800만 원이며, "
            "세액공제 한도와 공제율은 관련 세법 기준을 따름."
        ),
    },
]


# ---------------------------------------------------------------------------
# 임베딩
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    resp = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# FSS 상품 수집
# ---------------------------------------------------------------------------

def _fss_fetch(endpoint: str, fin_prdt_cd_filter: str | None = None) -> list[dict]:
    results = []
    page = 1
    while True:
        params = {
            "auth": FSS_API_KEY,
            "topFinGrpNo": "020000",
            "pageNo": str(page),
        }
        resp = requests.get(f"{FSS_BASE}/{endpoint}", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("result", {})

        base_list: list[dict] = data.get("baseList", [])
        opt_list: list[dict] = data.get("optionList", [])
        max_page = int(data.get("max_page_no", 1))

        opt_map: dict[str, list[dict]] = {}
        for opt in opt_list:
            key = opt.get("fin_prdt_cd", "")
            opt_map.setdefault(key, []).append(opt)

        for item in base_list:
            if item.get("kor_co_nm", "") != "우리은행":
                continue
            item["_options"] = opt_map.get(item.get("fin_prdt_cd", ""), [])
            results.append(item)

        if page >= max_page:
            break
        page += 1

    return results


def _deposit_description(item: dict) -> str:
    opts = item.get("_options", [])
    terms = ", ".join(
        f"{o.get('save_trm')}개월({o.get('intr_rate')}%)"
        for o in opts
        if o.get("save_trm") and o.get("intr_rate")
    )
    parts = [
        f"금융회사: 우리은행",
        f"상품명: {item.get('fin_prdt_nm', '')}",
        f"가입방법: {item.get('join_way', '')}",
        f"가입대상: {item.get('join_member', '')}",
        f"우대조건: {item.get('spcl_cnd', '')}",
        f"만기후이자: {item.get('mtrt_int', '')}",
        f"기타유의사항: {item.get('etc_note', '')}",
    ]
    if terms:
        parts.append(f"저축기간별금리: {terms}")
    return " | ".join(p for p in parts if p.split(": ", 1)[-1])


def collect_fss_deposits() -> list[dict]:
    raw = _fss_fetch("depositProductsSearch.json")
    products = []
    for item in raw:
        products.append({
            "product_type": "DEPOSIT",
            "institution": "우리은행",
            "name": item.get("fin_prdt_nm", ""),
            "interest_rate": None,
            "description": _deposit_description(item),
        })
    return products


def _saving_description(item: dict) -> str:
    opts = item.get("_options", [])
    terms = ", ".join(
        f"{o.get('save_trm')}개월/{o.get('rsrv_type_nm','')}({o.get('intr_rate')}%)"
        for o in opts
        if o.get("save_trm") and o.get("intr_rate")
    )
    parts = [
        f"금융회사: 우리은행",
        f"상품명: {item.get('fin_prdt_nm', '')}",
        f"가입방법: {item.get('join_way', '')}",
        f"가입대상: {item.get('join_member', '')}",
        f"우대조건: {item.get('spcl_cnd', '')}",
        f"만기후이자: {item.get('mtrt_int', '')}",
        f"기타유의사항: {item.get('etc_note', '')}",
        f"최고한도: {item.get('max_limit', '')}원",
    ]
    if terms:
        parts.append(f"저축기간별금리: {terms}")
    return " | ".join(p for p in parts if p.split(": ", 1)[-1])


def collect_fss_savings() -> list[dict]:
    raw = _fss_fetch("savingProductsSearch.json")
    products = []
    for item in raw:
        products.append({
            "product_type": "SAVING",
            "institution": "우리은행",
            "name": item.get("fin_prdt_nm", ""),
            "interest_rate": None,
            "description": _saving_description(item),
        })
    return products


# ---------------------------------------------------------------------------
# KRX ETF 수집 (KRX Open API 사용 & 1년 수익률 계산)
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


def _nearest_trading_date(base: datetime, max_lookback: int = 8) -> tuple[str, list[dict]]:
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

    # 1년 전 가장 가까운 거래일 데이터
    one_year_ago = today - timedelta(days=365)
    _, rows_1y = _nearest_trading_date(one_year_ago)

    # 1년 전 종가 맵: ISU_CD → 종가
    price_1y_map: dict[str, float] = {}
    for r in rows_1y:
        code = r.get("ISU_CD", "")
        try:
            price = float(r.get("TDD_CLSPRC", "0").replace(",", ""))
            if price > 0:
                price_1y_map[code] = price
        except ValueError:
            pass

    products = []
    for row in rows_today:
        code = row.get("ISU_CD", "")
        name = row.get("ISU_NM", "")
        if not name:
            continue

        # 1년 전에도 거래된 종목만 포함
        close_1y = price_1y_map.get(code)
        if close_1y is None:
            continue

        try:
            close_today = float(row.get("TDD_CLSPRC", "0").replace(",", ""))
        except ValueError:
            close_today = 0.0

        interest_rate = None
        if close_1y > 0 and close_today > 0:
            interest_rate = round(((close_today - close_1y) / close_1y) * 100, 2)

        brand = name.split()[0] if name else ""

        description = (
            f"ETF명: {name} | "
            f"종목코드: {code} | "
            f"종가: {row.get('TDD_CLSPRC', '')}원 | "
            f"거래량: {row.get('ACC_TRDVOL', '')} | "
        )
        if interest_rate is not None:
            description += f"최근 1년 수익률: {interest_rate}% | "
        description += f"운용사: {brand}"

        products.append({
            "product_type": "ETF",
            "institution": brand,
            "name": name,
            "interest_rate": interest_rate,
            "description": description,
        })

    return products


# ---------------------------------------------------------------------------
# KRX 국채 수집 (1년 수익률 계산 추가)
# ---------------------------------------------------------------------------

def _fetch_krx_bonds(date_str: str) -> list[dict]:
    resp = requests.get(
        url=KRX_BOND_URL,
        headers={"AUTH_KEY": KRX_API_KEY},
        params={"basDd": date_str},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("OutBlock_1", [])

def collect_krx_bonds() -> list[dict]:
    today = datetime.now(timezone.utc)
    
    # 최근 영업일 데이터 확보
    rows_today = []
    for delta in range(0, 8):
        d = (today - timedelta(days=delta)).strftime("%Y%m%d")
        rows = _fetch_krx_bonds(d)
        if rows:
            rows_today = rows
            break
            
    # 1년 전 영업일 데이터 확보
    one_year_ago = today - timedelta(days=365)
    rows_1y = []
    for delta in range(0, 8):
        d = (one_year_ago - timedelta(days=delta)).strftime("%Y%m%d")
        rows = _fetch_krx_bonds(d)
        if rows:
            rows_1y = rows
            break

    # 1년 전 종가 맵핑
    price_1y_map = {}
    for r in rows_1y:
        code = r.get("ISU_CD", "")
        try:
            price = float(r.get("CLSPRC", "0").replace(",", ""))
            if price > 0:
                price_1y_map[code] = price
        except ValueError:
            pass

    products = []
    for row in rows_today:
        name: str = row.get("ISU_NM", "")
        if not name:
            continue

        code = row.get('ISU_CD', '')
        try:
            close_today = float(row.get('CLSPRC', '0').replace(",", ""))
        except ValueError:
            close_today = 0.0

        # 최근 1년 수익률 계산
        interest_rate = None
        close_1y = price_1y_map.get(code)
        if close_1y and close_today > 0:
            interest_rate = round(((close_today - close_1y) / close_1y) * 100, 2)

        description = (
            f"채권명: {name} | "
            f"종목코드: {code} | "
            f"시장구분: {row.get('MKT_NM', '')} | "
            f"종목구분: {row.get('GOVBND_ISU_TP_NM', '')} | "
            f"만기연수: {row.get('BND_EXP_TP_NM', '')} | "
            f"종가(가격): {row.get('CLSPRC', '')} | "
        )
        if interest_rate is not None:
             description += f"최근 1년 수익률: {interest_rate}% | "

        products.append({
            "product_type": "BOND",
            "institution": "한국거래소",
            "name": name,
            "interest_rate": interest_rate,
            "description": description,
        })

    return products


# ---------------------------------------------------------------------------
# DB 적재
# ---------------------------------------------------------------------------

UPSERT_SQL = """
INSERT INTO products (
    id, product_type, institution, name,
    interest_rate, description, embedding,
    created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
ON CONFLICT (name, institution)
DO UPDATE SET
    interest_rate = EXCLUDED.interest_rate,
    description   = EXCLUDED.description,
    embedding     = EXCLUDED.embedding,
    updated_at    = EXCLUDED.updated_at
"""

async def ensure_schema(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    col_exists = await conn.fetchval("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'products' AND column_name = 'embedding'
    """)
    if not col_exists:
        await conn.execute(
            f"ALTER TABLE products ADD COLUMN embedding vector({VECTOR_DIM})"
        )
        print(f"embedding vector({VECTOR_DIM}) 컬럼 추가 완료")

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
        print("UNIQUE(name, institution) 제약 추가 완료")

async def load(products: list[dict]) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conn: asyncpg.Connection = await asyncpg.connect(DB_URL)

    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await ensure_schema(conn)

        total = len(products)
        for i, p in enumerate(products, 1):
            desc = p["description"] or p["name"]
            print(f"[{i}/{total}] 임베딩 생성: {p['name'][:40]}")
            vector = embed(desc)
            vector_str = "[" + ",".join(str(v) for v in vector) + "]"

            await conn.execute(
                UPSERT_SQL,
                str(uuid.uuid4()),
                p["product_type"],
                p["institution"],
                p["name"],
                p["interest_rate"],  # 이제 수익률 값도 들어갑니다.
                p["description"],
                vector_str,
                now,
            )
            print(f"  → 적재 완료")
    finally:
        await conn.close()


def _try_collect(label: str, fn) -> list[dict]:
    print(f"{label} 수집 중...")
    try:
        result = fn()
        print(f"  → {len(result)}건\n")
        return result
    except Exception as e:
        print(f"  → 실패 (스킵): {e}\n")
        return []


async def main() -> None:
    print("=== 상품 데이터 수집 시작 ===\n")

    deposits = _try_collect("[1/5] FSS 정기예금", collect_fss_deposits)
    savings  = _try_collect("[2/5] FSS 적금",        collect_fss_savings)
    etfs     = _try_collect("[3/5] KRX ETF (KRX API)", collect_krx_etf)
    bonds    = _try_collect("[4/5] KRX 국채",        collect_krx_bonds)

    print("[5/5] ISA/IRP/연금저축계좌 정적 데이터 준비...")
    static = STATIC_PRODUCTS
    print(f"  → {len(static)}건\n")

    all_products = deposits + savings + etfs + bonds + static
    if not all_products:
        print("수집된 상품이 없어 종료합니다.")
        return

    print(f"총 {len(all_products)}건 → DB 적재 시작\n")
    await load(all_products)

    print("\n=== 완료 ===")


if __name__ == "__main__":
    asyncio.run(main())