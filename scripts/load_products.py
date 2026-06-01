"""
products 테이블에 금융 상품 데이터를 적재하는 스크립트.

적재 대상:
  - 우리은행 정기예금 (FSS API)
  - 우리은행 적금     (FSS API)
  - 우리자산운용 ETF  (KRX API, ISU_NM에 'WON' 포함)
  - 우리투자증권 ISA  (하드코딩)
  - 우리투자증권 IRP  (하드코딩)

실행 전 준비:
  pip install asyncpg asyncpg pgvector requests openai python-dotenv

실행:
  python scripts/load_products.py
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any

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
KRX_ETF_URL = "https://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"

# 우리투자증권 ISA/IRP 상품 (하드코딩)
STATIC_PRODUCTS: list[dict] = [
    {
        "product_type": "STOCK",
        "institution": "우리투자증권",
        "name": "우리투자증권 중개형 ISA",
        "interest_rate": None,
        "description": (
            "개인종합자산관리계좌(ISA). 주식, ETF, 펀드, 채권 등 다양한 금융상품을 "
            "하나의 계좌에서 운용하며 발생한 순이익 200만 원(서민형 400만 원)까지 비과세, "
            "초과분은 9.9% 분리과세 혜택. 의무가입기간 3년. 납입한도 연 2,000만 원(총 1억 원). "
            "국내 상장 주식 및 국내 주식형 펀드 투자 가능."
        ),
    },
    {
        "product_type": "IRP",
        "institution": "우리투자증권",
        "name": "우리투자증권 개인형 IRP",
        "interest_rate": None,
        "description": (
            "개인형 퇴직연금(IRP). 퇴직금 또는 자기 부담금을 납입해 운용하며 "
            "연간 최대 900만 원(ISA 합산 포함)까지 세액공제(16.5% 또는 13.2%). "
            "ETF, 펀드, 채권, 예금 등 안전자산 30% 이상 의무 편입. "
            "만 55세 이후 연금 수령 시 낮은 연금소득세 적용. "
            "중도 해지 시 기타소득세(16.5%) 부과."
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
    """FSS API를 페이지 단위로 전체 조회하여 우리은행 상품만 반환."""
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
        names = sorted({i.get("kor_co_nm", "") for i in base_list})
        print(f"  [DEBUG] page={page}/{max_page} 전체={len(base_list)}건 회사목록: {names}")

        # 상품코드 → 옵션 매핑
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
        f"{o.get('save_trm')}개월/{o.get('rsrv_type_nm','')}"
        f"({o.get('intr_rate')}%)"
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
# KRX ETF 수집
# ---------------------------------------------------------------------------

def collect_krx_etf() -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    resp = requests.post(
        KRX_ETF_URL,
        json={"AUTH_KEY": KRX_API_KEY, "basDd": today},
        timeout=15,
    )
    resp.raise_for_status()

    rows: list[dict] = resp.json().get("OutBlock_1", [])

    # 거래 없는 날(주말/공휴일)이면 전 영업일로 재시도
    if not rows:
        from datetime import timedelta
        for delta in range(1, 8):
            prev = (datetime.now(timezone.utc) - timedelta(days=delta)).strftime("%Y%m%d")
            resp = requests.post(
                KRX_ETF_URL,
                json={"AUTH_KEY": KRX_API_KEY, "basDd": prev},
                timeout=15,
            )
            resp.raise_for_status()
            rows = resp.json().get("OutBlock_1", [])
            if rows:
                break

    products = []
    for row in rows:
        name: str = row.get("ISU_NM", "")
        if "WON" not in name:
            continue

        try:
            close_price = float(row.get("TDD_CLSPRC", "0").replace(",", ""))
        except ValueError:
            close_price = 0.0

        try:
            nav = float(row.get("NAV", "0").replace(",", ""))
        except ValueError:
            nav = 0.0

        description = (
            f"ETF명: {name} | "
            f"종목코드: {row.get('ISU_CD', '')} | "
            f"기초지수: {row.get('IDX_IND_NM', '')} | "
            f"종가: {row.get('TDD_CLSPRC', '')}원 | "
            f"NAV: {row.get('NAV', '')} | "
            f"등락률: {row.get('FLUC_RT', '')}% | "
            f"시가총액: {row.get('MKTCAP', '')}원 | "
            f"운용사: 우리자산운용"
        )

        products.append({
            "product_type": "STOCK",
            "institution": "우리자산운용",
            "name": name,
            "interest_rate": None,
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


async def ensure_embedding_column(conn: asyncpg.Connection) -> None:
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
    else:
        print("embedding 컬럼 이미 존재, 스킵")


async def load(products: list[dict]) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conn: asyncpg.Connection = await asyncpg.connect(DB_URL)

    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await ensure_embedding_column(conn)

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
                p["interest_rate"],
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

    deposits = _try_collect("[1/4] FSS 정기예금", collect_fss_deposits)
    savings  = _try_collect("[2/4] FSS 적금",    collect_fss_savings)
    etfs     = _try_collect("[3/4] KRX ETF",     collect_krx_etf)

    print("[4/4] ISA/IRP 정적 데이터 준비...")
    static = STATIC_PRODUCTS
    print(f"  → {len(static)}건\n")

    all_products = deposits + savings + etfs + static
    if not all_products:
        print("수집된 상품이 없어 종료합니다.")
        return

    print(f"총 {len(all_products)}건 → DB 적재 시작\n")
    await load(all_products)

    print("\n=== 완료 ===")


if __name__ == "__main__":
    asyncio.run(main())
