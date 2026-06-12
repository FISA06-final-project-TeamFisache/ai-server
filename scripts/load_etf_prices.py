"""
1. KRX API → 순자산총액 TOP 200 + 3년 이력 필터로 적재 대상 ETF 선정
2. 선정 종목의 일간 종가 → MySQL etf_prices 적재

사용법:
  python scripts/load_etf_prices.py              # 최근 30일 (기본)
  python scripts/load_etf_prices.py --days 60
  python scripts/load_etf_prices.py --start 20230609 --end 20260609
"""

import argparse
import asyncio
import os
import time
from datetime import date, timedelta
from urllib.parse import urlparse

import pymysql
import requests
from dotenv import load_dotenv

load_dotenv()

KRX_API_KEY = os.getenv("KRX_API_KEY", "")
MYSQL_URL   = os.getenv("MYSQL_URL", "")

KRX_ETF_URL = "https://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"

# ── KRX ───────────────────────────────────────────────────────────────────────

def _fetch_krx(date_str: str) -> list[dict]:
    try:
        resp = requests.get(
            url=KRX_ETF_URL,
            headers={"AUTH_KEY": KRX_API_KEY},
            params={"basDd": date_str},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("OutBlock_1", [])
    except Exception as e:
        print(f"  KRX 조회 실패 ({date_str}): {e}")
        return []


def _nearest_trading_day(base: date, max_lookback: int = 14) -> tuple[str, list[dict]]:
    for delta in range(max_lookback):
        d = (base - timedelta(days=delta)).strftime("%Y%m%d")
        rows = _fetch_krx(d)
        if rows:
            return d, rows
    return "", []


def _parse_int(val) -> int | None:
    try:
        v = int(str(val).replace(",", ""))
        return v if v > 0 else None
    except (ValueError, AttributeError):
        return None


def _parse_float(val) -> float | None:
    try:
        v = float(str(val).replace(",", ""))
        return v if v > 0 else None
    except (ValueError, AttributeError):
        return None


def select_products(today: date | None = None) -> list[dict]:
    """순자산총액 TOP 200 + 3년 이력 조건을 통과한 ETF 메타데이터 목록 반환."""
    if today is None:
        today = date.today()
    try:
        three_years_ago = today.replace(year=today.year - 3)
    except ValueError:
        three_years_ago = today.replace(year=today.year - 3, day=28)

    print("오늘 기준 ETF 데이터 조회 중...")
    today_str, rows_today = _nearest_trading_day(today)
    if not rows_today:
        print("  기준일 데이터 없음")
        return []
    print(f"  기준일: {today_str}, {len(rows_today)}건")

    print("3년 전 기준 ETF 데이터 조회 중...")
    past_str, rows_3y = _nearest_trading_day(three_years_ago)
    print(f"  3년 전 기준일: {past_str}, {len(rows_3y)}건")

    past_codes = {r.get("ISU_CD") for r in rows_3y if r.get("ISU_CD")}

    candidates = []
    for r in rows_today:
        code = r.get("ISU_CD", "")
        name = r.get("ISU_NM", "")
        if not code or not name:
            continue
        if "레버리지" in name or "인버스" in name:
            continue
        nav_total = _parse_int(r.get("INVSTASST_NETASST_TOTAMT")) or 0
        candidates.append({"code": code, "name": name, "nav_total": nav_total, "raw": r})

    candidates.sort(key=lambda x: x["nav_total"], reverse=True)
    top200 = candidates[:200]

    selected = [c for c in top200 if c["code"] in past_codes]
    print(f"  TOP 200 중 3년 이력 있음: {len(selected)}개 (제외: {len(top200) - len(selected)}개)\n")

    products = []
    for c in selected:
        r    = c["raw"]
        name = c["name"]
        institution       = name.split()[0] if name else ""
        avg_trading_value = _parse_int(r.get("ACC_TRDVAL"))
        acc_trdvol        = _parse_int(r.get("ACC_TRDVOL"))
        close_prc         = _parse_float(r.get("TDD_CLSPRC"))
        nav               = _parse_float(r.get("NAV"))
        idx_ind_nm        = (r.get("IDX_IND_NM") or "").strip()

        desc_parts = [f"ETF명: {name}", f"종목코드: {c['code']}"]
        if close_prc:
            desc_parts.append(f"종가: {close_prc:,.0f}원")
        if avg_trading_value:
            desc_parts.append(f"거래대금: {avg_trading_value:,}원")
        if idx_ind_nm:
            desc_parts.append(f"기초지수: {idx_ind_nm}")
        desc_parts.append(f"운용사: {institution}")

        products.append({
            "product_type":      "ETF",
            "institution":       institution,
            "name":              name,
            "ticker":            c["code"],
            "description":       " | ".join(desc_parts),
            "avg_trading_value": avg_trading_value,
            "acc_trdvol":        acc_trdvol,
            "idx_ind_nm":        idx_ind_nm,
            "close_prc":         close_prc,
            "nav":               nav,
        })

    return products


# ── MySQL: 가격 적재 ───────────────────────────────────────────────────────────

_CREATE_PRICE_TABLE = """
CREATE TABLE IF NOT EXISTS etf_prices (
    isu_cd    VARCHAR(20)    NOT NULL,
    bas_dt    DATE           NOT NULL,
    close_prc DECIMAL(15, 2) NOT NULL,
    PRIMARY KEY (isu_cd, bas_dt)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_UPSERT_PRICE = """
INSERT INTO etf_prices (isu_cd, bas_dt, close_prc)
VALUES (%s, %s, %s)
ON DUPLICATE KEY UPDATE close_prc = VALUES(close_prc)
"""


def _parse_mysql_url(url: str) -> dict:
    raw = url.replace("mysql+pymysql://", "mysql://")
    p = urlparse(raw)
    return {
        "host": p.hostname, "port": p.port or 3306,
        "user": p.username, "password": p.password,
        "database": p.path.lstrip("/"), "charset": "utf8mb4",
    }


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _parse_date(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


def load_prices_to_mysql(tickers: set[str], start_dt: date, end_dt: date) -> None:
    conn = pymysql.connect(**_parse_mysql_url(MYSQL_URL))
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_PRICE_TABLE)
        conn.commit()

        all_dates    = list(_daterange(start_dt, end_dt))
        is_backfill  = len(all_dates) > 60
        total_rows   = 0
        trading_days = 0

        for d in all_dates:
            date_str = d.strftime("%Y%m%d")
            rows = _fetch_krx(date_str)
            if not rows:
                continue

            trading_days += 1
            batch = []
            for r in rows:
                code = r.get("ISU_CD", "")
                if not code or code not in tickers:
                    continue
                try:
                    price = float(r.get("TDD_CLSPRC", "0").replace(",", ""))
                except ValueError:
                    continue
                if price <= 0:
                    continue
                batch.append((code, d.isoformat(), price))

            if batch:
                with conn.cursor() as cur:
                    cur.executemany(_UPSERT_PRICE, batch)
                conn.commit()
                total_rows += len(batch)
                print(f"  {date_str}: {len(batch)}건 (누적 {trading_days}거래일)")

            if is_backfill:
                time.sleep(0.3)
    finally:
        conn.close()

    print(f"  MySQL 적재 완료: {trading_days}거래일, {total_rows:,}건\n")


# ── 진입점 ────────────────────────────────────────────────────────────────────

async def async_main(start_dt: date, end_dt: date) -> None:
    print("=== [1/2] ETF 종목 선정 ===\n")
    products = select_products(end_dt)
    if not products:
        print("선정된 ETF 없음, 중단")
        return

    tickers = {p["ticker"] for p in products}
    print(f"선정 종목 {len(tickers)}개\n")

    print(f"=== [2/2] MySQL 가격 적재 ({start_dt} ~ {end_dt}) ===\n")
    load_prices_to_mysql(tickers, start_dt, end_dt)

    print("load_etf_prices.py 완료")


def main() -> None:
    parser = argparse.ArgumentParser(description="KRX ETF → MySQL 가격")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--days",  type=int, default=30, help="최근 N일 (기본 30)")
    group.add_argument("--start", type=str, metavar="YYYYMMDD")
    parser.add_argument("--end",  type=str, metavar="YYYYMMDD")
    args = parser.parse_args()

    today = date.today()
    if args.start:
        start_dt = _parse_date(args.start)
        end_dt   = _parse_date(args.end) if args.end else today
    else:
        start_dt = today - timedelta(days=args.days)
        end_dt   = today

    asyncio.run(async_main(start_dt, end_dt))


if __name__ == "__main__":
    main()
