"""
ETF 일간 종가 데이터를 KRX API에서 수집해 MySQL에 적재하는 스크립트.
load_products.py가 volatility 계산에 사용한다.

사용법:
  # MVP: 최근 30일 (기본값)
  python scripts/load_etf_prices.py

  # 최근 N일
  python scripts/load_etf_prices.py --days 60

  # 밤사이 3년치 backfill
  python scripts/load_etf_prices.py --start 20230609 --end 20260609

실행 전 준비:
  pip install requests pymysql python-dotenv
"""

import argparse
import os
import time
from datetime import date, timedelta
from urllib.parse import urlparse

import pymysql
import requests
from dotenv import load_dotenv

load_dotenv()

KRX_API_KEY = os.getenv("KRX_API_KEY", "")
MYSQL_URL = os.getenv("MYSQL_URL", "")

KRX_ETF_URL = "https://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS etf_prices (
    isu_cd    VARCHAR(20)    NOT NULL,
    bas_dt    DATE           NOT NULL,
    close_prc DECIMAL(15, 2) NOT NULL,
    PRIMARY KEY (isu_cd, bas_dt)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

UPSERT_SQL = """
INSERT INTO etf_prices (isu_cd, bas_dt, close_prc)
VALUES (%s, %s, %s)
ON DUPLICATE KEY UPDATE close_prc = VALUES(close_prc)
"""


def _parse_mysql_url(url: str) -> dict:
    raw = url.replace("mysql+pymysql://", "mysql://")
    p = urlparse(raw)
    return {
        "host": p.hostname,
        "port": p.port or 3306,
        "user": p.username,
        "password": p.password,
        "database": p.path.lstrip("/"),
        "charset": "utf8mb4",
    }


def _fetch_active_tickers() -> set[str]:
    """품질 필터를 통과한 ETF 종목코드 집합을 반환 (가장 최근 거래일 기준).
    레버리지·인버스 제외, 일평균 거래대금 1억 미만 제외.
    load_products.py가 어떤 종목을 선택하더라도 가격 이력이 항상 존재하도록
    top-100 캡 없이 전체를 저장한다."""
    today = date.today()
    for delta in range(0, 14):
        d = (today - timedelta(days=delta)).strftime("%Y%m%d")
        rows = _fetch_krx(d)
        if not rows:
            continue
        active = set()
        for r in rows:
            code = r.get("ISU_CD", "")
            name = r.get("ISU_NM", "")
            if not code or not name:
                continue
            if "레버리지" in name or "인버스" in name:
                continue
            try:
                trd_val = int(r.get("ACC_TRDVAL", "0").replace(",", ""))
            except (ValueError, AttributeError):
                trd_val = 0
            if trd_val < 100_000_000:
                continue
            active.add(code)
        print(f"  기준일 {d}: 품질 필터 통과 {len(active)}개 ETF 파악")
        return active
    return set()


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


def _daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _parse_date(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


def main() -> None:
    parser = argparse.ArgumentParser(description="KRX ETF 일간 가격 → MySQL 적재")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--days", type=int, default=30, help="최근 N일 (기본 30)")
    group.add_argument("--start", type=str, metavar="YYYYMMDD", help="backfill 시작일")
    parser.add_argument("--end", type=str, metavar="YYYYMMDD", help="backfill 종료일 (--start 필요)")
    args = parser.parse_args()

    today = date.today()

    if args.start:
        start_dt = _parse_date(args.start)
        end_dt = _parse_date(args.end) if args.end else today
    else:
        start_dt = today - timedelta(days=args.days)
        end_dt = today

    print(f"=== ETF 가격 적재: {start_dt} ~ {end_dt} ===\n")

    print("현재 상장 ETF 목록 파악 중...")
    active_tickers = _fetch_active_tickers()
    if not active_tickers:
        print("  활성 ETF 목록 조회 실패, 중단")
        return
    print()

    conn = pymysql.connect(**_parse_mysql_url(MYSQL_URL))
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        print("  테이블 준비 완료\n")

        all_dates = list(_daterange(start_dt, end_dt))
        is_backfill = len(all_dates) > 60
        total_rows = 0
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
                if not code or code not in active_tickers:
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
                    cur.executemany(UPSERT_SQL, batch)
                conn.commit()
                total_rows += len(batch)
                print(f"  {date_str}: {len(batch)}건 적재 (누적 {trading_days}거래일)")

            # backfill 시 API 부하 경감
            if is_backfill:
                time.sleep(0.3)

    finally:
        conn.close()

    print(f"\n총 {trading_days}거래일, {total_rows:,}건 적재 완료")


if __name__ == "__main__":
    main()
