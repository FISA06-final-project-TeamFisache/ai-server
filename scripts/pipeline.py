"""
ETF 데이터 파이프라인

  STEP 1 — load_etf_prices: KRX 수집 → MySQL 가격 + PostgreSQL products upsert
  STEP 2 — load_products  : MySQL → interest_rate · volatility → PostgreSQL 갱신

사용법:
  python scripts/pipeline.py              # 최근 30일
  python scripts/pipeline.py --days 60
  python scripts/pipeline.py --start 20230609 --end 20260609
"""

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import load_etf_prices
import load_products


async def run(start_dt: date, end_dt: date) -> None:
    print("=" * 60)
    print("STEP 1 / 2  —  load_etf_prices")
    print("=" * 60)
    await load_etf_prices.async_main(start_dt, end_dt)

    print()
    print("=" * 60)
    print("STEP 2 / 2  —  load_products")
    print("=" * 60)
    await load_products.async_main()

    print()
    print("파이프라인 완료")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF 데이터 파이프라인")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--days",  type=int, default=30, help="최근 N일 (기본 30)")
    group.add_argument("--start", type=str, metavar="YYYYMMDD")
    parser.add_argument("--end",  type=str, metavar="YYYYMMDD")
    args = parser.parse_args()

    today = date.today()
    if args.start:
        start_dt = load_etf_prices._parse_date(args.start)
        end_dt   = load_etf_prices._parse_date(args.end) if args.end else today
    else:
        start_dt = today - timedelta(days=args.days)
        end_dt   = today

    asyncio.run(run(start_dt, end_dt))


if __name__ == "__main__":
    main()
