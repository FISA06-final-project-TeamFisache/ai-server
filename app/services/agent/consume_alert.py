from __future__ import annotations

import asyncio
import calendar
import logging
from datetime import date, datetime, timedelta, timezone

import asyncpg
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.schemas.kafka import KafkaAnomalyAlert
from app.services.daily_cache import daily_cache

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))

_EXCLUDED_CATEGORIES = frozenset({"급여", "통신", "공과금", "보험료"})

_SYSTEM_PROMPT = (
    "당신은 개인 소비 패턴을 분석하는 금융 AI 어시스턴트입니다. "
    "친절하고 간결하게 소비 추세를 설명해주세요."
)


def _same_day_last_month(d: date) -> date:
    if d.month == 1:
        year, month = d.year - 1, 12
    else:
        year, month = d.year, d.month - 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return d.replace(year=year, month=month, day=day)


async def _fetch_by_category(
    conn: asyncpg.Connection,
    asset_number: str,
    start: date,
    end_exclusive: date,
) -> dict[str, int]:
    tt = settings.db_transaction_table
    at = settings.db_asset_table
    asset_pk = settings.db_asset_pk_col
    asset_num = settings.db_asset_number_col
    fk = settings.db_transaction_asset_fk_col
    amount_col = settings.db_amount_col
    category_col = settings.db_category_col
    date_col = settings.db_date_col
    rows = await conn.fetch(
        f"""
        SELECT t.{category_col} AS category, SUM(t.{amount_col}) AS total
        FROM {tt} t
        JOIN {at} a ON a.{asset_pk} = t.{fk}
        WHERE a.{asset_num} = $1
          AND t.{date_col} >= $2
          AND t.{date_col} < $3
          AND t.{category_col} NOT IN ('급여', '통신', '공과금', '보험료')
        GROUP BY t.{category_col}
        """,
        asset_number,
        start,
        end_exclusive,
    )
    return {row["category"]: int(row["total"]) for row in rows}


async def _generate_extra_comment(
    this_total: int,
    last_total: int,
    top_category: str,
    today_by_category: dict[str, int],
) -> str:
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
    )

    category_lines = "\n".join(
        f"  - {cat}: {amt:,}원" for cat, amt in today_by_category.items()
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"이번 달 누적 소비(오늘 포함): {this_total:,}원\n"
            f"전달 같은 기간 누적 소비: {last_total:,}원\n"
            f"차이: +{this_total - last_total:,}원\n"
            f"주요 초과 카테고리: {top_category}\n\n"
            f"오늘 카테고리별 소비:\n{category_lines or '  - (없음)'}\n\n"
            "소비 관리를 권고하는 짧은 추가 멘트를 1~2문장으로 작성해주세요."
        )),
    ]

    response = await asyncio.wait_for(
        llm.ainvoke(messages),
        timeout=settings.agent_timeout_consume_alert,
    )
    return response.content


async def process_user_alert(
    conn: asyncpg.Connection,
    asset_number: str,
    today: date,
    today_by_category: dict[str, int],
) -> KafkaAnomalyAlert | None:
    this_month_start = today.replace(day=1)
    last_month_today = _same_day_last_month(today)
    last_month_start = last_month_today.replace(day=1)

    today_filtered = {
        cat: amt
        for cat, amt in today_by_category.items()
        if cat not in _EXCLUDED_CATEGORIES
    }

    cached = daily_cache.get_db_cache(asset_number, today)
    if cached:
        db_this_by_cat, last_by_cat = cached
    else:
        db_this_by_cat = await _fetch_by_category(conn, asset_number, this_month_start, today)
        last_by_cat = await _fetch_by_category(
            conn, asset_number, last_month_start, last_month_today + timedelta(days=1)
        )
        daily_cache.set_db_cache(asset_number, db_this_by_cat, last_by_cat, today)

    this_by_cat = dict(db_this_by_cat)
    for cat, amt in today_filtered.items():
        this_by_cat[cat] = this_by_cat.get(cat, 0) + amt

    this_total = sum(this_by_cat.values())
    last_total = sum(last_by_cat.values())

    if last_total == 0 or this_total <= last_total:
        return None

    top_category = max(
        this_by_cat.keys(),
        key=lambda cat: this_by_cat[cat] - last_by_cat.get(cat, 0),
    )

    now_kst = datetime.now(_KST)
    prefix = (
        f"{now_kst.day}일 {now_kst.hour}시에 지난 달 소비 추세를 벗어났어요! "
        f"저번 달 대비 {top_category} 지출이 높아요."
    )

    try:
        extra = await _generate_extra_comment(this_total, last_total, top_category, today_filtered)
    except Exception:
        logger.exception("LLM 호출 실패 (asset_number=%s)", asset_number)
        return None

    return KafkaAnomalyAlert(
        asset_number=asset_number,
        content=f"{prefix} {extra}",
        created_at=datetime.now(timezone.utc),
    )
