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


async def _fetch_cumulative(
    conn: asyncpg.Connection,
    asset_number: str,
    start: date,
    end_exclusive: date,
) -> int:
    tt = settings.db_transaction_table
    at = settings.db_asset_table
    asset_pk = settings.db_asset_pk_col
    asset_num = settings.db_asset_number_col
    fk = settings.db_transaction_asset_fk_col
    amount_col = settings.db_amount_col
    date_col = settings.db_date_col
    val = await conn.fetchval(
        f"""
        SELECT COALESCE(SUM(t.{amount_col}), 0)
        FROM {tt} t
        JOIN {at} a ON a.{asset_pk} = t.{fk}
        WHERE a.{asset_num} = $1
          AND t.{date_col} >= $2
          AND t.{date_col} < $3
        """,
        asset_number,
        start,
        end_exclusive,
    )
    return int(val)


async def _generate_message(
    this_total: int,
    last_total: int,
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
            f"차이: +{this_total - last_total:,}원\n\n"
            f"오늘 카테고리별 소비:\n{category_lines or '  - (없음)'}\n\n"
            "위 내용을 바탕으로 소비 추세가 가팔라지고 있음을 알리는 짧은 알림 메시지를 작성해주세요. "
            "오늘 가장 많이 소비한 카테고리를 언급하고, 소비 관리를 권고해 주세요. 2~3문장으로 작성하세요."
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
    today_total: int,
    today_by_category: dict[str, int],
) -> KafkaAnomalyAlert | None:
    this_month_start = today.replace(day=1)
    last_month_today = _same_day_last_month(today)
    last_month_start = last_month_today.replace(day=1)

    cached = daily_cache.get_db_cache(asset_number, today)
    if cached:
        db_this_total, last_total = cached
    else:
        db_this_total = await _fetch_cumulative(conn, asset_number, this_month_start, today)
        last_total = await _fetch_cumulative(conn, asset_number, last_month_start, last_month_today + timedelta(days=1))
        daily_cache.set_db_cache(asset_number, db_this_total, last_total, today)

    this_total = db_this_total + today_total

    if last_total == 0 or this_total <= last_total:
        return None

    try:
        message = await _generate_message(this_total, last_total, today_by_category)
    except Exception:
        logger.exception("LLM 호출 실패 (asset_number=%s)", asset_number)
        return None

    return KafkaAnomalyAlert(
        asset_number=asset_number,
        content=message,
        created_at=datetime.now(timezone.utc),
    )
