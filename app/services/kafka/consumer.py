import json
import logging
from datetime import date

from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.db.connection import get_pool
from app.schemas.kafka import KafkaTransactionMessage
from app.services.agent.consume_alert import process_user_alert
from app.services.kafka.producer import alert_producer

logger = logging.getLogger(__name__)

_alerted_today: dict[str, date] = {}
_reference_date: date = date.today()
_today_totals: dict[str, int] = {}
_today_by_category: dict[str, dict[str, int]] = {}


def _reset_if_new_day(today: date) -> None:
    global _reference_date
    if today != _reference_date:
        _reference_date = today
        _today_totals.clear()
        _today_by_category.clear()


class TransactionConsumer:
    def __init__(self) -> None:
        self._consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            settings.kafka_input_topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=settings.kafka_consumer_group_id,
            auto_offset_reset="earliest",
        )
        await self._consumer.start()
        logger.info(
            "Kafka consumer started (topic=%s, group=%s)",
            settings.kafka_input_topic,
            settings.kafka_consumer_group_id,
        )

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()
            logger.info("Kafka consumer stopped")

    async def consume(self) -> None:
        if not self._consumer:
            return

        async for msg in self._consumer:
            try:
                data = json.loads(msg.value.decode("utf-8"))
                transaction = KafkaTransactionMessage.model_validate(data)
                await self._handle_transaction(transaction)
            except Exception:
                logger.exception("Failed to process Kafka message (offset=%s)", msg.offset)

    async def _handle_transaction(self, transaction: KafkaTransactionMessage) -> None:
        today = date.today()
        if today.day == 1:
            return

        _reset_if_new_day(today)

        asset_number = transaction.asset_number

        # 오늘 소비 인메모리 누적
        _today_totals[asset_number] = _today_totals.get(asset_number, 0) + transaction.amount
        cat_map = _today_by_category.setdefault(asset_number, {})
        cat_map[transaction.category] = cat_map.get(transaction.category, 0) + transaction.amount

        if _alerted_today.get(asset_number) == today:
            return

        today_by_category = dict(
            sorted(_today_by_category[asset_number].items(), key=lambda x: -x[1])
        )

        pool = await get_pool()
        async with pool.acquire() as conn:
            alert = await process_user_alert(
                conn,
                asset_number,
                today,
                _today_totals[asset_number],
                today_by_category,
            )

        if alert:
            _alerted_today[asset_number] = today
            await alert_producer.send_alert(alert)
            logger.info("소비 추세 알림 전송 (asset_number=%s)", asset_number)


transaction_consumer = TransactionConsumer()
