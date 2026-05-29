import json
import logging
from datetime import date

from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.db.connection import get_pool
from app.schemas.kafka import KafkaTransactionMessage
from app.services.agent.consume_alert import process_user_alert
from app.services.daily_cache import daily_cache
from app.services.kafka.producer import alert_producer

logger = logging.getLogger(__name__)


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

        daily_cache.reset_if_new_day(today)
        daily_cache.accumulate(transaction.asset_number, transaction.amount, transaction.category)

        asset_number = transaction.asset_number
        if daily_cache.is_alerted(asset_number, today):
            return

        pool = await get_pool()
        async with pool.acquire() as conn:
            alert = await process_user_alert(
                conn,
                asset_number,
                today,
                daily_cache.get_today_total(asset_number),
                daily_cache.get_today_by_category(asset_number),
            )

        if alert:
            daily_cache.mark_alerted(asset_number, today)
            await alert_producer.send_alert(alert)
            logger.info("소비 추세 알림 전송 (asset_number=%s)", asset_number)


transaction_consumer = TransactionConsumer()
