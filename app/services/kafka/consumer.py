import json
import logging

from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.schemas.kafka import KafkaTransactionMessage
from app.services.agent.anomaly import detect_anomaly_agent
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
                alert = await detect_anomaly_agent(transaction)
                if alert:
                    await alert_producer.send_alert(alert)
            except Exception:
                logger.exception("Failed to process Kafka message (offset=%s)", msg.offset)


transaction_consumer = TransactionConsumer()
