import logging

from aiokafka import AIOKafkaProducer

from app.core.config import settings
from app.schemas.kafka import KafkaAnomalyAlert

logger = logging.getLogger(__name__)


class AnomalyAlertProducer:
    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
        )
        await self._producer.start()
        logger.info("Kafka producer started (topic=%s)", settings.kafka_output_topic)

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka producer stopped")

    async def send_alert(self, alert: KafkaAnomalyAlert) -> None:
        if not self._producer:
            logger.error("Producer is not started")
            return
        payload = alert.model_dump_json().encode("utf-8")
        await self._producer.send_and_wait(settings.kafka_output_topic, value=payload)
        logger.info("Anomaly alert sent (asset_number=%s)", alert.asset_number)


alert_producer = AnomalyAlertProducer()
