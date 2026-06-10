from __future__ import annotations

import json
import logging

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_producer: AIOKafkaProducer | None = None
TOPIC = "service-logs"


async def start_producer(bootstrap_servers: str) -> None:
    global _producer
    _producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        retry_backoff_ms=200,
        request_timeout_ms=5000,
    )
    await _producer.start()
    logger.info("Kafka producer started (bootstrap=%s)", bootstrap_servers)


async def stop_producer() -> None:
    global _producer
    if _producer:
        await _producer.stop()
        logger.info("Kafka producer stopped")


async def send_log(event: BaseModel) -> None:
    if _producer is None:
        logger.warning("Kafka producer not initialized — skipping log")
        return
    try:
        await _producer.send(TOPIC, value=event.model_dump())
    except Exception as e:
        logger.error("Kafka send failed: %s", e)
