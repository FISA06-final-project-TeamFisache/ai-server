import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.exceptions import register_exception_handlers
from app.routers.event import router as event_router
from app.routers.portfolio import router as portfolio_router
from app.routers.report import router as report_router
from app.routers.salary import router as salary_router
from app.services.kafka.consumer import transaction_consumer
from app.services.kafka.producer import alert_producer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    consumer_task = None
    kafka_ok = False
    try:
        await alert_producer.start()
        await transaction_consumer.start()
        consumer_task = asyncio.create_task(transaction_consumer.consume())
        kafka_ok = True
    except Exception as e:
        logger.warning("Kafka 연결 실패 — Kafka 없이 실행합니다. (%s)", e)

    yield

    if kafka_ok:
        if consumer_task:
            consumer_task.cancel()
        await transaction_consumer.stop()
        await alert_producer.stop()


app = FastAPI(
    title="FastAPI AI/ML Server",
    description="AI/ML Serving server delegated from Spring Boot",
    version="0.1.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.include_router(event_router)
app.include_router(portfolio_router)
app.include_router(report_router)
app.include_router(salary_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
