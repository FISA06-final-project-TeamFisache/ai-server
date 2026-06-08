import logging
import os
from contextlib import asynccontextmanager

import socket

# 기존의 통신 함수 저장
original_getaddrinfo = socket.getaddrinfo

# localhost로 요청이 오면 127.0.0.1로 바꿔서 보내도록 가로채기
def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host == 'localhost':
        host = '127.0.0.1'
    return original_getaddrinfo(host, port, family, type, proto, flags)

# 바꿔치기 적용
socket.getaddrinfo = patched_getaddrinfo

from dotenv import load_dotenv

load_dotenv()

from app.kafka.producer import start_producer, stop_producer

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prometheus_fastapi_instrumentator import Instrumentator

from app.core.exceptions import register_exception_handlers
from app.db.connection import close_pool
from app.routers.mini_challenge import router as mini_challenge_router
from app.routers.portfolio import router as portfolio_router
from app.routers.report import router as report_router
from app.routers.consultant import router as consultant_router
from app.routers.salary import router as salary_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    await start_producer(kafka_servers)
    yield
    await stop_producer()


app = FastAPI(
    title="FastAPI AI/ML Server",
    description="AI/ML Serving server delegated from Spring Boot",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
Instrumentator().instrument(app).expose(app)
app.include_router(mini_challenge_router)
app.include_router(portfolio_router)
app.include_router(report_router)
app.include_router(consultant_router)
app.include_router(salary_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
