import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

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
    yield


app = FastAPI(
    title="FastAPI AI/ML Server",
    description="AI/ML Serving server delegated from Spring Boot",
    version="0.1.0",
    lifespan=lifespan,
)

# 허용할 프론트 origin 목록(콤마 구분). 배포 시 CloudFront 도메인을 env로 주입.
_allowed_origins = os.environ.get(
    "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
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
