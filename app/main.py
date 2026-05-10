from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.exceptions import register_exception_handlers
from app.routers.agent import router as agent_router
from app.routers.ml import router as ml_router
from app.services.ml.model_loader import load_all_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield


app = FastAPI(
    title="FastAPI AI/ML Server",
    description="AI/ML Serving server delegated from Spring Boot",
    version="0.1.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.include_router(agent_router)
app.include_router(ml_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
