import asyncio

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.ml import AnomalyRequest, AnomalyResponse, RecommendRequest, RecommendResponse
from app.services.ml.anomaly import detect_anomaly
from app.services.ml.recommend import recommend_products

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post("/anomaly", response_model=AnomalyResponse)
async def anomaly_detection(req: AnomalyRequest) -> AnomalyResponse:
    try:
        return await asyncio.wait_for(detect_anomaly(req), timeout=settings.ml_timeout_anomaly)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Anomaly detection timed out")


@router.post("/recommend", response_model=RecommendResponse)
async def recommendation(req: RecommendRequest) -> RecommendResponse:
    try:
        return await asyncio.wait_for(recommend_products(req), timeout=settings.ml_timeout_recommend)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Recommendation timed out")
