import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.schemas.ml import ProductRecommendation, RecommendRequest, RecommendResponse
from app.services.ml.model_loader import get_model

_executor = ThreadPoolExecutor(max_workers=4)

_INCOME_MAP = {"low": 0, "middle": 1, "high": 2}
_RISK_MAP = {"low": 0, "moderate": 1, "high": 2}

_PRODUCT_CATALOG = [
    {"product_id": "P001", "product_name": "CMA 통장", "product_type": "savings"},
    {"product_id": "P002", "product_name": "국내 주식형 펀드", "product_type": "fund"},
    {"product_id": "P003", "product_name": "채권형 ETF", "product_type": "etf"},
    {"product_id": "P004", "product_name": "달러 예금", "product_type": "deposit"},
    {"product_id": "P005", "product_name": "글로벌 인덱스 펀드", "product_type": "fund"},
]


def _predict(request: RecommendRequest) -> list[ProductRecommendation]:
    model = get_model("recommend")
    features = np.array([[
        request.age,
        _INCOME_MAP.get(request.income_level, 1),
        _RISK_MAP.get(request.risk_tolerance, 1),
    ]])
    scores = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else model.predict(features)[0]

    recommendations = []
    for i, product in enumerate(_PRODUCT_CATALOG):
        if product["product_id"] in request.current_products:
            continue
        score = float(scores[i % len(scores)])
        recommendations.append(
            ProductRecommendation(
                product_id=product["product_id"],
                product_name=product["product_name"],
                product_type=product["product_type"],
                score=round(score, 4),
                reason="사용자 프로필 기반 추천",
            )
        )
    recommendations.sort(key=lambda x: x.score, reverse=True)
    return recommendations[:3]


async def recommend_products(request: RecommendRequest) -> RecommendResponse:
    loop = asyncio.get_event_loop()
    recommendations = await loop.run_in_executor(_executor, _predict, request)
    return RecommendResponse(user_id=request.user_id, recommendations=recommendations)
