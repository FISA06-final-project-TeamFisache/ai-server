import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.schemas.ml import AnomalyRequest, AnomalyResponse, AnomalyResult
from app.services.ml.model_loader import get_model

_executor = ThreadPoolExecutor(max_workers=4)


def _predict(request: AnomalyRequest) -> list[AnomalyResult]:
    model = get_model("anomaly")
    features = np.array([[t.amount] for t in request.transactions])
    scores = model.decision_function(features)
    predictions = model.predict(features)
    return [
        AnomalyResult(
            transaction_id=t.transaction_id,
            is_anomaly=bool(pred == -1),
            anomaly_score=float(round(1 - (score - scores.min()) / (scores.max() - scores.min() + 1e-9), 4)),
        )
        for t, score, pred in zip(request.transactions, scores, predictions)
    ]


async def detect_anomaly(request: AnomalyRequest) -> AnomalyResponse:
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(_executor, _predict, request)
    return AnomalyResponse(user_id=request.user_id, results=results)
