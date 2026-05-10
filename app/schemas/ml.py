from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class TransactionItem(BaseModel):
    transaction_id: str
    amount: float
    category: str
    timestamp: datetime


class AnomalyRequest(BaseModel):
    user_id: str
    transactions: list[TransactionItem]


class AnomalyResult(BaseModel):
    transaction_id: str
    is_anomaly: bool
    anomaly_score: float


class AnomalyResponse(BaseModel):
    user_id: str
    results: list[AnomalyResult]


class RecommendRequest(BaseModel):
    user_id: str
    age: int
    income_level: Literal["low", "middle", "high"]
    risk_tolerance: str
    current_products: list[str]
    spending_pattern: dict


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    product_type: str
    score: float
    reason: str


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[ProductRecommendation]
