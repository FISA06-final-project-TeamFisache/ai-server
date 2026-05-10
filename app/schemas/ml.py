from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ApprovedTransaction(BaseModel):
    approved_dtime: datetime
    approved_amt: float
    approved_type: str
    merchant_name: str
    is_canceled: bool


class AnomalyRequest(BaseModel):
    user_id: str
    approved_list: list[ApprovedTransaction]


class AnomalyResponse(BaseModel):
    title: str
    content: str


class Transaction(BaseModel):
    trans_dtime: str
    trans_type: str
    trans_class: str
    trans_amt: float
    balance_amt: float
    trans_memo: str


class RecommendRequest(BaseModel):
    user_id: str
    trans_list: list[Transaction]


class Product(BaseModel):
    id: str
    product_type: str
    institution: str
    name: str
    description: str


class RecommendResponse(BaseModel):
    ai_comment: str
    products: list[Product]
