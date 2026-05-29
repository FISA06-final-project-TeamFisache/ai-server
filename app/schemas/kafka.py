from datetime import datetime

from pydantic import BaseModel


class KafkaTransactionMessage(BaseModel):
    asset_number: str
    amount: int
    category: str
    sender_name: str
    transactionAt: datetime


class KafkaAnomalyAlert(BaseModel):
    asset_number: str
    content: str
    created_at: datetime
