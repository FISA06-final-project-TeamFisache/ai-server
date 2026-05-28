from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class AssetSnapshot(BaseModel):
    snapshot_at: datetime
    total_amount: int
    savings_amount: int
    invest_amount: int


class TransactionLog(BaseModel):
    asset_number: str
    amount: int
    category: str
    sender_name: str
    transactionAt: str


class ReportRequest(BaseModel):
    user_id: UUID
    year: int
    month: int
    title: str
    deadline: datetime
    target_amount: str
    asset_snapshots: list[AssetSnapshot]
    transaction_log: list[TransactionLog]


class ReportResponse(BaseModel):
    created_at: datetime
    trend_comment: str
    event_comment: str
    market_condition: str
    hover_description: str
    guideline: str
    performance_status: str
    performance_comment: str
