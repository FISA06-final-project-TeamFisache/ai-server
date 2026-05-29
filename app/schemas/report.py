from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class AssetSnapshot(BaseModel):
    snapshot_at: datetime
    total_amount: int
    savings_amount: int
    invest_amount: int


class TransactionLog(BaseModel):
    amount: int
    category: str
    sender_name: str
    transaction_at: datetime


class HoverDescription(BaseModel):
    category: str
    content: str


class ReportRequest(BaseModel):
    user_id: UUID
    year: int
    month: int
    title: str
    deadline: datetime
    target_amount: int
    goal_progress: int
    asset_snapshots: list[AssetSnapshot]
    transaction_log: list[TransactionLog]


class ReportResponse(BaseModel):
    created_at: datetime
    trend_comment: str
    event_comment: str
    market_condition: str
    hover_description: list[HoverDescription]
    guideline: str
    performance_status: str
    performance_comment: str
