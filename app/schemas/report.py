from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class MiniChallenges(BaseModel):
    title: str
    description: str
    status: str
    challenge_type: str
    target: int
    started_at: datetime
    completed_at: datetime | None

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
    mini_challenges: list[MiniChallenges]
    asset_snapshots: list[AssetSnapshot]
    transaction_log: list[TransactionLog]


class ReportResponse(BaseModel):
    created_at: datetime
    trend_comment: str
    challenge_comment: str
    market_condition: str
    hover_description: list[HoverDescription]
    guideline: str
