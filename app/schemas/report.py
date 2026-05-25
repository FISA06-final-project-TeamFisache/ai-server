from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class PortfolioDetail(BaseModel):
    asset_type: str
    asset_amount: int
    institution: str
    asset_name: str
    asset_number: str


class PortfolioSnapshot(BaseModel):
    total_asset_amount: int
    portfolio_details: list[PortfolioDetail]


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
    prev_month_portfolio: PortfolioSnapshot
    now_portfolio: PortfolioSnapshot
    transaction_log: list[TransactionLog]


class ReportResponse(BaseModel):
    created_at: datetime
    trend_comment: str
    market_condition: str
    hover_description: str
    performance_status: str
    performance_comment: str
