from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class PortfolioDetailItem(BaseModel):
    item_name: str
    item_amount: int


class PortfolioDetail(BaseModel):
    asset_type: str
    asset_amount: int
    items: list[PortfolioDetailItem]


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
    title: str
    deadline: datetime
    target_amount: str
    prev_month_portfolio: PortfolioSnapshot
    now_portfolio: PortfolioSnapshot
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
