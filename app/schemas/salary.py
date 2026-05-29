from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class PortfolioItem(BaseModel):
    asset_id: UUID
    category: str
    amount: int

class FlowItem(BaseModel):
    title: str
    term: str
    summary: str
    asset_id: UUID
    amount: int


class SalaryRequest(BaseModel):
    user_id: UUID
    salary_diff: int
    portfolio_items: list[PortfolioItem]
    flow_items: list[FlowItem]


class SalaryResponse(BaseModel):
    created_at: datetime
    portfolio_items: list[PortfolioItem]
    flow_items: list[FlowItem]
    rebalance_comment: str
