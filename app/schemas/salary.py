from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

class CategoryExpenseItem(BaseModel):
    name: str
    expense: int

class PortfolioItem(BaseModel):
    asset_id: UUID
    account_purpose: str
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
    category_expense: list[CategoryExpenseItem]
    portfolio_items: list[PortfolioItem]
    flow_items: list[FlowItem]


class SalaryResponse(BaseModel):
    created_at: datetime
    portfolio_items: list[PortfolioItem]
    flow_items: list[FlowItem]
    rebalance_comment: str
