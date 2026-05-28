from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


# ── Shared sub-models ─────────────────────────────────────────────────────────
class CategoryExpenseItem(BaseModel):
    name: str
    expense: int


class AssetItem(BaseModel):
    asset_type: str
    account_name: str
    asset_id: UUID
    balance: str


# ── POST /portfolio/profile ───────────────────────────────────────────────────
class ProfileRequest(BaseModel):
    user_id: UUID
    category_expense: list[CategoryExpenseItem]
    porti_type: str
    porti_comment: str
    assets: list[AssetItem]


class ProfileResponse(BaseModel):
    created_at: datetime
    expense_comment: str
    invest_comment: str
    savings_comment: str


# ── POST /portfolio/rebalance ─────────────────────────────────────────────────
class RebalanceRequest(BaseModel):
    user_id: UUID
    category_expense: list[CategoryExpenseItem]
    porti_type: str
    porti_comment: str
    assets: list[AssetItem]
    fixed_expense: int
    salary: int


class SalaryRebalanceItem(BaseModel):
    asset_id: UUID
    category: str
    amount: int


class RebalanceResponse(BaseModel):
    created_at: datetime
    invest_amount: int
    salary_rebalance: list[SalaryRebalanceItem]


# ── POST /portfolio/asset-portfolio ──────────────────────────────────────────

class ProductItem(BaseModel):
    product_type: str
    institution: str
    name: str
    interest_rate: float
    description: str


class AssetPortfolioRequest(BaseModel):
    user_id: UUID
    invest_amount: int
    porti_type: str
    porti_comment: str
    invest_assets: list[AssetItem]
    products: list[ProductItem]


class FundingSource(BaseModel):
    asset_id: UUID
    amount: int


class PortfolioItem(BaseModel):
    name: str
    ratio: int


class InvestmentPlan(BaseModel):
    title: str
    term: str
    summary: str
    funding_sources: list[FundingSource]
    gathering_account: UUID
    portfolio: list[PortfolioItem]


class AssetPortfolioResponse(BaseModel):
    created_at: datetime
    investment_flows: list[InvestmentPlan]
