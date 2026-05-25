from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


# ── Shared sub-models ─────────────────────────────────────────────────────────
class SalaryRebalanceItem(BaseModel):
    asset_number: str
    category: str
    ratio: int


class InvestAssetItem(BaseModel):
    asset_type: str
    balance: int


class ProductItem(BaseModel):
    product_type: str
    institution: str
    name: str
    interest_rate: float
    description: str


# ── POST /event/rebalance ─────────────────────────────────────────────────────
class RebalanceInfo(BaseModel):
    salary: int
    invest_amount: int
    salary_rebalance: list[SalaryRebalanceItem]


class EventRebalanceRequest(BaseModel):
    user_id: UUID
    user_input: str
    porti_type: str
    porti_comment: str
    rebalance: RebalanceInfo


class EventRebalanceResponse(BaseModel):
    created_at: datetime
    salary_rebalance: list[SalaryRebalanceItem]
    rebalance_comment: str


# ── POST /event/asset-portfolio ───────────────────────────────────────────────
class EventAssetPortfolioRequest(BaseModel):
    user_id: UUID
    user_input: str
    invest_amount: int
    porti_type: str
    porti_comment: str
    invest_assets: list[InvestAssetItem]
    products: list[ProductItem]


class FundingSource(BaseModel):
    account_name: str
    amount: int


class PortfolioItem(BaseModel):
    name: str
    ratio: int


class InvestmentPlan(BaseModel):
    funding_sources: list[FundingSource]
    gathering_account: str
    portfolio: list[PortfolioItem]
    description: str


class EventAssetPortfolioResponse(BaseModel):
    created_at: datetime
    investment_plans: list[InvestmentPlan]
