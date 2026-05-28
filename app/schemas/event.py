from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


# ── Shared sub-models ─────────────────────────────────────────────────────────
class SalaryRebalanceItem(BaseModel):
    account_name: str
    asset_id: UUID
    category: str
    amount: int


class InvestAssetItem(BaseModel):
    asset_type: str
    account_name: str
    asset_id: UUID
    balance: int


class ProductItem(BaseModel):
    product_type: str
    institution: str
    name: str
    interest_rate: float
    description: str


class FundingSource(BaseModel):
    account_name: str
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


# ── POST /event/input ─────────────────────────────────────────────────────────
class EventInputRequest(BaseModel):
    user_id: UUID
    user_input: str


class EventInputResponse(BaseModel):
    created_at: datetime
    title: str
    target_amount: str
    deadline: datetime


# ── POST /event/rebalance ─────────────────────────────────────────────────────
class RebalanceInfo(BaseModel):
    salary: int
    invest_amount: int
    salary_rebalance: list[SalaryRebalanceItem]


class EventRebalanceRequest(BaseModel):
    user_id: UUID
    title: str
    target_amount: str
    deadline: datetime
    porti_type: str
    porti_comment: str
    rebalance: RebalanceInfo


class EventRebalanceResponse(BaseModel):
    created_at: datetime
    invest_amount: int
    salary_rebalance: list[SalaryRebalanceItem]
    rebalance_comment: str


# ── POST /event/asset-portfolio ───────────────────────────────────────────────
class EventAssetPortfolioRequest(BaseModel):
    user_id: UUID
    title: str
    target_amount: str
    deadline: datetime
    invest_amount: int
    porti_type: str
    porti_comment: str
    invest_assets: list[InvestAssetItem]
    products: list[ProductItem]
    investment_flows: list[InvestmentPlan]


class EventAssetPortfolioResponse(BaseModel):
    created_at: datetime
    investment_flows: list[InvestmentPlan]
