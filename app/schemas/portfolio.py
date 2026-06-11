from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel


# ── Shared sub-models ─────────────────────────────────────────────────────────
class CategoryExpenseItem(BaseModel):
    name: str
    expense: int


class InvestmentAssetItem(BaseModel):
    asset_type: str
    account_name: str
    asset_id: UUID
    balance: int

class RebalanceAssetItem(BaseModel):
    asset_type: Literal["CHECKING", "PARKING", "CMA", "DEPOSIT"]
    account_name: str
    asset_id: UUID
    balance: int

# ── POST /portfolio/profile ───────────────────────────────────────────────────
class ProfileRequest(BaseModel):
    user_id: UUID
    category_expense: list[CategoryExpenseItem]
    porti_type: str
    porti_comment: str
    assets_safe: int
    assets_moderate: int # 투자 성향이 중간인 자산 비중
    assets_risky: int


class ProfileResponse(BaseModel):
    created_at: datetime
    expense_comment: str
    invest_comment: str


# ── POST /portfolio/rebalance ─────────────────────────────────────────────────
class RebalanceRequest(BaseModel):
    user_id: UUID
    category_expense: list[CategoryExpenseItem]
    porti_type: str
    porti_comment: str
    assets: list[RebalanceAssetItem]
    fixed_expense: int
    salary: int


class SalaryRebalanceItem(BaseModel):
    asset_id: UUID
    account_purpose: str
    amount: int
    comment: str  # 용도와 금액 산정에 대한 설명


class RebalanceResponse(BaseModel):
    created_at: datetime
    invest_amount: int
    reasoning: str = ""
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
    interest: str
    invest_interests: list[str]
    porti_type: str
    porti_comment: str
    invest_assets: list[InvestmentAssetItem]


class GatheringAccount(BaseModel):
    name: str
    type: str
    institution: str
    interest_rate: float


class PortfolioItem(BaseModel):
    type: str
    name: str
    ratio: int
    ticker: str
    interest_rate: float


class InvestmentPlan(BaseModel):
    title: str
    term: str
    summary: str
    reasoning: str = ""
    gathering_id: Optional[UUID] = None
    gathering_account: Optional[GatheringAccount] = None
    amount: int
    portfolio: list[PortfolioItem]
    expected_rr_pct: float
    investment_months: int
    expected_amount: float
    rr_comment: str


class AssetPortfolioResponse(BaseModel):
    created_at: datetime
    investment_flows: list[InvestmentPlan]
