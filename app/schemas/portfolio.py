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
    assets: list[AssetItem]
    fixed_expense: int
    salary: int


class SalaryRebalanceItem(BaseModel):
    asset_id: UUID
    account_purpose: str
    ratio: int
    comment: str  # 용도와 금액 산정에 대한 설명


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
    interest: str
    invest_interest: list[str]
    porti_type: str
    porti_comment: str
    invest_assets: list[AssetItem]


class GatheringAccount(BaseModel):
    name: str
    type: str
    institution: str
    interest_rate: float


class PortfolioItem(BaseModel):
    type: str
    name: str
    ratio: int
    interest_rate: float
    comment: str


class InvestmentPlan(BaseModel):
    title: str
    term: str
    summary: str
    gathering_id: UUID
    gathering_account: GatheringAccount
    amount: int
    account_comment: str
    portfolio: list[PortfolioItem]
    expected_rr_pct: float
    investment_months: int
    expected_amount: float
    rr_comment: str


class AssetPortfolioResponse(BaseModel):
    created_at: datetime
    investment_flows: list[InvestmentPlan]
