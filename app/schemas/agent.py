from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


# ── Audit Log ─────────────────────────────────────────────────────────────────
class AuditLog(BaseModel):
    log_id: str
    user_id: str
    endpoint: str
    requested_at: datetime


# ── /agent/portfolio/profile ──────────────────────────────────────────────────
class CategoryExpenseItem(BaseModel):
    name: str
    expense: int


class AssetItem(BaseModel):
    asset_type: str
    asset_number: str = ""
    balance: str = "0"


class ProfileRequest(BaseModel):
    user_id: UUID
    porti_type: str = ""
    porti_comment: str = ""
    category_expense: list[CategoryExpenseItem] = []
    assets: list[AssetItem] = []


class ProfileResponse(BaseModel):
    created_at: datetime
    porti_comment: str
    expense_comment: str
    invest_comment: str
    savings_comment: str
    audit_log: AuditLog


# ── /agent/portfolio/recommend & /agent/event/input ───────────────────────────
class SalaryRebalanceItem(BaseModel):
    asset_number: str = ""
    category: str
    ratio: int


class PortfolioRecommendResult(BaseModel):
    stock_ratio: int
    stock_recs: list[str]
    bond_ratio: int
    bond_recs: list[str]
    cash_ratio: int
    cash_recs: list[str]


class RecommendRequest(BaseModel):
    user_id: UUID
    porti_type: str = ""
    porti_comment: str = ""
    category_expense: list[CategoryExpenseItem] = []
    assets: list[AssetItem] = []
    fixed_expense: int = 0
    salary: int = 0


class RecommendResponse(BaseModel):
    created_at: datetime
    salary_rebalance: list[SalaryRebalanceItem]
    portfolio_recommend: PortfolioRecommendResult
    audit_log: AuditLog


class EventInputRequest(BaseModel):
    user_id: UUID
    user_input: str
    target_amount: int = 0
    deadline_months: int = 12
    current_portfolio: dict = {"stock_ratio": 50, "bond_ratio": 30, "cash_ratio": 20}


class EventInputResponse(BaseModel):
    created_at: datetime
    salary_rebalance: list[SalaryRebalanceItem]
    portfolio_recommend: PortfolioRecommendResult
    audit_log: AuditLog


# ── /agent/report ─────────────────────────────────────────────────────────────
class ReportRequest(BaseModel):
    user_id: UUID
    year: int
    month: int


class Portfolios(BaseModel):
    stock_change: float
    bond_change: float
    cash_change: float


class ExpenseCategory(BaseModel):
    category: str
    value: int


class RecommendedRebalanceRatio(BaseModel):
    stock_ratio: int
    bond_ratio: int
    cash_ratio: int


class ReportResponse(BaseModel):
    total_income: int
    total_expense: int
    surplus: int
    monthly_change: str
    portfolios: Portfolios
    portfolio_comment: str
    expense_categories: list[ExpenseCategory]
    expense_analysis: str
    recommended_rebalance_ratio: RecommendedRebalanceRatio
    next_month_guideline: str


# ── /agent/goal/portfolio ─────────────────────────────────────────────────────
class GoalPortfolioRequest(BaseModel):
    user_id: str
    deadline: str
    initial_capital: int
    monthly_seed: int
    target_amount: int


class PortfolioComposition(BaseModel):
    cash_pct: float
    stocks_etf_pct: float
    bonds_pct: float


class GoalPortfolioResponse(BaseModel):
    portfolio_detail: str
    portfolio_composition: PortfolioComposition


# ── /agent/goal/analysis ──────────────────────────────────────────────────────
class PortfolioUser(BaseModel):
    cash_ratio: int
    stock_ratio: int
    bond_ratio: int


class GoalAnalysisRequest(BaseModel):
    user_id: str
    portfolio_user: PortfolioUser


class GoalAnalysisResponse(BaseModel):
    analysis_report: str
    summary: str
