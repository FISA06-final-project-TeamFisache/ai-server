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
class ProfileRequest(BaseModel):
    user_id: UUID


class ProfileResponse(BaseModel):
    created_at: datetime
    porti_comment: str
    expense_comment: str
    invest_comment: str
    savings_comment: str
    audit_log: AuditLog


# ── /agent/portfolio/recommend & /agent/event/input ───────────────────────────
class SalaryRebalanceItem(BaseModel):
    asset_number: str
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


class RecommendResponse(BaseModel):
    created_at: datetime
    salary_rebalance: list[SalaryRebalanceItem]
    portfolio_recommend: PortfolioRecommendResult
    audit_log: AuditLog


class EventInputRequest(BaseModel):
    user_id: UUID
    user_input: str


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
    monthly_change: str
    portfolios: Portfolios
    portfolio_comment: str
    expense_categories: list[ExpenseCategory]
    expense_analysis: str
    recommended_rebalance_ratio: RecommendedRebalanceRatio
    next_month_guideline: str
