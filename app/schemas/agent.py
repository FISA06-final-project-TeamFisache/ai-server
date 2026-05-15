from datetime import date
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


# ── Report ──────────────────────────────────────────────────────
class PortfolioComposition(BaseModel):
    cash_pct: int
    stocks_etf_pct: int
    bonds_pct: int


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


# ── /agent/goal/portfolio ────────────────────────────────────────────────────

class PortfolioRequest(BaseModel):
    user_id: str
    deadline: date
    initial_capital: int
    monthly_seed: int
    target_amount: int


class PortfolioResponse(BaseModel):
    portfolio_detail: str
    portfolio_composition: PortfolioComposition


# ── /agent/goal/seed-money ──────────────────────────────────────────────────

class SeedMoneyRequest(BaseModel):
    user_id: str
    deadline: date
    target_amount: int
    PorTI: str


class SeedMoneyResponse(BaseModel):
    pass


# ── /agent/goal/wedding ─────────────────────────────────────────────────────

class WeddingScale(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"


class WeddingRequest(BaseModel):
    user_id: str
    deadline: date
    wedding_region: str
    wedding_month: int
    honeymoon_scale: WeddingScale
    sdrme_scale: WeddingScale


class WeddingBudget(BaseModel):
    venue: int
    honeymoon: int
    sdrme: int
    total: int


class WeddingResponse(BaseModel):
    budget: WeddingBudget
    reasoning: str


# ── /agent/goal/travel ──────────────────────────────────────────────────────

class TravelStyle(str, Enum):
    budget = "budget"
    luxury = "luxury"


class TravelRequest(BaseModel):
    user_id: str
    deadline: date
    maximum_budget: int
    destination: str
    travel_style: TravelStyle
    travel_days: int
    departure_month: str


class TravelBudget(BaseModel):
    accommodation: int
    flight: int
    food: int
    transportation: int
    sightseeing: int
    total: int


class TravelResponse(BaseModel):
    budget: TravelBudget


# ── /agent/goal/purchase ────────────────────────────────────────────────────

class PurchaseRequest(BaseModel):
    user_id: str
    deadline: date
    item_name: str


class PurchaseCandidate(BaseModel):
    product_name: str
    estimated_price: int
    description: str


class PurchaseResponse(BaseModel):
    item_name: str
    candidates: list[PurchaseCandidate]


# ── /agent/goal/analysis ────────────────────────────────────────────────────

class UserPortfolio(BaseModel):
    cash_ratio: int
    stock_ratio: int
    bond_ratio: int


class AnalysisRequest(BaseModel):
    user_id: str
    portfolio_user: UserPortfolio


class AnalysisResponse(BaseModel):
    analysis_report: str
    summary: str
