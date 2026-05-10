from datetime import datetime
from enum import Enum

from pydantic import BaseModel


# ── Rebalance / Report ──────────────────────────────────────────────────────

class AllocationItem(BaseModel):
    category: str
    amount: float


class RebalanceRequest(BaseModel):
    user_id: str
    monthly_salary: float
    current_allocations: list[AllocationItem]
    financial_goals: list[str]


class RebalanceAllocation(BaseModel):
    category: str
    amount: float
    change: float
    reason: str


class RebalanceResponse(BaseModel):
    user_id: str
    recommended_allocations: list[RebalanceAllocation]
    summary: str


class Transaction(BaseModel):
    transaction_id: str
    amount: float
    category: str
    timestamp: datetime


class ReportRequest(BaseModel):
    user_id: str
    year_month: str
    transactions: list[Transaction]
    portfolio_snapshot: dict


class KeyMetrics(BaseModel):
    total_spending: float
    savings_rate: float
    top_categories: list[str]


class ReportResponse(BaseModel):
    user_id: str
    year_month: str
    report_markdown: str
    key_metrics: KeyMetrics


# ── 목표 포트폴리오 Agent 공통 ──────────────────────────────────────────────

class PortfolioComposition(BaseModel):
    cash_pct: float
    stocks_etf_pct: float
    bonds_pct: float


class PortfolioGoalBase(BaseModel):
    user_id: str
    duration_months: int    # 달성 기간 (개월)
    initial_capital: float  # 초기 자본금 (원)


# ── /agent/portfolio ────────────────────────────────────────────────────────

class PortfolioRequest(PortfolioGoalBase):
    target_amount: float  # 사용자가 확정한 총 비용 (원)


class PortfolioResponse(BaseModel):
    user_id: str
    annual_return_rate: float
    portfolio_composition: PortfolioComposition


# ── 종잣돈 Agent ────────────────────────────────────────────────────────────

class SeedMoneyRequest(PortfolioGoalBase):
    target_amount: float  # 목표 금액 (원)


class SeedMoneyResponse(BaseModel):
    user_id: str
    target_amount: float
    required_monthly_savings: float  # 월 필요 저축액 (원)


# ── 결혼 Agent ──────────────────────────────────────────────────────────────

class WeddingScale(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"


class WeddingRequest(PortfolioGoalBase):
    wedding_region: str          # 예식 지역
    wedding_month: int           # 예식 시기 (1–12월)
    honeymoon_scale: WeddingScale
    sdrme_scale: WeddingScale    # 스드메 규모


class WeddingBudget(BaseModel):
    venue: float       # 예식장
    honeymoon: float   # 신혼여행
    sdrme: float       # 스드메
    total: float


class WeddingResponse(BaseModel):
    user_id: str
    budget: WeddingBudget


# ── 해외여행 Agent ──────────────────────────────────────────────────────────

class TravelStyle(str, Enum):
    budget = "budget"   # 가성비
    luxury = "luxury"   # 럭셔리


class TravelRequest(PortfolioGoalBase):
    destination: str          # 여행 나라
    travel_style: TravelStyle
    travel_days: int          # 여행 기간 (일)
    departure_month: str      # 출발 예정 시기 (예: "2025-08")


class TravelBudget(BaseModel):
    accommodation: float   # 숙박
    flight: float          # 항공권
    food: float            # 식비
    transportation: float  # 교통
    sightseeing: float     # 관광
    total: float


class TravelResponse(BaseModel):
    user_id: str
    budget: TravelBudget


# ── 물건 사기 Agent ─────────────────────────────────────────────────────────

class PurchaseRequest(PortfolioGoalBase):
    item_name: str  # 물건명


class PurchaseCandidate(BaseModel):
    product_name: str
    estimated_price: float
    description: str


class PurchaseResponse(BaseModel):
    user_id: str
    item_name: str
    candidates: list[PurchaseCandidate]
