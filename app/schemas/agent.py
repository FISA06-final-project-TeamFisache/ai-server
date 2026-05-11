from enum import Enum

from pydantic import BaseModel


# ── Rebalance / Report ──────────────────────────────────────────────────────

class AllocationItem(BaseModel):
    category: str
    amount: int


class RebalanceRequest(BaseModel):
    user_id: str
    monthly_salary: int
    current_allocations: list[AllocationItem]
    financial_goals: list[str]


class RebalanceAllocation(BaseModel):
    category: str
    amount: int
    change: int
    reason: str


class RebalanceResponse(BaseModel):
    user_id: str
    recommended_allocations: list[RebalanceAllocation]
    summary: str


class PortfolioComposition(BaseModel):
    cash_pct: int
    stocks_etf_pct: int
    bonds_pct: int


class PortfolioChangeRate(BaseModel):
    stock_rate: float
    bond_rate: float
    cash_rate: float


class ExistingPortfolioRatio(BaseModel):
    stock_ratio: int
    bond_ratio: int
    cash_ratio: int


class ReportRequest(BaseModel):
    user_id: str
    total_assets: int
    asset_change_prev_month: int
    portfolio_change_rate: PortfolioChangeRate
    existing_portfolio_ratio: ExistingPortfolioRatio


class RecommendedRebalanceRatio(BaseModel):
    stock_ratio: int
    bond_ratio: int
    cash_ratio: int


class ReportResponse(BaseModel):
    portfolio_report: str
    recommended_rebalance_ratio: RecommendedRebalanceRatio
    recommendation_comment: str


# ── 목표 포트폴리오 Agent 공통 ──────────────────────────────────────────────


class PortfolioGoalBase(BaseModel):
    user_id: str
    duration_months: int    # 달성 기간 (개월)
    initial_capital: int  # 초기 자본금 (원)


# ── /agent/portfolio ────────────────────────────────────────────────────────

class PortfolioRequest(PortfolioGoalBase):
    target_amount: int  # 사용자가 확정한 총 비용 (원)


class PortfolioResponse(BaseModel):
    annual_return_rate: float
    portfolio_composition: PortfolioComposition


# ── 종잣돈 Agent ────────────────────────────────────────────────────────────

class SeedMoneyRequest(PortfolioGoalBase):
    target_amount: int  # 목표 금액 (원)


class SeedMoneyResponse(BaseModel):
    target_amount: int
    required_monthly_savings: int  # 월 필요 저축액 (원)


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
    venue: int       # 예식장
    honeymoon: int   # 신혼여행
    sdrme: int       # 스드메
    total: int


class WeddingResponse(BaseModel):
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
    accommodation: int   # 숙박
    flight: int          # 항공권
    food: int            # 식비
    transportation: int  # 교통
    sightseeing: int     # 관광
    total: int


class TravelResponse(BaseModel):
    budget: TravelBudget


# ── 물건 사기 Agent ─────────────────────────────────────────────────────────

class PurchaseRequest(PortfolioGoalBase):
    item_name: str  # 물건명


class PurchaseCandidate(BaseModel):
    product_name: str
    estimated_price: int
    description: str


class PurchaseResponse(BaseModel):
    item_name: str
    candidates: list[PurchaseCandidate]
