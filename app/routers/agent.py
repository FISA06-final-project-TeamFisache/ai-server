import asyncio

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.agent import (
    PortfolioRequest, PortfolioResponse,
    ReportRequest, ReportResponse,
    SeedMoneyRequest, SeedMoneyResponse,
    WeddingRequest, WeddingResponse,
    TravelRequest, TravelResponse,
    PurchaseRequest, PurchaseResponse,
    AnalysisRequest, AnalysisResponse,
)
from app.services.agent.portfolio import build_portfolio
from app.services.agent.report import generate_report
from app.services.agent.seed_money import build_seed_money
from app.services.agent.wedding import build_wedding
from app.services.agent.travel import build_travel
from app.services.agent.purchase import build_purchase
from app.services.agent.analysis import build_analysis

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/report", response_model=ReportResponse)
async def report(req: ReportRequest) -> ReportResponse:
    try:
        return await asyncio.wait_for(generate_report(req), timeout=settings.agent_timeout_report)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Report agent timed out")


@router.post("/goal/portfolio", response_model=PortfolioResponse, status_code=201)
async def goal_portfolio(req: PortfolioRequest) -> PortfolioResponse:
    try:
        return await asyncio.wait_for(build_portfolio(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Portfolio agent timed out")


@router.post("/goal/seed-money", response_model=SeedMoneyResponse)
async def goal_seed_money(req: SeedMoneyRequest) -> SeedMoneyResponse:
    try:
        return await asyncio.wait_for(build_seed_money(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Seed money agent timed out")


@router.post("/goal/wedding", response_model=WeddingResponse)
async def goal_wedding(req: WeddingRequest) -> WeddingResponse:
    try:
        return await asyncio.wait_for(build_wedding(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Wedding agent timed out")


@router.post("/goal/travel", response_model=TravelResponse)
async def goal_travel(req: TravelRequest) -> TravelResponse:
    try:
        return await asyncio.wait_for(build_travel(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Travel agent timed out")


@router.post("/goal/purchase", response_model=PurchaseResponse)
async def goal_purchase(req: PurchaseRequest) -> PurchaseResponse:
    try:
        return await asyncio.wait_for(build_purchase(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Purchase agent timed out")


@router.post("/goal/analysis", response_model=AnalysisResponse)
async def goal_analysis(req: AnalysisRequest) -> AnalysisResponse:
    try:
        return await asyncio.wait_for(build_analysis(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Analysis agent timed out")
