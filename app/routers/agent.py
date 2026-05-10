import asyncio

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.agent import (
    PortfolioRequest, PortfolioResponse,
    RebalanceRequest, RebalanceResponse,
    ReportRequest, ReportResponse,
    SeedMoneyRequest, SeedMoneyResponse,
    WeddingRequest, WeddingResponse,
    TravelRequest, TravelResponse,
    PurchaseRequest, PurchaseResponse,
)
from app.services.agent.portfolio import build_portfolio
from app.services.agent.rebalance import rebalance_salary
from app.services.agent.report import generate_report
from app.services.agent.seed_money import build_seed_money
from app.services.agent.wedding import build_wedding
from app.services.agent.travel import build_travel
from app.services.agent.purchase import build_purchase

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/portfolio", response_model=PortfolioResponse)
async def portfolio(req: PortfolioRequest) -> PortfolioResponse:
    try:
        return await asyncio.wait_for(build_portfolio(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Portfolio agent timed out")


@router.post("/rebalance", response_model=RebalanceResponse)
async def rebalance(req: RebalanceRequest) -> RebalanceResponse:
    try:
        return await asyncio.wait_for(rebalance_salary(req), timeout=settings.agent_timeout_rebalance)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Rebalance agent timed out")


@router.post("/report", response_model=ReportResponse)
async def report(req: ReportRequest) -> ReportResponse:
    try:
        return await asyncio.wait_for(generate_report(req), timeout=settings.agent_timeout_report)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Report agent timed out")


@router.post("/portfolio/seed-money", response_model=SeedMoneyResponse)
async def portfolio_seed_money(req: SeedMoneyRequest) -> SeedMoneyResponse:
    try:
        return await asyncio.wait_for(build_seed_money(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Seed money agent timed out")


@router.post("/portfolio/wedding", response_model=WeddingResponse)
async def portfolio_wedding(req: WeddingRequest) -> WeddingResponse:
    try:
        return await asyncio.wait_for(build_wedding(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Wedding agent timed out")


@router.post("/portfolio/travel", response_model=TravelResponse)
async def portfolio_travel(req: TravelRequest) -> TravelResponse:
    try:
        return await asyncio.wait_for(build_travel(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Travel agent timed out")


@router.post("/portfolio/purchase", response_model=PurchaseResponse)
async def portfolio_purchase(req: PurchaseRequest) -> PurchaseResponse:
    try:
        return await asyncio.wait_for(build_purchase(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Purchase agent timed out")
