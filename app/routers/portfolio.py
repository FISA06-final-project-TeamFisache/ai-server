import asyncio

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.portfolio import (
    AssetPortfolioRequest,
    AssetPortfolioResponse,
    ProfileRequest,
    ProfileResponse,
    RebalanceRequest,
    RebalanceResponse,
)
from app.services.agent.asset_portfolio import recommend_asset_portfolio
from app.services.agent.portfolio_profile import analyze_profile
from app.services.agent.rebalance import rebalance_salary

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/profile", response_model=ProfileResponse, status_code=201)
async def portfolio_profile(req: ProfileRequest) -> ProfileResponse:
    try:
        return await asyncio.wait_for(analyze_profile(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Profile agent timed out")


@router.post("/rebalance", response_model=RebalanceResponse, status_code=201)
async def portfolio_rebalance(req: RebalanceRequest) -> RebalanceResponse:
    try:
        return await asyncio.wait_for(rebalance_salary(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Rebalance agent timed out")


@router.post("/asset-portfolio", response_model=AssetPortfolioResponse, status_code=201)
async def asset_portfolio(req: AssetPortfolioRequest) -> AssetPortfolioResponse:
    try:
        return await asyncio.wait_for(recommend_asset_portfolio(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Asset portfolio agent timed out")
