import asyncio

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.event import (
    EventAssetPortfolioRequest,
    EventAssetPortfolioResponse,
    EventInputRequest,
    EventInputResponse,
    EventRebalanceRequest,
    EventRebalanceResponse,
)
from app.services.agent.event_asset_portfolio import asset_portfolio_event
from app.services.agent.event_input import analyze_event_input
from app.services.agent.event_rebalance import rebalance_event

router = APIRouter(prefix="/event", tags=["event"])


@router.post("/input", response_model=EventInputResponse, status_code=201)
async def event_input(req: EventInputRequest) -> EventInputResponse:
    try:
        return await asyncio.wait_for(analyze_event_input(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Event input agent timed out")


@router.post("/rebalance", response_model=EventRebalanceResponse, status_code=201)
async def event_rebalance(req: EventRebalanceRequest) -> EventRebalanceResponse:
    try:
        return await asyncio.wait_for(rebalance_event(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Event rebalance agent timed out")


@router.post("/asset-portfolio", response_model=EventAssetPortfolioResponse, status_code=201)
async def event_asset_portfolio(req: EventAssetPortfolioRequest) -> EventAssetPortfolioResponse:
    try:
        return await asyncio.wait_for(asset_portfolio_event(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Event asset portfolio agent timed out")
