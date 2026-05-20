import asyncio

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.agent import (
    EventInputRequest, EventInputResponse,
    ProfileRequest, ProfileResponse,
    RecommendRequest, RecommendResponse,
    ReportRequest, ReportResponse,
)
from app.services.agent.event import handle_event_input
from app.services.agent.profile import analyze_profile
from app.services.agent.recommend import recommend_portfolio
from app.services.agent.report import generate_report

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/portfolio/profile", response_model=ProfileResponse, status_code=201)
async def portfolio_profile(req: ProfileRequest) -> ProfileResponse:
    try:
        return await asyncio.wait_for(analyze_profile(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Profile agent timed out")


@router.post("/portfolio/recommend", response_model=RecommendResponse, status_code=201)
async def portfolio_recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        return await asyncio.wait_for(recommend_portfolio(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Recommend agent timed out")


@router.post("/event/input", response_model=EventInputResponse, status_code=201)
async def event_input(req: EventInputRequest) -> EventInputResponse:
    try:
        return await asyncio.wait_for(handle_event_input(req), timeout=settings.agent_timeout_portfolio)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Event input agent timed out")


@router.post("/report", response_model=ReportResponse, status_code=201)
async def report(req: ReportRequest) -> ReportResponse:
    try:
        return await asyncio.wait_for(generate_report(req), timeout=settings.agent_timeout_report)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Report agent timed out")
