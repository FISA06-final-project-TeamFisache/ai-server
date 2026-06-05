import asyncio

from fastapi import APIRouter, HTTPException

from app.core import config
from app.schemas.report import ReportRequest, ReportResponse
from app.services.agent.report import generate_report

router = APIRouter(prefix="/report", tags=["report"])


@router.post("", response_model=ReportResponse, status_code=201)
async def report(req: ReportRequest) -> ReportResponse:
    try:
        return await asyncio.wait_for(generate_report(req), timeout=config.AGENT_TIMEOUT_REPORT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Report agent timed out")
