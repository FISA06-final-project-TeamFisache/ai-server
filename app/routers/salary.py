import asyncio

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.salary import SalaryRequest, SalaryResponse
from app.services.agent.salary_rebalance import analyze_salary_rebalance

router = APIRouter(prefix="/salary", tags=["salary"])


@router.post("/", response_model=SalaryResponse, status_code=201)
async def salary_rebalance(req: SalaryRequest) -> SalaryResponse:
    try:
        return await asyncio.wait_for(analyze_salary_rebalance(req), timeout=settings.agent_timeout_rebalance)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Salary rebalance agent timed out")
