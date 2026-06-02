from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.consultant import (
    ResetAnalyzeRequest,
    ResetAnalyzeResponse,
    ResetProposeRequest,
    ResetProposeResponse,
)
from app.services.agent.consultant import analyze_goal, propose_reset

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consultant", tags=["consultant"])


@router.post("/analyze", response_model=ResetAnalyzeResponse)
async def analyze(req: ResetAnalyzeRequest) -> ResetAnalyzeResponse:
    try:
        return await asyncio.wait_for(analyze_goal(req), timeout=30)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="분석 시간 초과")
    except Exception as e:
        logger.exception("Consultant analyze error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/propose", response_model=ResetProposeResponse)
async def propose(req: ResetProposeRequest) -> ResetProposeResponse:
    try:
        return await asyncio.wait_for(propose_reset(req), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="제안 생성 시간 초과")
    except Exception as e:
        logger.exception("Consultant propose error")
        raise HTTPException(status_code=500, detail=str(e))
