from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.mini_challenge import (
    AdjustRequest,
    AdjustResponse,
    MiniChallengeRequest,
    MiniChallengeResponse,
    NagRequest,
    NagResponse,
)
from app.services.agent.mini_challenge_agent import (
    adjust_challenge,
    generate_nag,
    propose_mini_challenge,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mini_challenge", tags=["mini_challenge"])


@router.post("/", response_model=MiniChallengeResponse)
async def mini_challenge(req: MiniChallengeRequest) -> MiniChallengeResponse:
    try:
        return await asyncio.wait_for(propose_mini_challenge(req), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Mini challenge agent timed out")
    except Exception as e:
        logger.exception("Mini challenge propose error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adjust", response_model=AdjustResponse)
async def adjust(req: AdjustRequest) -> AdjustResponse:
    try:
        return await asyncio.wait_for(adjust_challenge(req), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Mini challenge adjust timed out")
    except Exception as e:
        logger.exception("Mini challenge adjust error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nag", response_model=NagResponse)
async def nag(req: NagRequest) -> NagResponse:
    try:
        return await asyncio.wait_for(generate_nag(req), timeout=30)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Nag agent timed out")
    except Exception as e:
        logger.exception("Nag agent error")
        raise HTTPException(status_code=500, detail=str(e))
