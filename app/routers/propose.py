from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.propose import ProposeRequest, ProposalResponse
from app.services.agent.propose import propose as propose_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/propose", tags=["propose"])


@router.post("", response_model=ProposalResponse)
async def propose(req: ProposeRequest) -> ProposalResponse:
    try:
        return await asyncio.wait_for(propose_agent(req), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Propose agent timed out")
    except Exception as e:
        logger.exception("Propose agent error")
        raise HTTPException(status_code=500, detail=str(e))
