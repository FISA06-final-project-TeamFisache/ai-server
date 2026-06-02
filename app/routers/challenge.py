from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.challenge import ChallengeRequest, ChallengeResponse, RewardRequest, RewardResponse
from app.services.agent.challenge import propose_challenge
from app.services.stock import get_all_prices, pick_stock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/challenge", tags=["challenge"])


@router.post("", response_model=ChallengeResponse)
async def challenge(req: ChallengeRequest) -> ChallengeResponse:
    try:
        return await asyncio.wait_for(propose_challenge(req), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Challenge agent timed out")
    except Exception as e:
        logger.exception("Challenge agent error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reward", response_model=RewardResponse)
async def reward(req: RewardRequest) -> RewardResponse:
    try:
        prices = await asyncio.wait_for(get_all_prices(), timeout=15)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="주식 데이터 조회 시간 초과")
    except Exception as e:
        logger.exception("yfinance error")
        raise HTTPException(status_code=503, detail=str(e))

    stock = pick_stock(prices, req.estimated_saving)
    if not stock:
        raise HTTPException(status_code=503, detail="조회 가능한 종목이 없습니다")

    name, ticker, price, shares = stock
    message = (
        f"절약한 {req.estimated_saving:,}원으로 "
        f"{name} {shares:.2f}주 살 수 있어요! "
        f"(현재가 {price:,}원)"
    )
    return RewardResponse(
        message=message,
        stock_name=name,
        ticker=ticker,
        current_price=price,
        shares=round(shares, 4),
    )
