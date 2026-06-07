from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.schemas.mini_challenge import (
    AdjustRequest,
    AdjustResponse,
    MiniChallengeRequest,
    MiniChallengeResponse,
    NagRequest,
    NagResponse,
    RewardRequest,
    RewardResponse,
)
from app.services.agent.mini_challenge_agent import (
    adjust_challenge,
    generate_nag,
    get_last_proposal,
    propose_mini_challenge,
)
from app.services.session import delete_session, get_session
from app.services.agent.tools import get_all_prices, pick_stock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mini_challenge", tags=["mini_challenge"])


@router.post("", response_model=MiniChallengeResponse)
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


@router.post("/reward", response_model=RewardResponse)
async def reward(req: RewardRequest) -> RewardResponse:
    session = await get_session(req.user_id)
    last = get_last_proposal(session)
    if not last:
        raise HTTPException(status_code=404, detail="진행 중인 챌린지가 없습니다.")

    ticker = last["ticker"]
    estimated_saving = last.get("estimated_saving", 0)

    try:
        prices = await asyncio.wait_for(get_all_prices(), timeout=15)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="주식 데이터 조회 시간 초과")
    except Exception as e:
        logger.exception("yfinance error")
        raise HTTPException(status_code=503, detail=str(e))

    # 세션 ticker 우선 조회, 없으면 전체 리스트에서 최적 종목 선택
    target = next(((n, t, p) for n, t, p in prices if t == ticker), None)
    if target and target[2] > 0:
        name, t, price = target
        shares = estimated_saving / price
    else:
        stock = pick_stock(prices, estimated_saving)
        if not stock:
            raise HTTPException(status_code=503, detail="조회 가능한 종목이 없습니다")
        name, t, price, shares = stock

    await delete_session(req.user_id)

    message = (
        f"절약한 {estimated_saving:,}원으로 "
        f"{name} {shares:.2f}주 살 수 있어요! "
        f"(현재가 {price:,}원)"
    )
    return RewardResponse(
        created_at=datetime.now(timezone.utc),
        message=message,
        stock_name=name,
        ticker=t,
        current_price=price,
        shares=round(shares, 4),
    )


@router.post("/nag", response_model=NagResponse)
async def nag(req: NagRequest) -> NagResponse:
    try:
        return await asyncio.wait_for(generate_nag(req), timeout=30)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Nag agent timed out")
    except Exception as e:
        logger.exception("Nag agent error")
        raise HTTPException(status_code=500, detail=str(e))
