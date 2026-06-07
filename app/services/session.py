from __future__ import annotations

import json
import logging
import os
from uuid import UUID

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_TTL = 7 * 24 * 3600  # 7일
_KEY_PREFIX = "mini_challenge"


def _make_key(user_id: UUID) -> str:
    return f"{_KEY_PREFIX}:{user_id}"


def _get_redis() -> aioredis.Redis:
    return aioredis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
    )


async def get_session(user_id: UUID) -> dict:
    r = _get_redis()
    try:
        raw = await r.get(_make_key(user_id))
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning("Redis 세션 조회 실패 (user=%s): %s", user_id, e)
    finally:
        await r.aclose()
    return {"category_expense": [], "stock_themes": [], "proposals": []}


async def delete_session(user_id: UUID) -> None:
    r = _get_redis()
    try:
        await r.delete(_make_key(user_id))
    except Exception as e:
        logger.warning("Redis 세션 삭제 실패 (user=%s): %s", user_id, e)
    finally:
        await r.aclose()


async def save_session(user_id: UUID, data: dict) -> None:
    r = _get_redis()
    try:
        await r.setex(
            _make_key(user_id),
            _TTL,
            json.dumps(data, ensure_ascii=False, default=str),
        )
    except Exception as e:
        logger.warning("Redis 세션 저장 실패 (user=%s): %s", user_id, e)
    finally:
        await r.aclose()
