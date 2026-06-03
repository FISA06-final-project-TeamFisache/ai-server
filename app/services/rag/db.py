from __future__ import annotations

import logging

import asyncpg

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool | None:
    global _pool
    if not settings.db_url:
        return None
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(
                settings.db_url,
                min_size=1,
                max_size=5,
                command_timeout=5,
            )
        except Exception as e:
            logger.warning("DB 연결 실패 (RAG 비활성화): %s", e)
            return None
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
