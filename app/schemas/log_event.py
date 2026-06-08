from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MiniChallengeAcceptedEvent(BaseModel):
    event_type: str = "mini_challenge_accepted"
    timestamp: str = Field(default_factory=_now)
    user_id: str
    adjust_count: int
    final_challenge_sub_type: str
