from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MiniChallengeProposedEvent(BaseModel):
    event_type: str = "mini_challenge_proposed"
    timestamp: str = Field(default_factory=_now)
    user_id: str
    proposal_round: int       # 누적 제안 횟수 (1부터 시작)
    title: str
    category: str
    challenge_sub_type: str
    challenge_type: str
    target: int
    estimated_saving: int
    ticker: str
    latency_ms: int


class MiniChallengeAdjustedEvent(BaseModel):
    event_type: str = "mini_challenge_adjusted"
    timestamp: str = Field(default_factory=_now)
    user_id: str
    proposal_round: int       # adjust 후 누적 제안 횟수
    feedback: str             # lower | higher | different
    title: str
    category: str
    challenge_sub_type: str
    challenge_type: str
    target: Optional[int] = None
    estimated_saving: int
    ticker: str
    latency_ms: int


class MiniChallengeAcceptedEvent(BaseModel):
    event_type: str = "mini_challenge_accepted"
    timestamp: str = Field(default_factory=_now)
    user_id: str
    total_proposals: int      # 수락까지 총 제안 횟수 (adjust 횟수 + 1)
    final_category: str
    final_challenge_sub_type: str
    final_estimated_saving: int
    final_ticker: str
