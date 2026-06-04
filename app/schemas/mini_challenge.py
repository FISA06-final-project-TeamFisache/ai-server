from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CategoryExpenseItem(BaseModel):
    name: str
    expense: int


# ── POST /mini_challenge/ ─────────────────────────────────────────────────────

class MiniChallengeRequest(BaseModel):
    user_id: UUID
    category_expense: list[CategoryExpenseItem]
    porti_type: str
    porti_comment: str


class MiniChallengeResponse(BaseModel):
    """초기 제안 응답. target = 달성 버튼 1회당 진행률 증가값 (step_size)."""
    created_at: datetime
    icon: str
    title: str
    description: str
    difficulty: int
    category: str
    target: int          # step_size: 달성 1회당 진행률 (100 / 목표달성횟수)
    challenge_type: str  # count | amount
    estimated_saving: int
    ticker: str


# ── POST /mini_challenge/adjust ───────────────────────────────────────────────

class PreviousProposalItem(BaseModel):
    """adjust 요청 시 기피할 이전 제안 목록."""
    model_config = ConfigDict(populate_by_name=True)

    title: str
    difficulty: int
    description: str = ""
    challenge_type: str = Field(default="count", alias="challengeType")
    category: str
    estimated_saving: int = Field(default=0, alias="estimatedSaving")
    ticker: str = Field(default="", alias="ticker")


class AdjustRequest(BaseModel):
    user_id: UUID
    category_expense: list[CategoryExpenseItem]
    previous_proposals: list[PreviousProposalItem] = []
    feedback: str | None = None   # lower | higher | different
    porti_type: str = ""
    porti_comment: str = ""


class AdjustResponse(BaseModel):
    """조정 제안 응답. target = 실제 목표값 (횟수 또는 금액)."""
    created_at: datetime
    icon: str
    title: str
    difficulty: int
    challenge_type: str
    target: int | None   # 실제 목표 (count → 횟수, amount → 원)
    category: str
    description: str
    ticker: str
    estimated_saving: int


# ── POST /mini_challenge/reward ───────────────────────────────────────────────

class RewardRequest(BaseModel):
    user_id: UUID
    challenge_title: str
    estimated_saving: int
    ticker: str


class RewardResponse(BaseModel):
    created_at: datetime
    message: str
    stock_name: str
    ticker: str
    current_price: int
    shares: float


# ── POST /mini_challenge/nag ──────────────────────────────────────────────────

class NagRequest(BaseModel):
    user_id: UUID
    title: str
    category: str
    challenge_type: str
    target: int | None   # 목표 횟수 / 금액
    current: int         # 현재 횟수 / 금액
    progress_pct: int    # 50 | 80 | 90


class NagResponse(BaseModel):
    created_at: datetime
    nag_message: str
