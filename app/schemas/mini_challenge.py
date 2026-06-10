from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChallengeType(str, Enum):
    COUNT = "COUNT"
    AMOUNT = "AMOUNT"


class CategoryExpenseItem(BaseModel):
    amount: int
    category: str
    sender_name: str
    transaction_at: datetime


# ── POST /mini_challenge ─────────────────────────────────────────────────────

class MiniChallengeRequest(BaseModel):
    user_id: UUID
    category_expense: list[CategoryExpenseItem]
    stock_themes: list[str]


class MiniChallengeResponse(BaseModel):
    """초기 제안 응답. target = 달성 버튼 1회당 진행률 증가값 (step_size)."""
    created_at: datetime
    title: str
    description: str
    category: str
    target: int
    challenge_type: ChallengeType
    estimated_saving: int
    ticker: str
    challenge_sub_type: str


# ── POST /mini_challenge/adjust ───────────────────────────────────────────────

class AdjustRequest(BaseModel):
    """세션에 이전 제안·소비 데이터가 있으므로 user_id만 필요."""
    user_id: UUID
    feedback: str  # 더 쉽게 조정해주세요 | 더 어렵게 조정해주세요 | 주제를 바꿔주세요


class AdjustResponse(BaseModel):
    """조정 제안 응답. target = 실제 목표값 (횟수 또는 금액)."""
    created_at: datetime
    title: str
    challenge_type: ChallengeType
    target: int | None
    category: str
    description: str
    ticker: str
    estimated_saving: int
    challenge_sub_type: str

# ── POST /mini_challenge/reward ───────────────────────────────────────────────

class RewardRequest(BaseModel):
    """세션의 마지막 챌린지 ticker·estimated_saving을 사용하므로 user_id만 필요."""
    user_id: UUID


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
    challenge_type: ChallengeType
    target: int | None
    current: int
    progress_pct: int  # 50 | 80 | 90


class NagResponse(BaseModel):
    created_at: datetime
    nag_message: str
