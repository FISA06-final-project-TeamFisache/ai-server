from __future__ import annotations

from pydantic import BaseModel


class ConsumptionCategory(BaseModel):
    categoryName: str
    expenseAmount: int
    percentage: float


class ChallengeRequest(BaseModel):
    consumption_categories: list[ConsumptionCategory]
    previous_proposals: list[dict] = []
    feedback: str | None = None          # null | 'lower' | 'higher' | 'different'
    current_proposal: dict | None = None  # 난이도 조정 시 현재 제안 컨텍스트


class _ChallengeAIOutput(BaseModel):
    """AI가 반환하는 원시 필드 (step_size 제외)."""
    icon: str
    title: str
    difficulty: int
    reasoning: str
    target_count: int       # 줄이고자 하는 횟수 — 제목의 숫자와 반드시 일치
    last_month_count: int   # 지난달 해당 카테고리 추정 횟수 (소비 금액 기반 임시값)
    estimated_saving: int


class ChallengeResponse(BaseModel):
    icon: str
    title: str
    difficulty: int
    reasoning: str
    target_count: int       # 줄이고자 하는 횟수
    last_month_count: int   # 지난달 추정 횟수
    step_size: int          # = round(100 / target_count), 서버 계산
    estimated_saving: int


class RewardRequest(BaseModel):
    challenge_title: str
    estimated_saving: int


class RewardResponse(BaseModel):
    message: str
    stock_name: str
    ticker: str
    current_price: int
    shares: float
