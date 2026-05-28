from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class SalaryRebalanceItem(BaseModel):
    asset_id: UUID
    category: str
    amount: int


class SalaryRequest(BaseModel):
    user_id: UUID
    salary_diff: int
    salary_rebalance: list[SalaryRebalanceItem]


class SalaryResponse(BaseModel):
    created_at: datetime
    salary_rebalance: list[SalaryRebalanceItem]
    rebalance_comment: str
