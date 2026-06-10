from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ResetAnalyzeRequest(BaseModel):
    user_goal: str
    dashboard_snapshot: dict[str, Any]


class ResetAnalyzeResponse(BaseModel):
    action: Literal["salary", "portfolio"]
    reasoning: str


class ResetProposeRequest(BaseModel):
    user_goal: str
    action: Literal["salary", "portfolio"]
    dashboard_snapshot: dict[str, Any]


class ResetAllocation(BaseModel):
    purpose: str
    plannedAmount: int
    ratio: int


class ResetPortfolioItem(BaseModel):
    assetType: str
    ratio: int


class FlowProductItem(BaseModel):
    productId: str
    productRatio: int


class FlowUpdate(BaseModel):
    flowId: str
    amount: int | None = None
    products: list[FlowProductItem] = []


class ResetProposeResponse(BaseModel):
    summary: str
    explanation: str
    salary_allocations: list[ResetAllocation] = []
    portfolio: list[ResetPortfolioItem] = []
    flows: list[FlowUpdate] = []
