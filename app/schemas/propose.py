from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class ProposeRequest(BaseModel):
    user_message: str
    dashboard_snapshot: dict[str, Any]


class ProposalEvent(BaseModel):
    title: str
    targetAmount: str
    deadline: str
    userInput: str


class ProposalAllocation(BaseModel):
    purpose: str
    plannedAmount: int


class ProposalPortfolioItem(BaseModel):
    assetType: str
    assetAmount: int


class ProposalChanges(BaseModel):
    events: list[ProposalEvent] = []
    salaryAllocations: list[ProposalAllocation] = []
    portfolio: list[ProposalPortfolioItem] = []


class ProposalResponse(BaseModel):
    summary: str
    explanation: str
    changes: ProposalChanges
