from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.agent import (
    AuditLog,
    EventInputRequest,
    EventInputResponse,
    PortfolioRecommendResult,
    SalaryRebalanceItem,
)


async def handle_event_input(request: EventInputRequest) -> EventInputResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return EventInputResponse(
        created_at=now,
        salary_rebalance=[
            SalaryRebalanceItem(category="저축", ratio=30),
            SalaryRebalanceItem(category="투자", ratio=40),
            SalaryRebalanceItem(category="생활비", ratio=30),
        ],
        portfolio_recommend=PortfolioRecommendResult(
            stock_ratio=50,
            bond_ratio=30,
            cash_ratio=20,
        ),
        audit_log=AuditLog(
            log_id=str(uuid4()),
            user_id=str(request.user_id),
            endpoint="/agent/event/input",
            requested_at=now,
        ),
    )
