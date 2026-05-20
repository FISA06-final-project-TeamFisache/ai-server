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
            SalaryRebalanceItem(asset_number="123456-78-9012345", category="저축", ratio=30),
            SalaryRebalanceItem(asset_number="234567-89-0123456", category="투자", ratio=40),
            SalaryRebalanceItem(asset_number="345678-90-1234567", category="생활비", ratio=30),
        ],
        portfolio_recommend=PortfolioRecommendResult(
            stock_ratio=50,
            stock_recs=["[STUB] 삼성전자", "[STUB] SK하이닉스"],
            bond_ratio=30,
            bond_recs=["[STUB] 국고채 3년"],
            cash_ratio=20,
            cash_recs=["[STUB] MMF"],
        ),
        audit_log=AuditLog(
            log_id=str(uuid4()),
            user_id=str(request.user_id),
            endpoint="/agent/event/input",
            requested_at=now,
        ),
    )
