from datetime import datetime, timezone

from app.schemas.event import EventRebalanceRequest, EventRebalanceResponse, SalaryRebalanceItem


async def rebalance_event(request: EventRebalanceRequest) -> EventRebalanceResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return EventRebalanceResponse(
        created_at=now,
        title="[STUB] 유럽 여행 자금 마련",
        target_amount="10,000,000",
        deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
        salary_rebalance=[
            SalaryRebalanceItem(asset_number="123456-78-9012345", category="생활비", ratio=40),
            SalaryRebalanceItem(asset_number="234567-89-0123456", category="비상금", ratio=20),
            SalaryRebalanceItem(asset_number="345678-90-1234567", category="목표저축", ratio=40),
        ],
        rebalance_comment="[STUB] 이벤트 목표에 따른 자산 재설계 결과입니다.",
    )
