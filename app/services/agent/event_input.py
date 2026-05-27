from datetime import datetime, timezone

from app.schemas.event import EventInputRequest, EventInputResponse


async def analyze_event_input(request: EventInputRequest) -> EventInputResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return EventInputResponse(
        created_at=now,
        title="[STUB] 유럽 여행 자금 마련",
        target_amount="10,000,000",
        deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
    )
