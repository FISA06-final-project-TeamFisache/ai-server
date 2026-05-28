from datetime import datetime, timezone

from app.schemas.event import EventRebalanceRequest, EventRebalanceResponse, SalaryRebalanceItem


async def rebalance_event(request: EventRebalanceRequest) -> EventRebalanceResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return EventRebalanceResponse(
        created_at=now,
        invest_amount=500000,
        salary_rebalance=[
            SalaryRebalanceItem(account_name="우리 WON 생활통장", asset_id=request.rebalance.salary_rebalance[0].asset_id, category="생활비", amount=400000),
            SalaryRebalanceItem(account_name="하나 파킹통장", asset_id=request.rebalance.salary_rebalance[0].asset_id, category="비상금", amount=200000),
            SalaryRebalanceItem(account_name="KB 이벤트통장", asset_id=request.rebalance.salary_rebalance[0].asset_id, category="목표저축", amount=400000),
        ],
        rebalance_comment="[STUB] 이벤트 목표에 따른 자산 재설계 결과입니다.",
    )
