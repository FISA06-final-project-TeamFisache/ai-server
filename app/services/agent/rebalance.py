from datetime import datetime, timezone

from app.schemas.portfolio import RebalanceRequest, RebalanceResponse, SalaryRebalanceItem


async def rebalance_salary(request: RebalanceRequest) -> RebalanceResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return RebalanceResponse(
        created_at=now,
        invest_amount=500000,
        salary_rebalance=[
            SalaryRebalanceItem(asset_number="123456-78-9012345", category="저축", ratio=30),
            SalaryRebalanceItem(asset_number="234567-89-0123456", category="생활비", ratio=40),
            SalaryRebalanceItem(asset_number="345678-90-1234567", category="비상금", ratio=30),
        ],
    )
