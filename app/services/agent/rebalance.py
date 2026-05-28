from datetime import datetime, timezone

from app.schemas.portfolio import RebalanceRequest, RebalanceResponse, SalaryRebalanceItem


async def rebalance_salary(request: RebalanceRequest) -> RebalanceResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return RebalanceResponse(
        created_at=now,
        invest_amount=500000,
        salary_rebalance=[
            SalaryRebalanceItem(account_name="우리 WON 생활통장", asset_id=request.assets[0].asset_id, category="생활비", amount=400000),
            SalaryRebalanceItem(account_name="하나 파킹통장", asset_id=request.assets[0].asset_id, category="비상금", amount=200000),
            SalaryRebalanceItem(account_name="KB 이벤트통장", asset_id=request.assets[0].asset_id, category="목표저축", amount=400000),
        ],
    )
