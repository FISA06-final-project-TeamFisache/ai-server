from datetime import datetime, timezone
from uuid import UUID

from app.schemas.salary import SalaryRequest, SalaryResponse, PortfolioItem


async def analyze_salary_rebalance(request: SalaryRequest) -> SalaryResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return SalaryResponse(
        created_at=now,
        portfolio_items=[
            PortfolioItem(asset_id=UUID("00000000-0000-0000-0000-000000000001"), category="생활비", amount=400000),
            PortfolioItem(asset_id=UUID("00000000-0000-0000-0000-000000000002"), category="저축", amount=300000),
            PortfolioItem(asset_id=UUID("00000000-0000-0000-0000-000000000003"), category="비상금", amount=300000),
        ],
        flow_items=request.flow_items,
        rebalance_comment="[STUB] 월급 초과/결손에 따라 이렇게 금액을 바꿔서 추천드려요!",
    )
