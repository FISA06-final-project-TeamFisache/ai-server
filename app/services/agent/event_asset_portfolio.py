from datetime import datetime, timezone

from app.schemas.event import (
    EventAssetPortfolioRequest,
    EventAssetPortfolioResponse,
    FundingSource,
    InvestmentPlan,
    PortfolioItem,
)


async def asset_portfolio_event(request: EventAssetPortfolioRequest) -> EventAssetPortfolioResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return EventAssetPortfolioResponse(
        created_at=now,
        investment_flows=[
            InvestmentPlan(
                title="[STUB] 목표 달성 투자 플랜",
                term="장기",
                summary="장기적인 목표를 달성하기 위해 플랜을 짜봤어요",
                funding_sources=[
                    FundingSource(account_name="[STUB] 입출금 계좌", asset_id=request.investment_flows[0].funding_sources[0].asset_id, amount=request.invest_amount),
                ],
                gathering_account=request.investment_flows[0].gathering_account,
                amount=300000,
                portfolio=[
                    PortfolioItem(name="[STUB] 국내주식 ETF", ratio=50),
                    PortfolioItem(name="[STUB] 채권형 펀드", ratio=30),
                    PortfolioItem(name="[STUB] 정기예금", ratio=20),
                ],
            )
        ],
    )
