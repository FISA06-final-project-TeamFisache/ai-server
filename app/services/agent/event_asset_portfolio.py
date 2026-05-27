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
                priority=1,
                funding_sources=[
                    FundingSource(asset_number="[STUB] 입출금 계좌", amount=request.invest_amount),
                ],
                gathering_account="[STUB] 목표 모음 통장",
                portfolio=[
                    PortfolioItem(name="[STUB] 국내주식 ETF", ratio=50),
                    PortfolioItem(name="[STUB] 채권형 펀드", ratio=30),
                    PortfolioItem(name="[STUB] 정기예금", ratio=20),
                ],
                description="[STUB] 이벤트 목표에 따른 포트폴리오 재구성 결과입니다.",
            )
        ],
    )
