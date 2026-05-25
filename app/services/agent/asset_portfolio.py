from datetime import datetime, timezone

from app.schemas.portfolio import (
    AssetPortfolioRequest,
    AssetPortfolioResponse,
    FundingSource,
    InvestmentPlan,
    PortfolioItem,
)


async def recommend_asset_portfolio(request: AssetPortfolioRequest) -> AssetPortfolioResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return AssetPortfolioResponse(
        created_at=now,
        investment_plans=[
            InvestmentPlan(
                funding_sources=[
                    FundingSource(account_name="[STUB] 입출금 계좌", amount=request.invest_amount),
                ],
                gathering_account="[STUB] 모을 통장",
                portfolio=[
                    PortfolioItem(name="[STUB] 국내주식 ETF", ratio=50),
                    PortfolioItem(name="[STUB] 채권형 펀드", ratio=30),
                    PortfolioItem(name="[STUB] 정기예금", ratio=20),
                ],
                description="[STUB] 안정적인 수익을 위한 분산 투자 포트폴리오입니다.",
            )
        ],
    )
