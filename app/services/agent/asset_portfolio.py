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
        investment_flows=[
            InvestmentPlan(
                title="[STUB] 분산 투자 플랜",
                term="중기",
                summary="중기적이고 안정적으로 자금을 모을 수 있어요",
                funding_sources=[
                    FundingSource(account_name="[STUB] 입출금 계좌", asset_id=request.invest_assets[0].asset_id, amount=request.invest_amount),
                ],
                gathering_account=request.invest_assets[0].asset_id,
                amount=300000,
                portfolio=[
                    PortfolioItem(name="[STUB] 국내주식 ETF", ratio=50),
                    PortfolioItem(name="[STUB] 채권형 펀드", ratio=30),
                    PortfolioItem(name="[STUB] 정기예금", ratio=20),
                ],
            )
        ],
    )
