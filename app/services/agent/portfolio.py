from app.schemas.agent import PortfolioComposition, PortfolioRequest, PortfolioResponse


async def build_portfolio(request: PortfolioRequest) -> PortfolioResponse:
    # TODO: LangGraph 연동 (target_amount, duration_months, initial_capital 기반 포트폴리오 산출)
    return PortfolioResponse(
        user_id=request.user_id,
        annual_return_rate=5.0,
        portfolio_composition=PortfolioComposition(
            cash_pct=30.0,
            stocks_etf_pct=50.0,
            bonds_pct=20.0,
        ),
    )
