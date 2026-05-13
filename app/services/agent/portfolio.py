from app.schemas.agent import PortfolioComposition, PortfolioRequest, PortfolioResponse


async def build_portfolio(request: PortfolioRequest) -> PortfolioResponse:
    # TODO: LangGraph 연동 (deadline, initial_capital, monthly_seed, target_amount 기반 포트폴리오 산출)
    return PortfolioResponse(
        portfolio_detail="[STUB] 초기 자본과 월 저축액을 기반으로 산출된 최적 포트폴리오입니다.",
        portfolio_composition=PortfolioComposition(
            cash_pct=30,
            stocks_etf_pct=50,
            bonds_pct=20,
        ),
    )
