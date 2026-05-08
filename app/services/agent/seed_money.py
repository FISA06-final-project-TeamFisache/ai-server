from app.schemas.agent import (
    PortfolioComposition,
    SeedMoneyRequest,
    SeedMoneyResponse,
)


async def build_seed_money(request: SeedMoneyRequest) -> SeedMoneyResponse:
    # TODO: LangGraph 연동
    required_monthly_savings = max(
        0.0,
        (request.target_amount - request.initial_capital) / request.duration_months,
    )
    return SeedMoneyResponse(
        user_id=request.user_id,
        target_amount=request.target_amount,
        required_monthly_savings=round(required_monthly_savings, 2),
        annual_return_rate=5.0,
        portfolio_composition=PortfolioComposition(
            cash_pct=30.0,
            stocks_etf_pct=50.0,
            bonds_pct=20.0,
        ),
    )
