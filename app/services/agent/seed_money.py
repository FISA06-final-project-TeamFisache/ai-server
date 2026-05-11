from app.schemas.agent import SeedMoneyRequest, SeedMoneyResponse


async def build_seed_money(request: SeedMoneyRequest) -> SeedMoneyResponse:
    # TODO: LangGraph 연동
    required_monthly_savings = max(
        0.0,
        (request.target_amount - request.initial_capital) / request.duration_months,
    )
    return SeedMoneyResponse(
        target_amount=request.target_amount,
        required_monthly_savings=round(required_monthly_savings, 2),
    )
