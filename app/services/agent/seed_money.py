from app.schemas.agent import SeedMoneyRequest, SeedMoneyResponse


async def build_seed_money(_request: SeedMoneyRequest) -> SeedMoneyResponse:
    # TODO: LangGraph 연동 (PorTI 성향 및 deadline 기반 종잣돈 전략 분석)
    return SeedMoneyResponse()
