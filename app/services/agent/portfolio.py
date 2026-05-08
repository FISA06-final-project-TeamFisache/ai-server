from datetime import datetime, timezone

from app.schemas.agent import AssetAllocation, PortfolioRequest, PortfolioResponse


async def build_portfolio(request: PortfolioRequest) -> PortfolioResponse:
    # TODO: LangGraph 연동
    return PortfolioResponse(
        user_id=request.user_id,
        recommended_portfolio=[
            AssetAllocation(asset_type="cash", allocation_pct=40.0, reason="[STUB] 안정성 확보"),
            AssetAllocation(asset_type="stocks", allocation_pct=40.0, reason="[STUB] 수익성 추구"),
            AssetAllocation(asset_type="funds", allocation_pct=20.0, reason="[STUB] 분산 투자"),
        ],
        summary="[STUB] 포트폴리오 구성 결과입니다.",
        generated_at=datetime.now(timezone.utc),
    )
