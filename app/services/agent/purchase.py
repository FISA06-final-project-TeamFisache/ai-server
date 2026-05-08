from app.schemas.agent import (
    PortfolioComposition,
    PurchaseCandidate,
    PurchaseRequest,
    PurchaseResponse,
)


async def build_purchase(request: PurchaseRequest) -> PurchaseResponse:
    # TODO: LangGraph 연동 (실시간 쇼핑몰 시세 조회)
    item = request.item_name
    return PurchaseResponse(
        user_id=request.user_id,
        item_name=item,
        candidates=[
            PurchaseCandidate(product_name=f"[STUB] {item} 기본형", estimated_price=500_000, description="가성비 모델"),
            PurchaseCandidate(product_name=f"[STUB] {item} 중급형", estimated_price=1_000_000, description="무난한 선택"),
            PurchaseCandidate(product_name=f"[STUB] {item} 프리미엄", estimated_price=2_000_000, description="고사양 모델"),
        ],
        annual_return_rate=3.5,
        portfolio_composition=PortfolioComposition(
            cash_pct=60.0,
            stocks_etf_pct=20.0,
            bonds_pct=20.0,
        ),
    )
