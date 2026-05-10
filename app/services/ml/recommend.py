from app.schemas.ml import RecommendRequest, RecommendResponse, Product


async def recommend_products(request: RecommendRequest) -> RecommendResponse:
    # TODO: LangGraph 연동
    return RecommendResponse(
        ai_comment="[STUB] 거래 내역 분석 기반 상품 추천 결과가 여기에 표시됩니다.",
        products=[],
    )
