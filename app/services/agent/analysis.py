from app.schemas.agent import AnalysisRequest, AnalysisResponse


async def build_analysis(request: AnalysisRequest) -> AnalysisResponse:
    # TODO: LangGraph 연동 (AI 추천 비율과 사용자 조정 비율 비교 심층 분석)
    u = request.portfolio_user
    return AnalysisResponse(
        analysis_report=(
            f"[STUB] 사용자 포트폴리오 — 현금 {u.cash_ratio}% / 주식 {u.stock_ratio}% / 채권 {u.bond_ratio}%. "
            "AI 권장 비율과 비교 분석이 필요합니다."
        ),
        summary="[STUB] 목표 달성 가능성 분석 중입니다.",
    )
