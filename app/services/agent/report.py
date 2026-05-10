from app.schemas.agent import PortfolioComposition, ReportRequest, ReportResponse


async def generate_report(request: ReportRequest) -> ReportResponse:
    # TODO: LangGraph 연동
    return ReportResponse(
        ai_comment="[STUB] 이전 포트폴리오 대비 현재 포트폴리오 분석 결과가 여기에 표시됩니다.",
        new_ratio=PortfolioComposition(
            cash_pct=0.0,
            stocks_etf_pct=0.0,
            bonds_pct=0.0,
        ),
    )
