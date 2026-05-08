from app.schemas.agent import KeyMetrics, ReportRequest, ReportResponse


async def generate_report(request: ReportRequest) -> ReportResponse:
    # TODO: LangGraph 연동
    total_spending = sum(t.amount for t in request.transactions)
    return ReportResponse(
        user_id=request.user_id,
        year_month=request.year_month,
        report_markdown="[STUB] ## 월간 리포트\n\n분석 결과가 여기에 표시됩니다.",
        key_metrics=KeyMetrics(
            total_spending=total_spending,
            savings_rate=0.0,
            top_categories=[],
        ),
    )
