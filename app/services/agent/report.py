from app.schemas.agent import (
    ExpenseCategory,
    Portfolios,
    RecommendedRebalanceRatio,
    ReportRequest,
    ReportResponse,
)


async def generate_report(request: ReportRequest) -> ReportResponse:
    # TODO: LangGraph 연동
    return ReportResponse(
        monthly_change="[STUB] 전월 대비 자산 변동 요약이 여기에 표시됩니다.",
        portfolios=Portfolios(
            stock_change=0.0,
            bond_change=0.0,
            cash_change=0.0,
        ),
        portfolio_comment="[STUB] 포트폴리오 변동 코멘트가 여기에 표시됩니다.",
        expense_categories=[
            ExpenseCategory(category="식비", value=0),
        ],
        expense_analysis="[STUB] 지출 분석 내용이 여기에 표시됩니다.",
        recommended_rebalance_ratio=RecommendedRebalanceRatio(
            stock_ratio=0,
            bond_ratio=0,
            cash_ratio=0,
        ),
        next_month_guideline="[STUB] 다음 달 가이드라인이 여기에 표시됩니다.",
    )
