from app.schemas.agent import RecommendedRebalanceRatio, ReportRequest, ReportResponse


async def generate_report(request: ReportRequest) -> ReportResponse:
    # TODO: LangGraph 연동
    r = request
    report = f"""# 포트폴리오 월간 보고서

## 자산 현황
| 항목 | 값 |
|---|---|
| 현재 총 자산 | {r.total_assets:,.0f}원 |
| 전월 대비 자산 증감 | {r.asset_change_prev_month:+,.0f}원 |

## 자산별 수익률
| 자산 유형 | 수익률 |
|---|---|
| 주식 | {r.portfolio_change_rate.stock_rate * 100:+.2f}% |
| 채권 | {r.portfolio_change_rate.bond_rate * 100:+.2f}% |
| 현금성 자산 | {r.portfolio_change_rate.cash_rate * 100:+.2f}% |

## 기존 포트폴리오 비중
| 자산 유형 | 비중 |
|---|---|
| 주식 | {r.existing_portfolio_ratio.stock_ratio:.1f}% |
| 채권 | {r.existing_portfolio_ratio.bond_ratio:.1f}% |
| 현금성 자산 | {r.existing_portfolio_ratio.cash_ratio:.1f}% |
"""
    return ReportResponse(
        portfolio_report=report,
        recommended_rebalance_ratio=RecommendedRebalanceRatio(
            stock_ratio=0,
            bond_ratio=0,
            cash_ratio=0,
        ),
        recommendation_comment="[STUB] 리밸런싱 추천 코멘트가 여기에 표시됩니다.",
    )
