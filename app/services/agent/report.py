from datetime import datetime, timezone

from app.schemas.report import ReportRequest, ReportResponse


async def generate_report(request: ReportRequest) -> ReportResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return ReportResponse(
        created_at=now,
        trend_comment="[STUB] 전월 대비 자산 변화 추이 코멘트가 여기에 표시됩니다.",
        market_condition="[STUB] 현재 시장 상황 텍스트가 여기에 표시됩니다.",
        hover_description="[STUB] 소비 패턴에 대한 설명이 여기에 표시됩니다.",
        performance_status="OUTPERFORM",
        performance_comment="[STUB] 성과 관련 코멘트가 여기에 표시됩니다.",
    )
