from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.agent import (
    AnalysisResult,
    AuditLog,
    ProfileRequest,
    ProfileResponse,
    UserPreference,
)


async def analyze_profile(request: ProfileRequest) -> ProfileResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return ProfileResponse(
        created_at=now,
        user_preference=UserPreference(
            finance_type="[STUB] 안정형",
            comment="[STUB] 안정적인 투자를 선호하며 원금 보전을 우선시하는 성향입니다.",
        ),
        analysis_result=AnalysisResult(
            expense_comment="[STUB] 지출 패턴 분석 결과가 여기에 표시됩니다.",
            invest_comment="[STUB] 투자 성향 분석 결과가 여기에 표시됩니다.",
            savings_comment="[STUB] 저축 현황 분석 결과가 여기에 표시됩니다.",
        ),
        audit_log=AuditLog(
            log_id=str(uuid4()),
            user_id=str(request.user_id),
            endpoint="/agent/portfolio/profile",
            requested_at=now,
        ),
    )
