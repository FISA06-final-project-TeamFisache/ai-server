from datetime import datetime, timezone

from app.schemas.portfolio import ProfileRequest, ProfileResponse


async def analyze_profile(request: ProfileRequest) -> ProfileResponse:
    # TODO: LangGraph 연동
    now = datetime.now(timezone.utc)
    return ProfileResponse(
        created_at=now,
        expense_comment="[STUB] 카테고리별 지출 분석 결과가 여기에 표시됩니다.",
        invest_comment="[STUB] 투자 성향 분석 결과가 여기에 표시됩니다.",
        savings_comment="[STUB] 저축 목록 분석 결과가 여기에 표시됩니다.",
    )
