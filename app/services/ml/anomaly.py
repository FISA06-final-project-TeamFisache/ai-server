from app.schemas.ml import AnomalyRequest, AnomalyResponse


async def detect_anomaly(request: AnomalyRequest) -> AnomalyResponse:
    # TODO: LangGraph 연동
    return AnomalyResponse(
        title="[STUB] 거래 이상 감지 결과",
        content="[STUB] 거래 승인 일시, 금액, 유형, 가맹점, 취소 여부 등을 종합 분석한 결과가 여기에 표시됩니다.",
    )
