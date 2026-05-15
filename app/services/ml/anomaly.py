from collections import deque
from datetime import datetime, timedelta

from app.core.config import settings
from app.schemas.kafka import KafkaAnomalyAlert, KafkaTransactionMessage
from app.schemas.ml import AnomalyRequest, AnomalyResponse


class _RapidTransactionDetector:
    """단시간 내 다수 거래 rule-based 탐지기."""

    def __init__(self, window_seconds: int, max_count: int) -> None:
        self._window = timedelta(seconds=window_seconds)
        self._max_count = max_count
        # asset_number → 최근 거래 시각 목록
        self._history: dict[str, deque[datetime]] = {}

    def check(self, asset_number: str, transaction_at: datetime) -> tuple[bool, int]:
        """이상 여부와 윈도우 내 거래 건수를 반환한다."""
        history = self._history.setdefault(asset_number, deque())
        cutoff = transaction_at - self._window

        while history and history[0] < cutoff:
            history.popleft()

        history.append(transaction_at)
        count = len(history)
        return count > self._max_count, count


_detector = _RapidTransactionDetector(
    window_seconds=settings.kafka_anomaly_window_seconds,
    max_count=settings.kafka_anomaly_max_count,
)


def detect_from_kafka(transaction: KafkaTransactionMessage) -> KafkaAnomalyAlert | None:
    is_anomaly, count = _detector.check(transaction.asset_number, transaction.transactionAt)
    if not is_anomaly:
        return None

    window_minutes = settings.kafka_anomaly_window_seconds // 60
    content = (
        f"단시간 내 다수 거래 감지: "
        f"{window_minutes}분 이내 {count}건 거래 발생 "
        f"(기준: {settings.kafka_anomaly_max_count}건 초과)"
    )
    return KafkaAnomalyAlert(
        asset_number=transaction.asset_number,
        content=content,
        created_at=datetime.now(),
    )


# 기존 API 엔드포인트용 함수 (유지)
async def detect_anomaly(request: AnomalyRequest) -> AnomalyResponse:
    # TODO: LangGraph 연동
    return AnomalyResponse(
        title="[STUB] 거래 이상 감지 결과",
        content="[STUB] 거래 승인 일시, 금액, 유형, 가맹점, 취소 여부 등을 종합 분석한 결과가 여기에 표시됩니다.",
    )
