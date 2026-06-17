import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

_USER_ID = str(uuid4())


@pytest.fixture
def report_payload():
    return {
        "user_id": _USER_ID,
        "year": 2025,
        "month": 6,
        "mini_challenges": None,
        "asset_snapshots": [
            {
                "snapshot_at": "2025-06-01T00:00:00",
                "total_amount": 5_000_000,
                "savings_amount": 3_000_000,
                "invest_amount": 2_000_000,
            }
        ],
        "transaction_log": [
            {
                "amount": 50_000,
                "category": "식비",
                "sender_name": "스타벅스",
                "transaction_at": "2025-06-01T10:00:00",
            }
        ],
    }


def test_report_happy_path(client, report_payload):
    from app.schemas.report import ReportResponse
    mock_result = ReportResponse(
        created_at=datetime.now(timezone.utc),
        trend_comment="지출이 전월 대비 줄었습니다.",
        challenge_comment="챌린지를 잘 수행했습니다.",
        market_condition="미국 증시 강세",
        hover_description=[],
        guideline="현재 소비 패턴을 유지하세요.",
    )
    with patch("app.routers.report.generate_report", new=AsyncMock(return_value=mock_result)):
        response = client.post("/report", json=report_payload)
    assert response.status_code == 201
    data = response.json()
    assert "trend_comment" in data
    assert "market_condition" in data
    assert "guideline" in data


def test_report_timeout(client, report_payload):
    with patch("app.routers.report.generate_report", new=AsyncMock(side_effect=asyncio.TimeoutError)):
        response = client.post("/report", json=report_payload)
    assert response.status_code == 504


def test_report_invalid_request(client):
    response = client.post("/report", json={"user_id": _USER_ID})
    assert response.status_code == 422
