import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

_USER_ID = str(uuid4())


@pytest.fixture
def challenge_payload():
    return {
        "user_id": _USER_ID,
        "category_expense": [
            {
                "amount": 5_000,
                "category": "카페",
                "sender_name": "스타벅스",
                "transaction_at": "2025-06-01T09:00:00",
            }
        ],
        "stock_themes": ["AI", "반도체"],
    }


@pytest.fixture
def adjust_payload():
    return {"user_id": _USER_ID, "feedback": "더 쉽게 조정해주세요"}


@pytest.fixture
def reward_payload():
    return {"user_id": _USER_ID}


@pytest.fixture
def nag_payload():
    return {
        "user_id": _USER_ID,
        "title": "카페 지출 줄이기",
        "category": "카페",
        "challenge_type": "COUNT",
        "target": 10,
        "current": 5,
        "progress_pct": 50,
    }


# ── POST /mini_challenge ──────────────────────────────────────────────────────

def test_mini_challenge_happy_path(client, challenge_payload):
    from app.schemas.mini_challenge import ChallengeType, MiniChallengeResponse
    mock_result = MiniChallengeResponse(
        created_at=datetime.now(timezone.utc),
        title="카페 3회 줄이기",
        description="이번 달 카페 방문 3회를 줄여보세요",
        category="카페",
        target=3,
        challenge_type=ChallengeType.COUNT,
        estimated_saving=15_000,
        ticker="005930.KS",
        ticker_name="삼성전자",
        challenge_sub_type="REDUCE_COUNT",
    )
    with patch("app.routers.mini_challenge.propose_mini_challenge", new=AsyncMock(return_value=mock_result)):
        response = client.post("/mini_challenge", json=challenge_payload)
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert "challenge_type" in data


def test_mini_challenge_timeout(client, challenge_payload):
    with patch("app.routers.mini_challenge.propose_mini_challenge", new=AsyncMock(side_effect=asyncio.TimeoutError)):
        response = client.post("/mini_challenge", json=challenge_payload)
    assert response.status_code == 504


def test_mini_challenge_invalid_request(client):
    response = client.post("/mini_challenge", json={"stock_themes": ["AI"]})
    assert response.status_code == 422


# ── POST /mini_challenge/adjust ───────────────────────────────────────────────

def test_adjust_happy_path(client, adjust_payload):
    from app.schemas.mini_challenge import AdjustResponse, ChallengeType
    mock_result = AdjustResponse(
        created_at=datetime.now(timezone.utc),
        title="카페 2회 줄이기",
        challenge_type=ChallengeType.COUNT,
        target=2,
        category="카페",
        description="조금 더 쉬운 목표로 조정했습니다",
        ticker="005930.KS",
        ticker_name="삼성전자",
        estimated_saving=10_000,
        challenge_sub_type="REDUCE_COUNT",
    )
    with patch("app.routers.mini_challenge.adjust_challenge", new=AsyncMock(return_value=mock_result)):
        response = client.post("/mini_challenge/adjust", json=adjust_payload)
    assert response.status_code == 200
    data = response.json()
    assert "title" in data


def test_adjust_timeout(client, adjust_payload):
    with patch("app.routers.mini_challenge.adjust_challenge", new=AsyncMock(side_effect=asyncio.TimeoutError)):
        response = client.post("/mini_challenge/adjust", json=adjust_payload)
    assert response.status_code == 504


def test_adjust_invalid_request(client):
    response = client.post("/mini_challenge/adjust", json={})
    assert response.status_code == 422


# ── POST /mini_challenge/reward ───────────────────────────────────────────────

_MOCK_SESSION = {
    "proposals": [
        {"ticker": "005930.KS", "estimated_saving": 50_000, "challenge_sub_type": "REDUCE_COUNT"}
    ]
}
_MOCK_PROPOSAL = {"ticker": "005930.KS", "estimated_saving": 50_000, "challenge_sub_type": "REDUCE_COUNT"}
_MOCK_PRICES = [("삼성전자", "005930.KS", 78_000)]


def test_reward_happy_path(client, reward_payload):
    with (
        patch("app.routers.mini_challenge.get_session", new=AsyncMock(return_value=_MOCK_SESSION)),
        patch("app.routers.mini_challenge.get_last_proposal", return_value=_MOCK_PROPOSAL),
        patch("app.routers.mini_challenge.get_all_prices", new=AsyncMock(return_value=_MOCK_PRICES)),
        patch("app.routers.mini_challenge.send_log", new=AsyncMock()),
        patch("app.routers.mini_challenge.delete_session", new=AsyncMock()),
    ):
        response = client.post("/mini_challenge/reward", json=reward_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "005930.KS"
    assert "shares" in data
    assert "current_price" in data


def test_reward_no_active_challenge(client, reward_payload):
    with (
        patch("app.routers.mini_challenge.get_session", new=AsyncMock(return_value={})),
        patch("app.routers.mini_challenge.get_last_proposal", return_value=None),
    ):
        response = client.post("/mini_challenge/reward", json=reward_payload)
    assert response.status_code == 404


def test_reward_price_timeout(client, reward_payload):
    with (
        patch("app.routers.mini_challenge.get_session", new=AsyncMock(return_value=_MOCK_SESSION)),
        patch("app.routers.mini_challenge.get_last_proposal", return_value=_MOCK_PROPOSAL),
        patch("app.routers.mini_challenge.get_all_prices", new=AsyncMock(side_effect=asyncio.TimeoutError)),
    ):
        response = client.post("/mini_challenge/reward", json=reward_payload)
    assert response.status_code == 504


def test_reward_invalid_request(client):
    response = client.post("/mini_challenge/reward", json={})
    assert response.status_code == 422


# ── POST /mini_challenge/nag ──────────────────────────────────────────────────

def test_nag_happy_path(client, nag_payload):
    from app.schemas.mini_challenge import NagResponse
    mock_result = NagResponse(
        created_at=datetime.now(timezone.utc),
        nag_message="벌써 50%나 달성했어요! 조금만 더 힘내세요.",
    )
    with patch("app.routers.mini_challenge.generate_nag", new=AsyncMock(return_value=mock_result)):
        response = client.post("/mini_challenge/nag", json=nag_payload)
    assert response.status_code == 200
    data = response.json()
    assert "nag_message" in data


def test_nag_timeout(client, nag_payload):
    with patch("app.routers.mini_challenge.generate_nag", new=AsyncMock(side_effect=asyncio.TimeoutError)):
        response = client.post("/mini_challenge/nag", json=nag_payload)
    assert response.status_code == 504


def test_nag_invalid_request(client):
    response = client.post("/mini_challenge/nag", json={"user_id": str(uuid4())})
    assert response.status_code == 422
