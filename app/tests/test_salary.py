import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

_USER_ID = str(uuid4())
_ASSET_ID = str(uuid4())


@pytest.fixture
def salary_payload():
    return {
        "user_id": _USER_ID,
        "salary_diff": 100_000,
        "category_expense": [{"name": "식비", "expense": 300_000}],
        "portfolio_items": [
            {
                "asset_id": _ASSET_ID,
                "account_purpose": "비상금",
                "amount": 1_000_000,
            }
        ],
        "flow_items": [
            {
                "title": "비상금 목표",
                "term": "12개월",
                "summary": "비상금 적립",
                "asset_id": _ASSET_ID,
                "amount": 500_000,
            }
        ],
    }


def test_salary_happy_path(client, salary_payload):
    from app.schemas.salary import FlowItem, PortfolioItem, SalaryResponse
    asset_id = uuid4()
    mock_result = SalaryResponse(
        created_at=datetime.now(timezone.utc),
        portfolio_items=[
            PortfolioItem(asset_id=asset_id, account_purpose="비상금", amount=1_100_000)
        ],
        flow_items=[
            FlowItem(
                title="비상금 목표",
                term="12개월",
                summary="비상금 적립",
                asset_id=asset_id,
                amount=550_000,
            )
        ],
        rebalance_comment="월급이 올라 비상금을 10만원 더 배분했습니다.",
    )
    with patch("app.routers.salary.analyze_salary_rebalance", new=AsyncMock(return_value=mock_result)):
        response = client.post("/salary", json=salary_payload)
    assert response.status_code == 201
    data = response.json()
    assert "portfolio_items" in data
    assert "rebalance_comment" in data


def test_salary_timeout(client, salary_payload):
    with patch("app.routers.salary.analyze_salary_rebalance", new=AsyncMock(side_effect=asyncio.TimeoutError)):
        response = client.post("/salary", json=salary_payload)
    assert response.status_code == 504


def test_salary_invalid_request(client):
    response = client.post("/salary", json={"salary_diff": 100_000})
    assert response.status_code == 422
