import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

_USER_ID = str(uuid4())
_ASSET_ID = str(uuid4())


@pytest.fixture
def profile_payload():
    return {
        "user_id": _USER_ID,
        "category_expense": [{"name": "식비", "expense": 300_000}],
        "porti_type": "MODERATE",
        "porti_comment": "안정적인 투자",
        "assets_safe": 3_000_000,
        "assets_moderate": 2_000_000,
        "assets_risky": 1_000_000,
    }


@pytest.fixture
def rebalance_payload():
    return {
        "user_id": _USER_ID,
        "category_expense": [{"name": "식비", "expense": 300_000}],
        "porti_type": "MODERATE",
        "porti_comment": "안정적인 투자",
        "assets": [
            {
                "asset_type": "CHECKING",
                "account_name": "주거래통장",
                "asset_id": _ASSET_ID,
                "balance": 2_000_000,
            }
        ],
        "fixed_expense": 1_500_000,
        "salary": 3_000_000,
    }


@pytest.fixture
def asset_portfolio_payload():
    return {
        "user_id": _USER_ID,
        "invest_amount": 1_000_000,
        "interest": "장기 성장",
        "invest_interests": ["글로벌 주식", "채권"],
        "porti_type": "MODERATE",
        "porti_comment": "안정적인 투자",
        "invest_assets": [
            {
                "asset_type": "ETF",
                "account_name": "증권계좌",
                "asset_id": _ASSET_ID,
                "balance": 500_000,
            }
        ],
    }


# ── POST /portfolio/profile ───────────────────────────────────────────────────

def test_profile_happy_path(client, profile_payload):
    from app.schemas.portfolio import ProfileResponse
    mock_result = ProfileResponse(
        created_at=datetime.now(timezone.utc),
        expense_comment="소비 패턴이 안정적입니다.",
        invest_comment="중위험 포트폴리오를 권장합니다.",
    )
    with patch("app.routers.portfolio.analyze_profile", new=AsyncMock(return_value=mock_result)):
        response = client.post("/portfolio/profile", json=profile_payload)
    assert response.status_code == 201
    data = response.json()
    assert "expense_comment" in data
    assert "invest_comment" in data


def test_profile_timeout(client, profile_payload):
    with patch("app.routers.portfolio.analyze_profile", new=AsyncMock(side_effect=asyncio.TimeoutError)):
        response = client.post("/portfolio/profile", json=profile_payload)
    assert response.status_code == 504


def test_profile_invalid_request(client):
    response = client.post("/portfolio/profile", json={"porti_type": "MODERATE"})
    assert response.status_code == 422


# ── POST /portfolio/rebalance ─────────────────────────────────────────────────

def test_rebalance_happy_path(client, rebalance_payload):
    from app.schemas.portfolio import RebalanceResponse, SalaryRebalanceItem
    mock_result = RebalanceResponse(
        created_at=datetime.now(timezone.utc),
        invest_amount=500_000,
        reasoning="분산 투자 권장",
        salary_rebalance=[
            SalaryRebalanceItem(
                asset_id=uuid4(),
                account_purpose="비상금",
                amount=500_000,
                comment="비상금 유지",
            )
        ],
    )
    with patch("app.routers.portfolio.rebalance_salary", new=AsyncMock(return_value=mock_result)):
        response = client.post("/portfolio/rebalance", json=rebalance_payload)
    assert response.status_code == 201
    data = response.json()
    assert "salary_rebalance" in data
    assert "invest_amount" in data


def test_rebalance_timeout(client, rebalance_payload):
    with patch("app.routers.portfolio.rebalance_salary", new=AsyncMock(side_effect=asyncio.TimeoutError)):
        response = client.post("/portfolio/rebalance", json=rebalance_payload)
    assert response.status_code == 504


def test_rebalance_invalid_request(client):
    response = client.post("/portfolio/rebalance", json={"user_id": _USER_ID})
    assert response.status_code == 422


# ── POST /portfolio/asset-portfolio ──────────────────────────────────────────

def test_asset_portfolio_happy_path(client, asset_portfolio_payload):
    from app.schemas.portfolio import AssetPortfolioResponse
    mock_result = AssetPortfolioResponse(
        created_at=datetime.now(timezone.utc),
        investment_flows=[],
    )
    with patch("app.routers.portfolio.recommend_asset_portfolio", new=AsyncMock(return_value=mock_result)):
        response = client.post("/portfolio/asset-portfolio", json=asset_portfolio_payload)
    assert response.status_code == 201
    data = response.json()
    assert "investment_flows" in data


def test_asset_portfolio_timeout(client, asset_portfolio_payload):
    with patch("app.routers.portfolio.recommend_asset_portfolio", new=AsyncMock(side_effect=asyncio.TimeoutError)):
        response = client.post("/portfolio/asset-portfolio", json=asset_portfolio_payload)
    assert response.status_code == 504


def test_asset_portfolio_invalid_request(client):
    response = client.post("/portfolio/asset-portfolio", json={})
    assert response.status_code == 422
