from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.services.ml.model_loader.load_all_models"):
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_goal_portfolio_stub(client):
    payload = {
        "user_id": "user_001",
        "deadline": "2026-12-31",
        "initial_capital": 5000000,
        "monthly_seed": 500000,
        "target_amount": 30000000,
    }
    response = client.post("/agent/goal/portfolio", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "portfolio_detail" in data
    assert "portfolio_composition" in data


def test_rebalance_stub(client):
    payload = {
        "user_id": "user_001",
        "monthly_salary": 3000000,
        "current_allocations": [
            {"category": "식비", "amount": 500000},
            {"category": "저축", "amount": 1000000},
        ],
        "financial_goals": ["비상금 마련", "투자"],
    }
    response = client.post("/agent/rebalance", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user_001"


def test_report_stub(client):
    payload = {
        "user_id": "user_001",
        "year_month": "2024-01",
        "transactions": [
            {
                "transaction_id": "t1",
                "amount": 50000,
                "category": "식비",
                "timestamp": "2024-01-10T12:00:00Z",
            }
        ],
        "portfolio_snapshot": {},
    }
    response = client.post("/agent/report", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user_001"
    assert data["year_month"] == "2024-01"
