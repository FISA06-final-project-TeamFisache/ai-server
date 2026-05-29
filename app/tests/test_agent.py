from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.services.ml.model_loader.load_all_models"):
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_report_stub(client):
    payload = {
        "user_id": "00000000-0000-0000-0000-000000000001",
        "year": 2024,
        "month": 1,
    }
    response = client.post("/agent/report", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "monthly_change" in data
    assert "portfolios" in data
    assert "portfolio_comment" in data
    assert "expense_categories" in data
    assert "expense_analysis" in data
    assert "recommended_rebalance_ratio" in data
    assert "next_month_guideline" in data


def test_portfolio_profile_stub(client):
    payload = {"user_id": "00000000-0000-0000-0000-000000000001"}
    response = client.post("/agent/portfolio/profile", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "created_at" in data
    assert "user_preference" in data
    assert "finance_type" in data["user_preference"]
    assert "comment" in data["user_preference"]
    assert "analysis_result" in data
    assert "audit_log" in data


def test_portfolio_recommend_stub(client):
    payload = {
        "user_id": "00000000-0000-0000-0000-000000000001",
        "user_preference": "안정형",
        "analysis_result": "지출이 많고 저축이 부족합니다.",
    }
    response = client.post("/agent/portfolio/recommend", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "created_at" in data
    assert "salary_rebalance" in data
    assert isinstance(data["salary_rebalance"], list)
    assert "portfolio_recommend" in data
    assert "audit_log" in data


def test_event_input_stub(client):
    payload = {
        "user_id": "00000000-0000-0000-0000-000000000001",
        "user_input": "내년 유럽 여행을 위해 1000만원을 모으고 싶습니다.",
    }
    response = client.post("/agent/event/input", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "created_at" in data
    assert "salary_rebalance" in data
    assert isinstance(data["salary_rebalance"], list)
    assert "portfolio_recommend" in data
    assert "audit_log" in data
