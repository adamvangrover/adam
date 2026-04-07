import pytest
from fastapi.testclient import TestClient
from server.sentinel_api import app

client = TestClient(app)

def test_evaluate_credit_success():
    response = client.post(
        "/api/v1/evaluate_credit",
        json={
            "metrics_data": {"pd": 0.01, "lgd": 0.5, "ead": 1000.0},
            "conviction": 0.95,
            "npv_fees": 1000.0,
            "sigma": 0.15,
            "jurisdiction": "USA",
            "prompt": "Test API",
            "context": {}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "decision_state" in data
    assert data["decision_state"]["routing_path"] == "AUTOMATED"
    assert data["decision_state"]["requires_step_up"] is False
    assert "safe_prompt" in data

def test_evaluate_credit_hitl():
    response = client.post(
        "/api/v1/evaluate_credit",
        json={
            "metrics_data": {"pd": 0.5, "lgd": 0.5, "ead": 1000.0},
            "conviction": 0.95,
            "npv_fees": 1000.0,
            "sigma": 0.15,
            "jurisdiction": "USA",
            "prompt": "Test step up",
            "context": {}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision_state"]["routing_path"] == "HITL_TIER_3"
    assert data["decision_state"]["requires_step_up"] is True
