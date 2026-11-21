import pytest
import asyncio
from core.agents.fraud_detection_agent import FraudDetectionAgent

@pytest.mark.asyncio
async def test_fraud_detection():
    config = {"agent_id": "test_fraud_agent"}
    agent = FraudDetectionAgent(config)

    # Test Anomalies
    data = {
        "revenue": 1000,
        "cash_flow": 50, # < 10%
        "expenses": 0,
        "growth_rate": 0.60
    }

    result = await agent.execute(command="audit", data=data)

    assert result["status"] == "Audit Complete"
    assert len(result["anomalies"]) >= 2
    assert "Revenue high but operating cash flow dangerously low" in str(result["anomalies"])
    assert "Zero expenses reported" in str(result["anomalies"])
    assert "Unusual growth rate" in str(result["anomalies"])

@pytest.mark.asyncio
async def test_restatement():
    config = {"agent_id": "test_fraud_agent"}
    agent = FraudDetectionAgent(config)

    data = {
        "revenue": 1000,
        "expenses": 500,
        "net_income": 500
    }

    result = await agent.execute(command="restate", data=data)

    assert result["status"] == "Restatement Complete"
    restated = result["restated_data"]

    # Revenue * 0.85 = 850
    assert restated["revenue"] == 850.0
    # Expenses * 1.20 = 600
    assert restated["expenses"] == 600.0
    # Net Income = 850 - 600 = 250
    assert restated["net_income"] == 250.0
    assert restated["restatement_date"] == "2026-04-01"
