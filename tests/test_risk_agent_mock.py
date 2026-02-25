import pytest
import asyncio
from unittest.mock import MagicMock, patch
from core.agents.risk_assessment_agent import RiskAssessmentAgent

# Mock dependencies
class MockKnowledgeBase:
    def get(self, key, default=None):
        return default or {}

@pytest.fixture
def agent():
    config = {
        "persona": "Risk Officer",
        "description": "Risk Assessment",
        "knowledge_base_path": "mock.json"
    }
    with patch("builtins.open", new_callable=MagicMock), \
         patch("json.load", return_value={"risk_weights": {"market_risk": 1.0}}):
        agent = RiskAssessmentAgent(config)
        return agent

@pytest.mark.asyncio
async def test_investment_risk(agent):
    target_data = {
        "company_name": "Test Corp",
        "financial_data": {"credit_rating": "AAA", "industry": "Technology"},
        "market_data": {"price_data": [100, 101, 102], "trading_volume": 1000000}
    }

    # Mock assess methods if needed, or let them run with the mocked KB
    # Here we let them run as they are pure logic mostly

    result = await agent.execute(target_data, risk_type="investment")

    assert "overall_risk_score" in result
    assert result["overall_risk_score"] >= 0.0
    assert result["overall_risk_score"] <= 1.0
    assert "risk_factors" in result
    # Updated to match actual key in agent
    assert "market_risk_score" in result["risk_factors"]

@pytest.mark.asyncio
async def test_cached_risk(agent):
    target_data = {
        "company_name": "Test Corp 2",
    }

    # First run
    with patch.object(agent, 'assess_investment_risk', return_value={"overall_risk_score": 0.5}) as mock_assess:
        await agent.execute(target_data, risk_type="investment")
        assert mock_assess.call_count == 1

        # Second run should hit cache
        await agent.execute(target_data, risk_type="investment")
        assert mock_assess.call_count == 1  # Still 1, so cache was used
