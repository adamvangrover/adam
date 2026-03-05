import pytest
from core.agents.insider_activity_agent import InsiderActivityAgent
from core.schemas.agent_schema import AgentInput

@pytest.fixture
def insider_activity_agent():
    config = {
        "mock_mode": True
    }
    return InsiderActivityAgent(config)

@pytest.mark.asyncio
async def test_insider_activity_agent_standard_mode(insider_activity_agent):
    """Test standard execution with AgentInput."""
    input_data = AgentInput(query="NVDA")
    result = await insider_activity_agent.execute(input_data)
    
    assert result.confidence == 0.85
    assert result.metadata["status"] == "success"
    assert "sentiment_score" in result.metadata
    
    # Check mock data overrides
    details = result.metadata["details"]
    assert details["buy_volume"] == 1500000.0
    assert details["sell_volume"] == 500000.0
    assert details["cluster_buys"] is True
    assert details["officer_buys"] == 2
    
    # 1.5 / 2.0 = 0.75 buy ratio
    # score = 0.5 + ((0.75 - 0.3) / 0.7) * 0.5 = 0.5 + (0.45 / 0.7) * 0.5 = 0.5 + 0.3214 = 0.8214
    # cluster buys (+0.15) -> 0.9714
    # officer buys (+0.1) -> 1.0714 -> max(1.0) -> 1.0
    assert result.metadata["sentiment_score"] == 1.0

@pytest.mark.asyncio
async def test_insider_activity_agent_legacy_mode(insider_activity_agent):
    """Test legacy execution with dict input."""
    input_data = {"query": "AAPL"}
    result = await insider_activity_agent.execute(input_data)
    
    assert result["status"] == "success"
    assert "sentiment_score" in result
    assert result["sentiment_score"] == 1.0

@pytest.mark.asyncio
async def test_insider_activity_agent_no_input(insider_activity_agent):
    """Test execution with no input."""
    result = await insider_activity_agent.execute()
    
    assert result["status"] == "success"
    assert "sentiment_score" in result
    assert result["sentiment_score"] == 1.0
