import pytest
from core.agents.system_health_agent import SystemHealthAgent

@pytest.mark.asyncio
async def test_health_metrics():
    agent = SystemHealthAgent({"agent_id": "test_agent"})
    result = await agent.execute()
    assert result["status"] == "healthy"
    assert "metrics" in result

@pytest.mark.asyncio
async def test_ping():
    agent = SystemHealthAgent({"agent_id": "test_agent"})
    assert agent.ping() == "pong"
