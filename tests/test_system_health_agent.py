import pytest
from core.schemas.agent_schema import AgentInput
from core.agents.system_health_agent import SystemHealthAgent

@pytest.mark.asyncio
async def test_health_metrics():
    agent = SystemHealthAgent({"agent_id": "test_agent"})
    input_data = AgentInput(query="health check")
    result = await agent.execute(input_data)
    assert result.metadata["status"] == "healthy"
    assert "metrics" in result.metadata

@pytest.mark.asyncio
async def test_ping():
    agent = SystemHealthAgent({"agent_id": "test_agent"})
    assert agent.ping() == "pong"
