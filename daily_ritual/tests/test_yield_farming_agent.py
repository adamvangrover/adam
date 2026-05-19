import pytest
from core.schemas.agent_schema import AgentInput
from core.agents.yield_farming_agent import YieldFarmingAgent

@pytest.mark.asyncio
async def test_yield_farming_agent_low_risk():
    agent = YieldFarmingAgent(config={'llm': {'provider': 'mock', 'model': 'mock'}})
    input_data = AgentInput(query="Find me a stable yield", context={})
    result = await agent.execute(input_data)

    assert result.metadata["recommended_pool"] == "Curve 3pool"
    assert result.metadata["risk_level"] == "Low"

@pytest.mark.asyncio
async def test_yield_farming_agent_high_risk():
    agent = YieldFarmingAgent(config={'llm': {'provider': 'mock', 'model': 'mock'}})
    input_data = AgentInput(query="Find me a high risk yield", context={})
    result = await agent.execute(input_data)

    assert result.metadata["recommended_pool"] == "Uniswap V3 ETH/USDC (Narrow Range)"
    assert result.metadata["risk_level"] == "High"