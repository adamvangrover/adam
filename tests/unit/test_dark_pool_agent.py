import pytest
from core.agents.specialized.dark_pool_agent import DarkPoolAgent, DarkPoolAgentInput, DarkPoolAgentOutput

@pytest.fixture
def agent():
    return DarkPoolAgent(config={"agent_id": "test_dark_pool_agent", "anomaly_threshold": 0.10})

@pytest.mark.asyncio
async def test_normal_activity(agent):
    result = await agent.execute(
        total_volume=1000000.0,
        dark_pool_volume=400000.0,
        average_dark_pool_ratio=0.40
    )

    assert "answer" in result
    assert result["is_anomaly"] is False
    assert result["dark_pool_ratio"] == 0.40
    assert result["anomaly_score"] == 0.0
    assert "normal ranges" in result["answer"]

@pytest.mark.asyncio
async def test_high_dark_pool_activity(agent):
    result = await agent.execute(
        total_volume=1000000.0,
        dark_pool_volume=550000.0,  # 55%
        average_dark_pool_ratio=0.40
    )

    assert result["is_anomaly"] is True
    assert result["dark_pool_ratio"] == 0.55
    assert result["anomaly_score"] == pytest.approx(0.15)
    assert "High dark pool activity detected" in result["answer"]

@pytest.mark.asyncio
async def test_low_dark_pool_activity(agent):
    result = await agent.execute(
        total_volume=1000000.0,
        dark_pool_volume=200000.0,  # 20%
        average_dark_pool_ratio=0.40
    )

    assert result["is_anomaly"] is True
    assert result["dark_pool_ratio"] == 0.20
    assert result["anomaly_score"] == pytest.approx(-0.20)
    assert "Unusually low dark pool activity detected" in result["answer"]

@pytest.mark.asyncio
async def test_zero_total_volume_raises_error(agent):
    with pytest.raises(ValueError, match="Total volume must be greater than zero"):
        await agent.execute(
            total_volume=0.0,
            dark_pool_volume=0.0,
            average_dark_pool_ratio=0.40
        )
