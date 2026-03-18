import pytest

from core.agents.options_flow_agent import OptionsFlowAgent


@pytest.mark.asyncio
async def test_options_flow_bullish():
    """Test bullish options flow (P/C ratio < 1.0)"""
    config = {
        'mock_mode': True,
        'mock_put_call_ratio': 0.7,
        'mock_unusual_volume': False
    }
    agent = OptionsFlowAgent(config)
    result = await agent.execute()

    # 1.0 - ((0.7 - 0.5) / 1.0) = 1.0 - 0.2 = 0.8
    assert result["sentiment_score"] == pytest.approx(0.8)
    assert result["details"]["put_call_ratio"] == 0.7
    assert result["details"]["unusual_volume"] is False

@pytest.mark.asyncio
async def test_options_flow_bearish():
    """Test bearish options flow (P/C ratio > 1.0)"""
    config = {
        'mock_mode': True,
        'mock_put_call_ratio': 1.3,
        'mock_unusual_volume': False
    }
    agent = OptionsFlowAgent(config)
    result = await agent.execute()

    # 1.0 - ((1.3 - 0.5) / 1.0) = 1.0 - 0.8 = 0.2
    assert result["sentiment_score"] == pytest.approx(0.2)
    assert result["details"]["put_call_ratio"] == 1.3
    assert result["details"]["unusual_volume"] is False

@pytest.mark.asyncio
async def test_options_flow_neutral():
    """Test neutral options flow (P/C ratio = 1.0)"""
    config = {
        'mock_mode': True,
        'mock_put_call_ratio': 1.0,
        'mock_unusual_volume': False
    }
    agent = OptionsFlowAgent(config)
    result = await agent.execute()

    # 1.0 - ((1.0 - 0.5) / 1.0) = 1.0 - 0.5 = 0.5
    assert result["sentiment_score"] == pytest.approx(0.5)
    assert result["details"]["put_call_ratio"] == 1.0
    assert result["details"]["unusual_volume"] is False

@pytest.mark.asyncio
async def test_options_flow_bullish_unusual_volume():
    """Test bullish options flow with unusual volume amplification"""
    config = {
        'mock_mode': True,
        'mock_put_call_ratio': 0.7,
        'mock_unusual_volume': True
    }
    agent = OptionsFlowAgent(config)
    result = await agent.execute()

    # Base score = 0.8. Unusual volume adds 0.1 -> 0.9
    assert result["sentiment_score"] == pytest.approx(0.9)

@pytest.mark.asyncio
async def test_options_flow_bearish_unusual_volume():
    """Test bearish options flow with unusual volume amplification"""
    config = {
        'mock_mode': True,
        'mock_put_call_ratio': 1.3,
        'mock_unusual_volume': True
    }
    agent = OptionsFlowAgent(config)
    result = await agent.execute()

    # Base score = 0.2. Unusual volume subtracts 0.1 -> 0.1
    assert result["sentiment_score"] == pytest.approx(0.1)

@pytest.mark.asyncio
async def test_options_flow_bounds():
    """Test sentiment score bounding between 0.0 and 1.0"""
    # Extremely bullish
    config_bull = {
        'mock_mode': True,
        'mock_put_call_ratio': 0.1,
        'mock_unusual_volume': True
    }
    agent_bull = OptionsFlowAgent(config_bull)
    result_bull = await agent_bull.execute()

    # Base score = 1.0 - ((0.1 - 0.5)/1.0) = 1.4. Bounded to 1.0. Amplified to 1.1. Bounded to 1.0.
    assert result_bull["sentiment_score"] == 1.0

    # Extremely bearish
    config_bear = {
        'mock_mode': True,
        'mock_put_call_ratio': 2.0,
        'mock_unusual_volume': True
    }
    agent_bear = OptionsFlowAgent(config_bear)
    result_bear = await agent_bear.execute()

    # Base score = 1.0 - ((2.0 - 0.5)/1.0) = -0.5. Bounded to 0.0. Amplified to -0.1. Bounded to 0.0.
    assert result_bear["sentiment_score"] == 0.0
