import pytest
import asyncio
from core.agents.specialized.defi_liquidity_agent import DeFiLiquidityAgent

@pytest.mark.asyncio
async def test_impermanent_loss_calculation():
    config = {"agent_id": "test_defi_agent"}
    agent = DeFiLiquidityAgent(config)

    # 50% price drop
    # Ratio = 0.5
    # IL = 2 * sqrt(0.5) / (1 + 0.5) - 1
    # IL = 2 * 0.7071 / 1.5 - 1 = 1.4142 / 1.5 - 1 = 0.9428 - 1 = -0.0572 (5.72%)
    il = agent.calculate_impermanent_loss(100, 50)
    assert 5.7 < il < 5.8

    # No change
    il = agent.calculate_impermanent_loss(100, 100)
    assert il == 0.0

    # 2x price increase
    # Ratio = 2
    # IL = 2 * 1.4142 / 3 - 1 = 2.8284 / 3 - 1 = 0.9428 - 1 = -0.0572 (5.72%)
    il = agent.calculate_impermanent_loss(100, 200)
    assert 5.7 < il < 5.8

@pytest.mark.asyncio
async def test_execute_flow():
    config = {"agent_id": "test_defi_agent"}
    agent = DeFiLiquidityAgent(config)

    result = await agent.execute(pool_address="0x123", initial_price=100.0, current_price=200.0)

    assert result["pool_address"] == "0x123"
    assert "liquidity_metrics" in result
    assert result["liquidity_metrics"]["mock"] is True
    assert result["impermanent_loss_pct"] > 5.0
    assert "recommendation" in result
    assert "yield_analysis" in result
    assert result["recommendation"] == "WITHDRAW (High IL)"

@pytest.mark.asyncio
async def test_execute_flow_deposit():
    config = {"agent_id": "test_defi_agent"}
    agent = DeFiLiquidityAgent(config)

    # Low IL (stable), Good Yield (mocked logic checks liquidity score)
    # Mock data: reserve0=1000000, reserve1=500 -> score = sqrt(5e8) = ~22360
    # Logic: if score < 100000 -> APY = 0.15
    # So it should recommend DEPOSIT

    result = await agent.execute(pool_address="0x123", initial_price=100.0, current_price=100.0) # 0 IL

    assert result["impermanent_loss_pct"] == 0.0
    assert result["yield_analysis"]["estimated_apy"] == 0.15
    assert result["recommendation"] == "DEPOSIT (Good Yield)"
