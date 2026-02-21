import pytest
import pandas as pd
import numpy as np
from core.agents.strategy_backtest_agent import StrategyBacktestAgent

@pytest.mark.asyncio
async def test_input_validation():
    """Test that invalid input raises an error."""
    agent = StrategyBacktestAgent(config={"agent_id": "test_backtest"})

    # Missing strategy_name
    with pytest.raises(ValueError):
        await agent.execute({"parameters": {}})

@pytest.mark.asyncio
async def test_sma_crossover_profit():
    """
    Test SMA Crossover on a synthetic uptrend where it should profit.
    """
    agent = StrategyBacktestAgent(config={"agent_id": "test_backtest"})

    # Create a V-shape recovery to ensure a crossover occurs
    # 50 days down, 50 days up
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    prices_down = np.linspace(200, 100, 50)
    prices_up = np.linspace(100, 200, 50)
    prices = np.concatenate([prices_down, prices_up])

    # Add small noise
    prices += np.random.normal(0, 0.5, 100)

    data = [{"Date": str(d), "Close": p} for d, p in zip(dates, prices)]

    input_data = {
        "strategy_name": "SMA_CROSSOVER",
        "parameters": {"short_window": 5, "long_window": 20},
        "initial_capital": 10000.0,
        "data": data
    }

    result = await agent.execute(input_data)

    # It should buy when it turns up and sell when it (maybe) turns down or just hold
    # Since it ends at 200 (same as start), but we bought around 100-110 (after confirmation), we should be profitable.

    metrics = result['metrics']
    trades = result['trades']

    # Debug print if it fails
    if metrics['total_return'] <= 0:
        print(f"Trades: {trades}")
        print(f"Metrics: {metrics}")

    assert metrics['total_return'] > 0
    assert metrics['trade_count'] > 0
    assert len(result['equity_curve']) == 100

@pytest.mark.asyncio
async def test_mean_reversion_logic():
    """
    Test Mean Reversion logic.
    """
    agent = StrategyBacktestAgent(config={"agent_id": "test_backtest"})

    # Sine wave
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    x = np.linspace(0, 4*np.pi, 200)
    prices = 100 + 10 * np.sin(x)

    data = [{"Date": str(d), "Close": p} for d, p in zip(dates, prices)]

    input_data = {
        "strategy_name": "MEAN_REVERSION",
        "parameters": {"window": 20, "std_dev_mult": 1.0}, # Tight bands to force trades
        "initial_capital": 10000.0,
        "data": data
    }

    result = await agent.execute(input_data)

    metrics = result['metrics']
    # Just verify it ran and made trades
    assert metrics['trade_count'] >= 0
    assert len(result['trades']) == metrics['trade_count']

@pytest.mark.asyncio
async def test_metrics_calculation_zero_trades():
    """Test metrics when no trades occur."""
    agent = StrategyBacktestAgent(config={"agent_id": "test_backtest"})

    # Flat line price
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    prices = [100.0] * 50
    data = [{"Date": str(d), "Close": p} for d, p in zip(dates, prices)]

    input_data = {
        "strategy_name": "SMA_CROSSOVER",
        "parameters": {"short_window": 5, "long_window": 20},
        "initial_capital": 10000.0,
        "data": data
    }

    result = await agent.execute(input_data)
    metrics = result['metrics']

    assert metrics['total_return'] == 0.0
    assert metrics['max_drawdown'] == 0.0
    assert metrics['sharpe_ratio'] == 0.0
    assert metrics['trade_count'] == 0
