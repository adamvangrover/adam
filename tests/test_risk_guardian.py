import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from core.v30_architecture.python_intelligence.agents.risk_guardian import RiskGuardian

# Helper for async return
def async_return(result):
    f = asyncio.Future()
    f.set_result(result)
    return f

@pytest.fixture
def mock_market_data():
    # Create a 20-day DataFrame for 2 tickers + VIX
    # Use enough data to calculate metrics
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")

    # Random walk
    np.random.seed(42)
    returns1 = np.random.normal(0, 0.01, 50)
    price1 = 100 * (1 + returns1).cumprod()

    returns2 = np.random.normal(0, 0.02, 50)
    price2 = 20000 * (1 + returns2).cumprod()

    data = {
        "SPY": price1,
        "BTC-USD": price2,
    }
    df = pd.DataFrame(data, index=dates)
    return df

@pytest.mark.asyncio
async def test_risk_calculation(mock_market_data):
    """Test the math logic directly."""
    agent = RiskGuardian()
    agent.portfolio = ["SPY", "BTC-USD"]
    # 50/50 weights
    agent.weights = np.array([0.5, 0.5])

    # Calculate metrics
    metrics = agent._calculate_risk_metrics(mock_market_data)

    assert metrics is not None
    # Volatility should be positive
    assert metrics.volatility > 0
    # VaR should be positive (loss magnitude)
    assert metrics.portfolio_var_95 >= 0
    # Sharpe ratio should be calculated (float)
    assert isinstance(metrics.sharpe_ratio, float)
    # Correlation matrix should contain keys
    assert "SPY" in metrics.correlation_matrix
    assert "BTC-USD" in metrics.correlation_matrix

@pytest.mark.asyncio
async def test_risk_guardian_emit(mock_market_data):
    """Test the agent loop emitting data."""

    # Mock BaseAgent.emit to verify it's called
    with patch("core.v30_architecture.python_intelligence.agents.base_agent.BaseAgent.emit", new_callable=AsyncMock) as mock_emit:
        agent = RiskGuardian()
        agent.portfolio = ["SPY", "BTC-USD"]
        agent.weights = np.array([0.5, 0.5])

        # Mock _fetch_historical_data to return our dataframe immediately
        agent._fetch_historical_data = MagicMock(return_value=async_return(mock_market_data))

        # We will run the internal logic of `run` once manually to avoid infinite loop / sleep issues
        # Copying the logic from run() essentially:

        data = await agent._fetch_historical_data()
        metrics = agent._calculate_risk_metrics(data)

        if metrics:
            await agent.emit("risk_assessment", metrics.model_dump())

        assert mock_emit.called
        assert mock_emit.call_count == 1
        args, _ = mock_emit.call_args
        assert args[0] == "risk_assessment"
        payload = args[1]
        assert "portfolio_var_95" in payload
        assert "volatility" in payload
