import pytest
import numpy as np
import asyncio
from core.agents.specialized.monte_carlo_risk_agent import MonteCarloRiskAgent
from core.schemas.v23_5_schema import SimulationEngine

# Mock config
CONFIG = {"agent_id": "risk_agent_test"}

@pytest.mark.asyncio
async def test_gbm_simulation():
    """Test standard Geometric Brownian Motion simulation."""
    agent = MonteCarloRiskAgent(CONFIG)

    # Run GBM
    result = await agent.execute(
        current_ebitda=100.0,
        interest_expense=10.0,
        capex_maintenance=5.0,
        model_type="GBM",
        iterations=1000, # Lower for test speed
        volatility=0.2,
        time_horizon=1.0,
        dt_steps=12
    )

    assert isinstance(result, SimulationEngine)
    # Check that we got a percentage string
    assert "%" in result.monte_carlo_default_prob
    # With these numbers (EBITDA 100 >> Expense 15), default prob should be low (likely 0.0%)
    assert result.monte_carlo_default_prob != "N/A"
    assert "Base Case (GBM)" in result.quantum_scenarios[0].scenario_name

@pytest.mark.asyncio
async def test_heston_simulation():
    """Test Heston Stochastic Volatility simulation."""
    agent = MonteCarloRiskAgent(CONFIG)

    result = await agent.execute(
        current_ebitda=100.0,
        interest_expense=10.0,
        capex_maintenance=5.0,
        model_type="Heston",
        iterations=1000,
        heston_kappa=1.0,
        heston_theta=0.04,
        heston_xi=0.1,
        heston_v0=0.04,
        drift=0.02
    )

    assert isinstance(result, SimulationEngine)
    assert "Base Case (Heston)" in result.quantum_scenarios[0].scenario_name
    assert result.monte_carlo_default_prob != "N/A"

@pytest.mark.asyncio
async def test_ou_simulation():
    """Test Ornstein-Uhlenbeck simulation."""
    agent = MonteCarloRiskAgent(CONFIG)

    result = await agent.execute(
        current_ebitda=50.0,
        interest_expense=10.0,
        capex_maintenance=5.0,
        model_type="OU",
        iterations=1000,
        ou_mu=50.0,
        ou_theta=0.5,
        volatility=5.0
    )

    assert isinstance(result, SimulationEngine)
    assert "Base Case (OU)" in result.quantum_scenarios[0].scenario_name
    assert result.monte_carlo_default_prob != "N/A"

@pytest.mark.asyncio
async def test_validation_failure():
    """Test Pydantic validation failure (missing required field)."""
    agent = MonteCarloRiskAgent(CONFIG)

    # Missing current_ebitda
    result = await agent.execute(
        interest_expense=10.0
    )

    # The agent returns a fallback SimulationEngine on validation error
    assert result.monte_carlo_default_prob == "N/A"
    assert result.trading_dynamics.liquidity_risk == "High"

@pytest.mark.asyncio
async def test_zero_ebitda():
    """Test edge case where EBITDA is 0."""
    agent = MonteCarloRiskAgent(CONFIG)

    result = await agent.execute(
        current_ebitda=0.0,
        interest_expense=10.0
    )

    assert result.monte_carlo_default_prob == "100.0%"

@pytest.mark.asyncio
async def test_high_default_probability():
    """Test a scenario where default is likely."""
    agent = MonteCarloRiskAgent(CONFIG)

    # Expenses (110) > EBITDA (100) -> Immediate Default ?
    # Or close enough that volatility triggers it often.
    # Logic: Distress = Int + Capex
    # If Distress = 110, EBITDA = 100.
    # Initial EBITDA is below threshold?
    # Our logic: min(path) < threshold.
    # If S0 < threshold, min(path) <= S0 < threshold -> 100% default.

    result = await agent.execute(
        current_ebitda=100.0,
        interest_expense=100.0,
        capex_maintenance=10.0, # Total threshold 110
        model_type="GBM",
        iterations=1000,
        volatility=0.1
    )

    assert result.monte_carlo_default_prob == "100.0%"
