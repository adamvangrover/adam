import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

# Adjust path if necessary
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.v30_architecture.python_intelligence.agents.fundamental_analyst import FundamentalAnalyst, FundamentalInput

@pytest.mark.asyncio
async def test_fundamental_analyst_initialization():
    agent = FundamentalAnalyst()
    assert agent.name == "FundamentalAnalyst-V30"
    assert agent.role == "fundamental_analysis"
    assert len(agent.universe) > 0

@pytest.mark.asyncio
async def test_execute_method():
    agent = FundamentalAnalyst()

    # Mock emit to prevent network errors if run() was called (though we call execute directly)
    agent.emit = AsyncMock()

    ticker = "TEST_TICKER"
    input_data = FundamentalInput(ticker=ticker)

    # Execute
    result = await agent.execute(input_data)

    # Validation
    assert result.ticker == ticker
    assert isinstance(result.intrinsic_value, float)
    assert isinstance(result.distress_score, float)
    assert isinstance(result.quality_score, int)
    assert 0 <= result.quality_score <= 9
    assert isinstance(result.risks, list)
    assert isinstance(result.growth_drivers, list)
    assert isinstance(result.report_summary, str)
    assert "Fundamental Analysis for TEST_TICKER" in result.report_summary

def test_calculate_altman_z():
    agent = FundamentalAnalyst()
    # Mock data for a strong company
    data = {
        "total_assets": 1000.0,
        "working_capital": 400.0,      # A = 0.4 -> 1.2*0.4 = 0.48
        "retained_earnings": 300.0,    # B = 0.3 -> 1.4*0.3 = 0.42
        "ebit": 200.0,                 # C = 0.2 -> 3.3*0.2 = 0.66
        "market_cap": 2000.0,          # D = 4.0 -> 0.6*4.0 = 2.4
        "total_liabilities": 500.0,
        "revenue": 1000.0,             # E = 1.0 -> 1.0*1.0 = 1.0
        "equity": 500.0,
        "net_income": 150.0
    }
    # Expected Z = 0.48 + 0.42 + 0.66 + 2.4 + 1.0 = 4.96
    z = agent._calculate_altman_z(data)
    assert z > 4.9
    assert z < 5.0

def test_calculate_pietroski_f():
    agent = FundamentalAnalyst()
    # Mock data for a decent company
    data = {
        "net_income": 100.0,        # +1
        "working_capital": 50.0,    # +1
        "total_liabilities": 400.0,
        "total_assets": 1000.0,     # Leverage 0.4 < 0.5 -> +1
        "revenue": 600.0            # Turnover 0.6 > 0.5 -> +1
    }
    # Base score should be 4, plus random noise (0-5)
    f = agent._calculate_pietroski_f(data)
    assert f >= 4
    assert f <= 9

def test_analyze_text():
    agent = FundamentalAnalyst()
    text = "We are worried about regulatory changes and geopolitical tension."
    risks, drivers = agent._analyze_text(text)

    assert "Regulatory Scrutiny" in risks
    assert "Geopolitical Tension" in risks
    assert "AI Adoption" in drivers # Default
