import pytest
import numpy as np
from core.agents.market_risk_agent import MarketRiskAgent
from core.agents.credit_risk_agent import CreditRiskAgent
from core.agents.liquidity_risk_agent import LiquidityRiskAgent
from core.agents.operational_risk_agent import OperationalRiskAgent
from core.agents.geopolitical_risk_agent import GeopoliticalRiskAgent
from core.agents.industry_risk_agent import IndustryRiskAgent
from core.agents.economic_risk_agent import EconomicRiskAgent
from core.agents.volatility_risk_agent import VolatilityRiskAgent
from core.agents.currency_risk_agent import CurrencyRiskAgent

@pytest.mark.asyncio
async def test_market_risk_agent():
    agent = MarketRiskAgent(config={"agent_id": "test_market_risk"})

    # Mock data: Simple uptrend
    price_history = [100, 101, 102, 103, 104, 105]
    benchmark = [100, 102, 104, 106, 108, 110]

    data = {"price_history": price_history, "benchmark_history": benchmark}
    result = await agent.execute(data)

    assert "volatility_annualized" in result
    assert "var_95_pct" in result # Updated key
    assert "beta" in result
    assert isinstance(result["beta"], float)

@pytest.mark.asyncio
async def test_credit_risk_agent():
    agent = CreditRiskAgent(config={"agent_id": "test_credit_risk"})

    # Mock data for healthy company
    data = {
        "working_capital": 500,
        "retained_earnings": 300,
        "ebit": 200,
        "total_assets": 1000,
        "market_value_equity": 600,
        "total_liabilities": 400,
        "sales": 1500
    }

    result = await agent.execute(data)
    assert "z_score" in result
    assert result["zone"] == "Safe"

@pytest.mark.asyncio
async def test_liquidity_risk_agent():
    agent = LiquidityRiskAgent(config={"agent_id": "test_liquidity_risk"})

    data = {
        "financial_data": {
            "current_assets": 200,
            "current_liabilities": 100,
            "inventory": 50,
            "cash_and_equivalents": 50 # Added for LCR
        },
        "market_data": {
            "avg_daily_volume": 5000000,
            "bid_ask_spread": 0.01,
            "price": 100
        }
    }

    result = await agent.execute(data)
    assert result["financial_liquidity"]["current_ratio"] == 2.0
    assert result["market_liquidity"]["assessment"] == "Liquid"

@pytest.mark.asyncio
async def test_operational_risk_agent():
    agent = OperationalRiskAgent(config={"agent_id": "test_ops_risk"})

    data = {
        "years_in_business": 10,
        "employee_turnover_rate": 0.05,
        "compliance_incidents": 0,
        "management_tenure": 8
    }

    result = await agent.execute(data)
    assert result["risk_level"] == "Low"
    assert result["operational_risk_score"] < 20

@pytest.mark.asyncio
async def test_geopolitical_risk_agent():
    agent = GeopoliticalRiskAgent(config={"agent_id": "test_geo_risk"})

    regions = ["US", "EU"]
    result = await agent.execute(regions)

    assert "regional_assessments" in result
    assert "US" in result["regional_assessments"]
    assert "global_risk_index" in result

@pytest.mark.asyncio
async def test_industry_risk_agent():
    agent = IndustryRiskAgent(config={"agent_id": "test_industry_risk"})

    result = await agent.execute("Technology")
    assert result["risk_level"] == "High"
    # assert "High Disruption" in result["key_factors"] # Removed key_factors
    assert "forces_breakdown" in result

@pytest.mark.asyncio
async def test_economic_risk_agent():
    agent = EconomicRiskAgent(config={"agent_id": "test_econ_risk"})

    # Stagflation scenario
    data = {
        "gdp_growth": 0.005,
        "inflation_rate": 0.06,
        "yield_curve_inversion": True
    }

    result = await agent.execute(data)
    assert result["macro_scenario"] == "Stagflation" # Updated key
    assert result["risk_level"] in ["High", "Medium"]

@pytest.mark.asyncio
async def test_volatility_risk_agent():
    agent = VolatilityRiskAgent(config={"agent_id": "test_vol_risk"})

    data = {"vix_index": 35}
    result = await agent.execute(data)
    assert result["regime"] == "High Volatility (Stress)"

@pytest.mark.asyncio
async def test_currency_risk_agent():
    agent = CurrencyRiskAgent(config={"agent_id": "test_fx_risk"})

    # High exposure
    # Converted old input style to new style
    # foreign_revenue + foreign_debt -> exposures
    # Assume Portfolio Value 1000
    data = {
        "portfolio_value": 1000,
        "exposures": {"EUR": 400, "JPY": 200}, # 60% exposure
        "hedging_ratio": 0.1
    }

    result = await agent.execute(data)
    # result["net_exposure_pct"] is now in result["exposure_metrics"]["exposure_pct_of_portfolio"]
    assert result["exposure_metrics"]["exposure_pct_of_portfolio"] > 0.5
    assert result["risk_level"] == "High"
