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
async def test_market_risk_ewma_cvar():
    agent = MarketRiskAgent(config={})
    # Generate random walk
    prices = [100.0]
    np.random.seed(42)
    for _ in range(50):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))

    data = {"price_history": prices, "portfolio_value": 100000}
    result = await agent.execute(data)

    assert "volatility_ewma_annualized" in result
    assert "cvar_95_pct" in result
    assert result["cvar_95_pct"] > result["var_95_pct"] # CVaR usually > VaR (absolute loss pct)
    assert "monetary_risk" in result

@pytest.mark.asyncio
async def test_credit_risk_merton():
    agent = CreditRiskAgent(config={})
    data = {
        "working_capital": 200, "total_assets": 1000,
        "market_value_equity": 600, "total_liabilities": 400,
        "volatility": 0.20, "risk_free_rate": 0.04
    }
    result = await agent.execute(data)

    assert result["merton_distance_to_default"] > 3.0 # Safe
    assert result["estimated_credit_rating"] in ["AAA", "AA+", "AA", "AA-"]

@pytest.mark.asyncio
async def test_liquidity_risk_lcr():
    agent = LiquidityRiskAgent(config={})
    data = {
        "financial_data": {
            "current_assets": 100, "current_liabilities": 50,
            "cash_and_equivalents": 20, "government_bonds": 30, # HQLA = 50
            "net_cash_outflows_30d": 25 # LCR = 2.0
        },
        "market_data": {
            "avg_daily_volume": 1000000, "bid_ask_spread": 0.01, "price": 100
        }
    }
    result = await agent.execute(data)
    assert result["financial_liquidity"]["liquidity_coverage_ratio_approx"] == 2.0

@pytest.mark.asyncio
async def test_operational_risk_lda():
    agent = OperationalRiskAgent(config={})
    data = {
        "compliance_incidents": 2, # High frequency
        "revenue": 1000000
    }
    result = await agent.execute(data)

    assert "lda_simulation" in result
    assert result["lda_simulation"]["op_var_99_9"] > result["lda_simulation"]["expected_annual_loss"]

@pytest.mark.asyncio
async def test_geopolitical_risk_contagion():
    agent = GeopoliticalRiskAgent(config={})
    # Set up contagion scenario
    # US affects EU. If US risk is high, EU gets contagion.
    input_data = {
        "US": {"political_stability": 10, "trade_conflict": 90, "social_unrest": 80}, # High Risk ~ 90
        "EU": {"political_stability": 90} # Low Risk ~ 10
    }
    result = await agent.execute(input_data)

    assert result["regional_assessments"]["US"]["level"] == "High"
    # Check if EU has contagion
    contagion_eu = result["contagion_risks"].get("EU", {})
    # US is connected to EU in default map
    # Check if sources list contains US
    sources_str = str(contagion_eu.get("sources", []))
    assert "US" in sources_str

@pytest.mark.asyncio
async def test_industry_risk_porters():
    agent = IndustryRiskAgent(config={})
    input_data = {
        "sector": "Custom",
        "porter_forces": {
            "rivalry": 10, "new_entrants": 10, "substitutes": 10,
            "supplier_power": 10, "buyer_power": 10
        },
        "cyclicality_beta": 2.0
    }
    result = await agent.execute(input_data)
    assert result["industry_risk_score"] == 100.0

@pytest.mark.asyncio
async def test_economic_risk_misery():
    agent = EconomicRiskAgent(config={})
    data = {"gdp_growth": 0.01, "inflation_rate": 0.10, "unemployment_rate": 0.08} # Misery 18%
    result = await agent.execute(data)

    assert result["metrics"]["misery_classification"] == "Severe"
    assert result["economic_risk_score"] >= 70

@pytest.mark.asyncio
async def test_volatility_risk_garch():
    agent = VolatilityRiskAgent(config={})
    data = {
        "vix_index": 20,
        "current_return": -0.05, # Big drop
        "previous_volatility": 0.20
    }
    result = await agent.execute(data)

    assert "garch_forecast" in result
    assert result["garch_forecast"]["trend"] == "Rising" # -5% return should spike vol

@pytest.mark.asyncio
async def test_currency_risk_portfolio_var():
    agent = CurrencyRiskAgent(config={})
    data = {
        "portfolio_value": 1000000,
        "exposures": {"EUR": 500000},
        "hedging_ratio": 0.0
    }
    result = await agent.execute(data)

    metrics = result["exposure_metrics"]
    assert metrics["exposure_pct_of_portfolio"] == 0.5
    assert metrics["portfolio_fx_var_pct"] > 0.0 # Should be non-zero
