import pytest
import asyncio
from core.v30_architecture.python_intelligence.agents.macro_economic_agent import MacroEconomicAgent

@pytest.mark.asyncio
async def test_macro_economic_agent_execute():
    agent = MacroEconomicAgent()

    # Test US Sovereign
    result_us = await agent.execute(sovereign="US")
    assert result_us["status"] == "success"
    assert result_us["agent"] == "MacroEconomicAgent"
    assert result_us["target"] == "US"

    # Validate the synthesis score is within bounds
    assert 0.0 <= result_us["macro_score"] <= 1.0

    # Validate the presence of all analytical pillars
    pillars = result_us["pillars"]
    assert "sovereign_risk" in pillars
    assert "yield_environment" in pillars
    assert "geopolitics" in pillars
    assert "structural_trends" in pillars

    # Validate specific logic inside US Sovereign risk
    assert pillars["sovereign_risk"]["credit_rating"] == "AA+"
    assert pillars["sovereign_risk"]["debt_to_gdp"] == 122.0

    # Validate tech cycle in structural trends
    assert "AI capital expenditure" in pillars["structural_trends"]["tech_phase"]

    # Test Emerging Markets
    result_em = await agent.execute(sovereign="EM_Index")
    assert result_em["pillars"]["sovereign_risk"]["score"] == 0.45
    assert "FX mismatch" in result_em["pillars"]["sovereign_risk"]["ability_to_pay"]
