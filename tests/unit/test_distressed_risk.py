import pytest
from unittest.mock import MagicMock, patch
from core.agents.risk_assessment_agent import RiskAssessmentAgent

@pytest.fixture
def risk_agent():
    config = {"knowledge_base_path": "mock_kb.json"}
    with patch("builtins.open", new_callable=MagicMock), \
         patch("json.load", return_value={"risk_weights": {}}):
        agent = RiskAssessmentAgent(config)
        return agent

def test_leveraged_finance_risk(risk_agent):
    company_name = "LevCo"
    financial_data = {
        "ebitda": 100.0,
        "total_debt": 600.0, # 6x leverage (High)
        "fcf": 40.0,
        "interest_expense": 50.0
    }
    deal_structure = {"covenant_cushion": 0.05} # Tight cushion

    result = risk_agent.assess_leveraged_finance_risk(company_name, financial_data, deal_structure)

    assert "risk_factors" in result
    # Leverage 6x -> score should be high
    assert result["risk_factors"]["leverage_risk"] > 0.5
    # DSCR < 1.0 (40/50 = 0.8) -> score should be high
    assert result["risk_factors"]["cash_flow_risk"] > 0.8

def test_distressed_debt_risk(risk_agent):
    company_name = "DistressCo"
    financial_data = {
        "total_assets": 1000.0,
        "secured_debt": 400.0,
        "unsecured_debt": 400.0
    }
    # Liquidation Value = 500 (50% of 1000)
    # Remaining for unsecured = 500 - 400 = 100
    # Unsecured Recovery = 100 / 400 = 25%

    market_data = {
        "bond_price": 0.40 # Trading at 40c
    }
    # Model says 25c, Market says 40c -> Overvalued -> High Risk

    result = risk_agent.assess_distressed_debt_risk(company_name, financial_data, market_data)

    assert result["recovery_analysis"]["unsecured_recovery"] == 0.25
    assert result["risk_factors"]["valuation_risk"] > 0.0
