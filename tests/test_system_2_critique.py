import pytest
from core.agents.orchestrators.credit_memo_orchestrator import CreditMemoOrchestrator
from core.engine.icat import ICATOutput
from core.engine.consensus_engine import ConsensusEngine

@pytest.fixture
def orchestrator():
    # Initialize with mock mode to avoid live calls
    return CreditMemoOrchestrator(mock_library={}, mode="mock")

@pytest.fixture
def mock_icat_output():
    class MockAssumptions:
        terminal_valuation_method = "Gordon Growth"
        discount_rate = 0.10
        growth_rate = 0.05
        terminal_growth_rate = 0.02
        tax_rate = 0.21
        industry_multiples = {"EV/EBITDA": 12.0}
        initial_cash_flow = 100

    return ICATOutput(
        ticker="MOCK",
        scenario_name="Base",
        revenue=[100, 110, 120, 130, 140],
        ebitda=[20, 22, 24, 26, 28],
        fcf=[10, 11, 12, 13, 14],
        assumptions=MockAssumptions(),
        credit_metrics={
            "pd_1yr": 0.01,
            "lgd": 0.4,
            "ltv": 0.5,
            "dscr": 2.0,
            "avg_dscr": 2.1,
            "interest_coverage": 3.0,
            "net_leverage": 4.0
        },
        valuation_metrics={
            "enterprise_value": 1500,
            "equity_value": 1000
        },
        environment={"gdp_growth": 0.02, "interest_rate": 0.05, "inflation": 0.02, "market_sentiment": 0.5},
        generated_at="2025-01-01T00:00:00Z"
    )

def test_system_2_critique_alignment(orchestrator, mock_icat_output):
    # Setup aligned inputs
    risk = {
        "overall_risk_score": 0.2, # Low risk
        "risk_factors": {"market_risk": 0.1, "financial_risk": 0.2}
    }
    reg = {"compliance_report": "Clear. No major issues.", "analysis_results": []}
    valuation = {
        "dcf_base": {"npv": 1500},
        "dcf_bear": {"npv": 1200},
        "dcf_bull": {"npv": 1800},
        "current_price": 1400,
        "base_target_price": 1500,
        "base": {"intrinsic_share_price": 1500},
        "bull": {"intrinsic_share_price": 1800}
    }

    # Execute
    critique = orchestrator._system_2_critique(mock_icat_output, risk, reg, valuation)

    # Assert
    assert "critique_points" in critique
    assert "conviction_score" in critique
    assert critique["critique_points"] == ["Models are consistent across domains."]
    assert critique["conviction_score"] >= 0.8 # Should be high due to alignment

def test_system_2_critique_discrepancy(orchestrator, mock_icat_output):
    # Setup conflicting inputs (High risk, but positive valuation)
    risk = {
        "overall_risk_score": 0.85, # High risk
            "risk_factors": {"market_risk": 0.9, "financial_risk": 0.8},
            "risk_quant_metrics": {"PD": 0.08}
    }
    reg = {"compliance_report": "Severe violations detected.", "analysis_results": [{"violated_rules": ["KYC", "AML"]}]}
    valuation = {
        "dcf_base": {"npv": 2500}, # Very optimistic valuation
        "dcf_bear": {"npv": 2000},
        "dcf_bull": {"npv": 3000},
        "current_price": 1000,
        "base_target_price": 2500,
        "base": {"intrinsic_share_price": 2500},
        "bull": {"intrinsic_share_price": 3000}
    }

    # Execute
    critique = orchestrator._system_2_critique(mock_icat_output, risk, reg, valuation)

    # Assert
    assert "critique_points" in critique

    # Check for specific critiques
    finding_texts = " ".join(critique["critique_points"])
    assert "High Default Risk conflicts with Aggressive Bull Case Valuation" in finding_texts

    # The consensus should reflect the negative signals heavily
    assert critique["conviction_score"] < 0.9
