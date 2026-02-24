import pytest
import sys
import os

# Add the repository root to sys.path to ensure modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.risk_assessment_agent import RiskAssessmentAgent
from core.agents.legal_agent import LegalAgent

# --- RiskAssessmentAgent Tests ---

def test_calculate_lgd():
    config = {}
    agent = RiskAssessmentAgent(config)

    # Test base cases
    assert agent.calculate_lgd("Senior Secured", 1.0) == 0.05  # Max collateral, min floor
    assert agent.calculate_lgd("Senior Secured", 0.0) == 0.35  # No collateral, base LGD

    # Test seniority levels
    assert agent.calculate_lgd("Senior Unsecured", 0.0) == 0.50
    assert agent.calculate_lgd("Subordinated", 0.0) == 0.75
    assert agent.calculate_lgd("Equity", 0.0) == 1.00

    # Test partial collateral
    # 50% coverage -> 0.35 * (1 - 0.5) = 0.175
    assert abs(agent.calculate_lgd("Senior Secured", 0.5) - 0.175) < 0.001

def test_calculate_rwa():
    config = {}
    agent = RiskAssessmentAgent(config)

    pd = 0.05 # 5%
    lgd = 0.40 # 40%
    ead = 1000.0

    # Formula: RW = 0.20 + (PD * 20) + (LGD * 1.5)
    # RW = 0.20 + (0.05 * 20) + (0.40 * 1.5)
    # RW = 0.20 + 1.0 + 0.6 = 1.8 (180%)

    expected_rwa = ead * 1.8
    assert abs(agent.calculate_rwa(pd, lgd, ead) - expected_rwa) < 0.01

def test_calculate_expected_return_raroc():
    config = {}
    agent = RiskAssessmentAgent(config)

    interest_rate = 0.10 # 10%
    pd = 0.02
    lgd = 0.50
    capital_cost = 0.10
    rwa_ratio = 1.5 # 150% risk weight

    # EL = 0.02 * 0.50 = 0.01 (1%)
    # Risk Adj Income = 0.10 - 0.01 = 0.09
    # Economic Capital = 1.5 * 0.08 = 0.12 (12%)
    # RAROC = (0.09 - (0.12 * 0.10)) / 0.12
    # RAROC = (0.09 - 0.012) / 0.12
    # RAROC = 0.078 / 0.12 = 0.65 (65%)

    raroc = agent.calculate_expected_return(interest_rate, pd, lgd, capital_cost, rwa_ratio)
    assert abs(raroc - 0.65) < 0.001

# --- LegalAgent Tests ---

def test_review_credit_agreement():
    agent = LegalAgent()

    doc_text = "This agreement includes a negative pledge clause and a change of control provision."
    result = agent.review_credit_agreement(doc_text)

    assert "Negative Pledge" in result["clauses_identified"]
    assert "Change of Control" in result["clauses_identified"]
    assert "Cross-Default" not in result["clauses_identified"]

def test_check_covenants():
    agent = LegalAgent()

    # Case 1: Pass
    financials_good = {"leverage_ratio": 3.0, "dscr": 1.5}
    result_good = agent.check_covenants(financials_good)
    assert result_good["covenant_status"] == "Pass"

    # Case 2: Breach Leverage
    financials_bad_lev = {"leverage_ratio": 5.0, "dscr": 1.5}
    result_bad_lev = agent.check_covenants(financials_bad_lev)
    assert result_bad_lev["covenant_status"] == "Breach"
    assert any("Leverage" in v for v in result_bad_lev["violations"])

    # Case 3: Breach DSCR
    financials_bad_dscr = {"leverage_ratio": 3.0, "dscr": 1.0}
    result_bad_dscr = agent.check_covenants(financials_bad_dscr)
    assert result_bad_dscr["covenant_status"] == "Breach"
    assert any("DSCR" in v for v in result_bad_dscr["violations"])

def test_detect_fraud_signals():
    agent = LegalAgent()

    # Clean doc
    doc_clean = "Annual report implies standard operations."
    fin_clean = {"revenue": 12345}
    res_clean = agent.detect_fraud_signals(doc_clean, fin_clean)
    assert res_clean["fraud_risk_level"] == "Low"

    # Dirty doc
    doc_dirty = "Auditors noted a material weakness in internal controls."
    fin_dirty = {"revenue": 100000} # Round number suspicious?
    res_dirty = agent.detect_fraud_signals(doc_dirty, fin_dirty)
    assert res_dirty["fraud_risk_level"] == "High"
    assert any("material weakness" in s for s in res_dirty["signals_detected"])

def test_suggest_restructuring_strategy():
    agent = LegalAgent()

    assert "forbearance" in agent.suggest_restructuring_strategy("High").lower()
    assert "waiver" in agent.suggest_restructuring_strategy("Medium").lower()
    assert "monitor" in agent.suggest_restructuring_strategy("Low").lower()
