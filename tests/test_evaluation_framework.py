import pytest
import sys
import os

# Ensure core is in path
sys.path.append(os.getcwd())

from core.evaluation.red_team import ZombieFactory
from core.evaluation.judge import AuditorAgent
from core.evaluation.symbolic import SymbolicVerifier
from core.vertical_risk_agent.agents.supervisor import critique_node

def test_zombie_factory():
    """Test that ZombieFactory generates high-risk data."""
    state = ZombieFactory.generate_zombie_state()
    assert state["balance_sheet"]["total_debt"] == 400_000_000
    assert state["income_statement"]["consolidated_ebitda"] == 30_000_000
    # Leverage ~13.3x
    assert state["covenants"][0]["threshold"] == 4.5

def test_auditor_agent_logic_check():
    """Test that Auditor catches hallucinated leverage."""
    auditor = AuditorAgent()
    state = ZombieFactory.generate_zombie_state()

    # Mock Analysis that claims leverage is low (Optimism Bias)
    # True leverage is ~13.3x
    state["quant_analysis"] = "The company is healthy. Leverage (Gross): 2.0x. EBITDA: 30,000,000.00"

    logs = auditor.evaluate(state)

    # We expect a Logic Density failure
    logic_log = next((l for l in logs if l["category"] == "Logic Density"), None)
    assert logic_log is not None
    assert logic_log["score"] <= 3
    assert "Calculated Gross Leverage" in logic_log["reasoning"]

def test_symbolic_verifier_flags():
    """Test that SymbolicVerifier catches ontology violations."""
    verifier = SymbolicVerifier()

    # Trigger 1: Term Loan B Seniority Mismatch
    text = "We have a Term Loan B which is Senior Secured and very safe."
    flags = verifier.verify(text)
    assert any("Term Loan B" in f and "Subordinated" in f for f in flags)

    # Trigger 2: Parent Corp Structure
    text = "The Parent Corporation is an Operating Company generating cash."
    flags = verifier.verify(text)
    assert any("Parent" in f and "HoldingCompany" in f for f in flags)

def test_critique_node_integration():
    """Test full integration in supervisor node."""
    state = ZombieFactory.generate_zombie_state()
    # Mock inputs
    state["quant_analysis"] = "Leverage (Gross): 2.0x. EBITDA: 30,000,000.00"
    state["legal_analysis"] = "The Term Loan B is Senior Secured."

    result = critique_node(state)

    # Check Audit Logs
    assert "audit_logs" in result
    assert len(result["audit_logs"]) > 0
    # Should catch the leverage lie
    logic_log = next((l for l in result["audit_logs"] if l["category"] == "Logic Density"), None)
    assert logic_log["score"] <= 3

    # Check Verification Flags
    assert "verification_flags" in result
    assert len(result["verification_flags"]) > 0
    # Should catch the Senior vs Subordinated lie
    assert any("Term Loan B" in f for f in result["verification_flags"])

    # Check Messages
    messages = result["messages"]
    # Messages should contain warning
    assert any("Symbolic Verification Failed" in m for m in messages)
