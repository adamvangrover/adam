import pytest
from core.governance.constitution import Constitution

@pytest.fixture
def constitution():
    return Constitution()

def test_initialization(constitution):
    """Verify that the constitution initializes with default principles."""
    assert len(constitution.principles) >= 4
    assert constitution.principles[0].id == "DO_NO_HARM"

def test_check_action_allow(constitution):
    """Verify that a safe action is allowed."""
    action = "EXECUTE_TRADE"
    context = {"amount": 1000, "risk_score": 0.1}
    assert constitution.check_action(action, context) is True

    log = constitution.get_audit_log()
    assert len(log) == 1
    assert log[0]["allowed"] is True
    assert log[0]["action"] == action

def test_check_action_reject_high_risk(constitution):
    """Verify that a high-risk action is rejected."""
    action = "EXECUTE_TRADE"
    context = {"amount": 1000000, "risk_score": 0.9}
    assert constitution.check_action(action, context) is False

    log = constitution.get_audit_log()
    assert len(log) == 1
    assert log[0]["allowed"] is False
    assert log[0]["reason"] == "Risk score exceeds threshold."

def test_check_action_reject_illegal(constitution):
    """Verify that an illegal action is rejected."""
    action = "ILLEGAL_TRADE"
    context = {"amount": 100, "risk_score": 0.1}
    assert constitution.check_action(action, context) is False

    log = constitution.get_audit_log()
    assert len(log) == 1
    assert log[0]["allowed"] is False
    assert log[0]["reason"] == "Action appears illegal."

def test_audit_log_accumulation(constitution):
    """Verify that multiple checks accumulate in the audit log."""
    constitution.check_action("A1", {"risk_score": 0.1})
    constitution.check_action("A2", {"risk_score": 0.9})

    log = constitution.get_audit_log()
    assert len(log) == 2
    assert log[0]["allowed"] is True
    assert log[1]["allowed"] is False
