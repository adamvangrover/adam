from core.governance.policy import PolicyEvaluator
from core.governance.risk_class import RiskClass

def test_default_deny_policy():
    evaluator = PolicyEvaluator({"rules": []})
    proposed_action = {"risk_class": RiskClass.EXTERNAL.value}
    # No rules means it should default to deny
    assert not evaluator.evaluate(proposed_action, user_context={})

def test_approval_gate_enforcement():
    evaluator = PolicyEvaluator({
        "rules": [
            {"rule_id": "r1", "condition": {}, "action": "allow"}
        ]
    })
    proposed_action = {"risk_class": RiskClass.FINANCIAL.value}

    # Missing admin role and explicit approval
    assert not evaluator.evaluate(proposed_action, user_context={})

    # Has admin role and explicit approval
    assert evaluator.evaluate(proposed_action, user_context={"role": "admin", "explicit_approval": True})
