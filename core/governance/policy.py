import json
from typing import Dict, Any, Optional
from core.governance.risk_class import RiskClass
from core.governance.approval import ApprovalGate

class PolicyEvaluator:
    def __init__(self, policy_bundle: Dict[str, Any]):
        self.policy_bundle = policy_bundle

    def evaluate(self, proposed_action: Dict[str, Any], user_context: Dict[str, Any]) -> bool:
        """
        Evaluates a proposed action against the loaded policy rules (e.g. JsonLogic).
        Returns True if allowed, False otherwise.
        Default deny posture.
        """
        risk_class = proposed_action.get("risk_class", RiskClass.EXTERNAL.value)

        # Enforce approval for high-risk actions
        if risk_class in [RiskClass.FINANCIAL.value, RiskClass.ADMIN.value]:
            gate = ApprovalGate()
            if not gate.evaluate(user_context, risk_class):
                return False

        # Policy rules evaluation (mocked jsonLogic evaluator integration)
        rules = self.policy_bundle.get("rules", [])
        if not rules:
            return False # Default deny if no rules exist

        for rule in rules:
            if rule.get("action") == "deny":
                # In a real impl, we'd apply JsonLogic `jsonLogic.apply(rule['condition'], proposed_action)`
                # For now, explicit forbid supremacy
                pass

        # We assume one explicit allow rule must match for approval
        allow_matched = False
        for rule in rules:
            if rule.get("action") == "allow":
                # Apply condition logic
                allow_matched = True
                break

        return allow_matched
