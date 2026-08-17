from typing import Dict, Any
from json_logic import jsonLogic
from src.pdil.authorization.capability_matrix import CapabilityRequest

class PolicyEngine:
    """
    Evaluates capability requests against policy bundles using jsonLogic.
    """
    def __init__(self, policies: Dict[str, Any]):
        self.policies = policies

    def evaluate(self, request: CapabilityRequest, context: Dict[str, Any]) -> bool:
        """
        Evaluates a capability request using the specified policy bundle.
        Returns True if the policy permits the action, False otherwise.
        """
        policy_bundle = self.policies.get(request.policy_bundle)
        if not policy_bundle:
            return False

        evaluation_context = {
            "request": request.model_dump(),
            "context": context
        }

        try:
            return bool(jsonLogic(policy_bundle, evaluation_context))
        except Exception:
            return False
