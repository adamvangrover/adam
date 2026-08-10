from typing import Dict, Any, Optional
from mcp.registry import ToolRegistry
from mcp.capabilities import MCPCapability
from core.governance.policy import PolicyEvaluator

class MCPServer:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def handle_request(self, capability_id: str, payload: Dict[str, Any], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Acts as the Intent Gate.
        Translates the MCP capability request into an authorization query before execution.
        """
        capability = self.registry.get_capability(capability_id)
        if not capability:
            raise ValueError(f"Capability {capability_id} not found in registry.")

        # 1. Structural Intent Gate
        if not capability.validate_intent(payload):
            raise ValueError("Intent validation failed.")

        # 2. Governance Gate
        # In a real app, policy_bundle would load the JsonLogic rules for this capability
        policy_bundle = {"rules": [{"action": "allow"}]}
        evaluator = PolicyEvaluator(policy_bundle)

        proposed_action = {
            "capability_id": capability_id,
            "risk_class": capability.risk_class.value,
            "payload": payload
        }

        if not evaluator.evaluate(proposed_action, user_context):
            return {"status": "denied", "reason": "Governance policy denied execution."}

        # 3. Execution
        implementation = self.registry.get_implementation(capability_id)
        if not implementation:
            raise NotImplementedError(f"Implementation for {capability_id} not found.")

        result = implementation(**payload)

        # 4. (Provenance generation would occur here)

        return {"status": "success", "result": result}
