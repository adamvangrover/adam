import json
import logging
import sys
import os

# Ensure the root directory is in the path to allow absolute imports from core_kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_kernel.agnostic_math import AgnosticMathEngine
from core_kernel.orchestration_framework import OrchestratorEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GovernanceValidator")

def validate_jsonlogic_policy(payload: dict) -> dict:
    """
    Skill to evaluate proposed actions against deterministic jsonLogic rules.
    Links the governance domain's security needs to the execution engine.
    """
    logger.info("Initializing jsonLogic governance gatekeeper...")

    proposed_action = payload.get("action", {})
    action_id = proposed_action.get("id", "UNKNOWN")
    target = proposed_action.get("target", "")

    # Simple deterministic rule check: Deny if target is a private IP (e.g., 10.x, 192.168.x)
    # In a real system, this would evaluate a complex jsonLogic ruleset
    if target.startswith("10.") or target.startswith("192.168."):
        decision = "DENY"
        reason = "Private IP target blocked by Network Security Policy."
    else:
        decision = "ALLOW"
        reason = "Action compliant with deterministic logic."

    result_payload = {
        "action_id": action_id,
        "policy_decision": decision,
        "reasoning": reason
    }

    # Freeze the evaluation state with core kernel hash for W3C PROV-O telemetry
    state_hash = AgnosticMathEngine.deterministic_hash(result_payload)
    result_payload["provenance_hash"] = state_hash

    logger.info(f"Policy evaluation complete. Decision: {decision}")
    return result_payload

if __name__ == "__main__":
    orchestrator = OrchestratorEngine()
    orchestrator.register_skill("validate_policy", validate_jsonlogic_policy)

    test_payload_1 = {"action": {"id": "ACT-001", "target": "https://api.public.com/data"}}
    test_payload_2 = {"action": {"id": "ACT-002", "target": "10.0.0.5"}}

    print("Executing valid skill...")
    result_1 = orchestrator.execute_skill("validate_policy", test_payload_1)
    print(json.dumps(result_1, indent=2))

    print("\nExecuting invalid skill (should trigger DENY)...")
    result_2 = orchestrator.execute_skill("validate_policy", test_payload_2)
    print(json.dumps(result_2, indent=2))
