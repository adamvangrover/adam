"""h_ver — pre-stop hook: block ungrounded exits and force HITL on high-impact actions."""
import logging
from typing import Any, Dict, List, Optional

from core.engine.harness.policy import HarnessPolicy, VerificationPolicy
from core.schemas.agent_schema import AgentOutput

logger = logging.getLogger(__name__)

CONVICTION_BREACH_EVENT = "CRITICAL_CONVICTION_BREACH"


def confidence_tier(confidence: float, policy: HarnessPolicy) -> str:
    if confidence >= policy.autonomous_min:
        return "autonomous"
    if confidence >= policy.hitl_min:
        return "hitl_required"
    return "hard_abort"


def provenance_violations(output: AgentOutput) -> List[str]:
    if not output.provenance_trace:
        return ["missing_provenance"]
    if any(not (getattr(h, "jsonLogic_version", None) or "").strip() for h in output.provenance_trace):
        return ["missing_jsonLogic_version"]
    return []


def is_high_impact(metadata: Dict[str, Any], policy: VerificationPolicy) -> bool:
    actions = set(policy.high_impact_actions)
    return (str(metadata.get("impact", "")).lower() in policy.hitl_impact_levels
            or metadata.get("action") in actions or metadata.get("next_step") in actions)


def hard_abort_output(reasons: List[str], policy: HarnessPolicy, details: Optional[Dict[str, Any]] = None) -> AgentOutput:
    """Quarantined exit: confidence below the HITL floor, no answer content, no handoff."""
    logger.warning("%s: %s %s", CONVICTION_BREACH_EVENT, reasons, details or {})
    return AgentOutput(
        answer=f"HARD_ABORT: {', '.join(reasons)}",
        confidence=policy.verification.abort_confidence,
        metadata={"confidence_tier": "hard_abort", "hitl_required": False, "event": CONVICTION_BREACH_EVENT,
                  "harness_violations": reasons, **({"harness_details": details} if details else {})},
    )


def pre_stop(output: AgentOutput, policy: HarnessPolicy) -> AgentOutput:
    """Return the output with ``confidence_tier``/``hitl_required`` set, or a hard abort if ungrounded."""
    violations = provenance_violations(output)
    if violations:
        return hard_abort_output(violations, policy, {"pre_abort_confidence": output.confidence})
    tier = confidence_tier(output.confidence, policy)
    high_impact = is_high_impact(output.metadata, policy.verification)
    if high_impact and tier == "autonomous":
        tier = "hitl_required"
    metadata = {**output.metadata, "confidence_tier": tier, "hitl_required": tier == "hitl_required",
                "high_impact": high_impact}
    return output.model_copy(update={"metadata": metadata})
