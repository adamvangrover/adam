from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class IntegrationDecision:
    allowed: bool
    reason: str

def decide_integration(
    *,
    hard_gate_passed: bool,
    scope_passed: bool,
    policy_passed: bool,
    provenance_passed: bool,
    calibration_healthy: bool,
    confidence: float,
    threshold: float,
    human_approval: bool = False,
    human_required: bool = False,
) -> IntegrationDecision:
    # Hard failures always dominate statistical confidence.
    if not hard_gate_passed:
        return IntegrationDecision(False, "mandatory invariant failure")

    if not scope_passed:
        return IntegrationDecision(False, "scope violation")

    if not policy_passed:
        return IntegrationDecision(False, "policy violation")

    if not provenance_passed:
        return IntegrationDecision(False, "provenance validation failure")

    if not calibration_healthy:
        return IntegrationDecision(False, "confidence calibration unhealthy")

    if human_required and not human_approval:
        return IntegrationDecision(False, "human approval required")

    if confidence < threshold:
        return IntegrationDecision(
            False,
            f"confidence {confidence:.4f} below threshold {threshold:.4f}",
        )

    return IntegrationDecision(True, "all integration gates passed")
