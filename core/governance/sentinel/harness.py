import json
import hashlib
from typing import Dict, Any, Tuple
from core.schemas.sentinel import CreditMetrics, DecisionState
import os

class RiskSynthesisEngine:
    def __init__(self, governance_path: str = "core/governance/sentinel/governance.json"):
        self.governance_path = governance_path
        self._scalar = self._load_scalar()

    def _load_scalar(self) -> float:
        try:
            with open(self.governance_path, 'r') as f:
                config = json.load(f)
                return config.get("npv_fees_scalar", 0.15)
        except Exception as e:
            # Fallback
            return 0.15

    def evaluate(self, metrics: CreditMetrics, npv_fees: float, conviction_score: float, policy_breach: bool = False, prompt_context: str = "") -> DecisionState:
        gate = npv_fees * self._scalar
        el = metrics.expected_loss

        requires_step_up = False

        # C. Execution Modalities
        if el >= gate or policy_breach:
            routing_path = "HITL_TIER_3"
            requires_step_up = True
        elif el < gate and conviction_score > 0.9:
            routing_path = "AUTOMATED"
        else: # el < gate AND conviction_score <= 0.9
            routing_path = "HOTL"

        audit_hash = hashlib.sha256(f"{prompt_context}:{el}:{gate}:{conviction_score}".encode()).hexdigest()

        return DecisionState(
            conviction_score=conviction_score,
            routing_path=routing_path,
            requires_step_up=requires_step_up,
            audit_hash=audit_hash
        )


class GuardrailWrapper:
    def __init__(self):
        pass

    def sanitize_pii(self, prompt: str) -> str:
        """
        Intercepts prompt strings; replaces names/SSNs with tokens (e.g., `[ENTITY_A]`).
        Simple mock implementation for demo.
        """
        # In a real scenario, use NER or regex
        sanitized = prompt.replace("John Doe", "[ENTITY_A]")
        sanitized = sanitized.replace("123-45-6789", "[SSN_REDACTED]")
        return sanitized

    def inject_context(self, prompt: str, market_state: str = "Current VIX: 22.5 - Caution Mode Active") -> str:
        """
        Appends current market state to the system prompt.
        """
        return f"{prompt}\n\n[SYSTEM CONTEXT]: {market_state}"

    def enforce_redlines(self, context: Dict[str, Any]) -> bool:
        """
        A set of hardcoded "No-Go" rules. Returns True if there's a policy breach.
        """
        jurisdiction = context.get("jurisdiction", "").lower()
        if jurisdiction == "cayman islands":
            return True
        return False
