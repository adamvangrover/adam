import json
import os
import logging
from typing import Dict, Any, Tuple, Optional
from core.governance.sentinel_harness import CreditMetrics, DecisionState, GuardrailWrapper

logger = logging.getLogger(__name__)

class RiskSynthesisEngine:
    def __init__(self, wrapper: GuardrailWrapper = None):
        self.wrapper = wrapper or GuardrailWrapper()
        self.sigma = 0.15
        self.conviction_threshold = 0.90
        self._load_governance()

    def _load_governance(self):
        gov_path = os.path.join(os.path.dirname(__file__), '..', 'governance', 'governance.json')
        if os.path.exists(gov_path):
            with open(gov_path, 'r') as f:
                gov_data = json.load(f)
                thresholds = gov_data.get('risk_thresholds', {})
                self.sigma = thresholds.get('sigma', 0.15)
                self.conviction_threshold = thresholds.get('conviction_threshold', 0.90)

    def compute_gate_threshold(self, npv_fees: float, sigma: Optional[float] = None) -> float:
        """
        Input: Client AUM, Projected Fees, Risk Appetite Scalar (sigma).
        Equation: Gate = NPV_Fees * sigma.
        """
        return npv_fees * (sigma if sigma is not None else self.sigma)

    def evaluate_decision_state(self, metrics: CreditMetrics, conviction: float, npv_fees: float, sigma: Optional[float] = None, policy_breach: bool = False, prompt: str = "", context: Dict[str, Any] = None) -> DecisionState:
        """
        Evaluate logic transition based on gate and execution modalities:
        - Pass-Through: EL < Gate AND Conviction > 0.9 (Auto-Approved)
        - HOTL Review: EL < Gate AND Conviction < 0.9 (Retrospective Validation)
        - HITL Sync: EL >= Gate OR Policy Breach (MFA Step-Up Required)
        """
        if context is None:
            context = {}

        gate = self.compute_gate_threshold(npv_fees, sigma)
        el = metrics.expected_loss

        requires_step_up = False
        routing_path = "UNDEFINED"

        if el >= gate or policy_breach:
            routing_path = "HITL_TIER_3"
            requires_step_up = True
            logger.info("Risk threshold exceeded or policy breach. Triggering Tier 3 Escalation.")
        elif el < gate and conviction > self.conviction_threshold:
            routing_path = "AUTOMATED"
            logger.info("Pass-Through: Auto-Approved.")
        else:
            routing_path = "HOTL"
            logger.info("HOTL Review: Retrospective Validation.")

        audit_hash = self.wrapper.generate_audit_hash(prompt, context)

        return DecisionState(
            conviction_score=conviction,
            routing_path=routing_path,
            requires_step_up=requires_step_up,
            audit_hash=audit_hash
        )

    def process_orchestration(self, metrics_data: Dict[str, float], conviction: float, npv_fees: float, sigma: Optional[float] = None, jurisdiction: str = "USA", prompt: str = "", context: Dict[str, Any] = None) -> Tuple[DecisionState, str]:
        if context is None:
            context = {}

        metrics = CreditMetrics(**metrics_data)
        safe_prompt = self.wrapper.wrap_prompt(prompt, jurisdiction)

        # Policy breach if redlines are violated (e.g., Cayman Islands)
        policy_breach = self.wrapper.enforce_redlines(jurisdiction)

        decision_state = self.evaluate_decision_state(
            metrics=metrics,
            conviction=conviction,
            npv_fees=npv_fees,
            sigma=sigma,
            policy_breach=policy_breach,
            prompt=safe_prompt,
            context=context
        )

        return decision_state, safe_prompt
