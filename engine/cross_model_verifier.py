import hashlib
from enum import Enum
from typing import Dict, Any, List, Optional
from schemas.credit_contracts import FinancialAssertion


class DisagreementClassification(str, Enum):
    NONE = "NONE"
    MINOR_TOLERANCE_VARIANCE = "MINOR_TOLERANCE_VARIANCE"
    SUBSTANTIAL_DISAGREEMENT = "SUBSTANTIAL_DISAGREEMENT"


class CrossModelVerifier:
    """
    Independent cross-model verifier comparing Primary and Challenger model extractions,
    detecting >5% numerical discrepancies or unit/period mismatches, enforcing idempotency
    keys to prevent duplicate execution cascades, and routing disagreements to human 2nd Line review.
    """

    def __init__(self, tolerance_pct: float = 0.05):
        self.tolerance_pct = tolerance_pct
        self.processed_idempotency_keys: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def generate_idempotency_key(primary_assertion: FinancialAssertion, challenger_assertion: FinancialAssertion) -> str:
        payload = f"{primary_assertion.metric_name}:{primary_assertion.reporting_period}:{primary_assertion.value}:{challenger_assertion.value}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compare_assertions(
        self,
        primary: FinancialAssertion,
        challenger: FinancialAssertion,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        key = idempotency_key or self.generate_idempotency_key(primary, challenger)

        # Anti-cascade idempotency check
        if key in self.processed_idempotency_keys:
            cached_res = self.processed_idempotency_keys[key]
            cached_res["is_idempotent_replay"] = True
            return cached_res

        # Unit mismatch check
        if primary.unit.upper() != challenger.unit.upper():
            res = {
                "idempotency_key": key,
                "classification": DisagreementClassification.SUBSTANTIAL_DISAGREEMENT,
                "reason": f"Unit mismatch: Primary unit '{primary.unit}' vs Challenger unit '{challenger.unit}'",
                "variance_pct": None,
                "requires_human_escalation": True,
                "governance_action": "ROUTE_TO_SECOND_LINE_CREDIT_OFFICER",
                "is_idempotent_replay": False
            }
            self.processed_idempotency_keys[key] = res
            return res

        # Reporting period mismatch check
        if primary.reporting_period != challenger.reporting_period:
            res = {
                "idempotency_key": key,
                "classification": DisagreementClassification.SUBSTANTIAL_DISAGREEMENT,
                "reason": f"Reporting period mismatch: Primary period '{primary.reporting_period}' vs Challenger period '{challenger.reporting_period}'",
                "variance_pct": None,
                "requires_human_escalation": True,
                "governance_action": "ROUTE_TO_SECOND_LINE_CREDIT_OFFICER",
                "is_idempotent_replay": False
            }
            self.processed_idempotency_keys[key] = res
            return res

        # Relative percentage variance calculation
        denominator = abs(primary.value) if primary.value != 0 else 1.0
        variance = abs(primary.value - challenger.value) / denominator

        if variance > self.tolerance_pct:
            classification = DisagreementClassification.SUBSTANTIAL_DISAGREEMENT
            requires_escalation = True
            action = "ROUTE_TO_SECOND_LINE_CREDIT_OFFICER"
        elif variance > 0:
            classification = DisagreementClassification.MINOR_TOLERANCE_VARIANCE
            requires_escalation = False
            action = "PROCEED_WITH_PRIMARY_VALUE"
        else:
            classification = DisagreementClassification.NONE
            requires_escalation = False
            action = "PROCEED_AUTONOMOUSLY"

        res = {
            "idempotency_key": key,
            "metric_name": primary.metric_name,
            "primary_value": primary.value,
            "challenger_value": challenger.value,
            "variance_pct": round(variance * 100.0, 4),
            "classification": classification,
            "requires_human_escalation": requires_escalation,
            "governance_action": action,
            "is_idempotent_replay": False
        }
        self.processed_idempotency_keys[key] = res
        return res
