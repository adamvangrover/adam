import json
from datetime import datetime
from typing import Any, Dict

class EstimatorComplianceEvaluator:
    def __init__(self, tau_w: float = 0.3):
        self.tau_w = tau_w

    def evaluate_proposal(
        self,
        authority: str,
        t_e: datetime,
        t_k: datetime,
        t_d: datetime,
        t_x: datetime,
        wasserstein_div: float
    ) -> Dict[str, Any]:
        rejection_reasons = []

        # 1. Authority Check
        if authority != "NONE":
            rejection_reasons.append(f"Authority Violation: authority must be 'NONE', got '{authority}'")

        # 2. Temporal Checks
        if not (t_e <= t_k <= t_d <= t_x):
            rejection_reasons.append("Temporal Geometry Violation: must satisfy t_e <= t_k <= t_d <= t_x")

        # 3. Divergence Check
        epistemic_state = "SUPPORTED"
        if wasserstein_div > self.tau_w:
            epistemic_state = "CONFLICTED"
            rejection_reasons.append(f"Divergence Violation: Wasserstein divergence {wasserstein_div} exceeds threshold {self.tau_w}")

        return {
            "compliant": len(rejection_reasons) == 0,
            "rejection_reasons": rejection_reasons,
            "epistemic_state": epistemic_state
        }

if __name__ == "__main__":
    evaluator = EstimatorComplianceEvaluator()

    # Target Prompt: Predicts future spreads (t_e > t_k) & recommends capital allocation (authority)
    t_k = datetime(2026, 1, 1, 10, 0)
    t_e = datetime(2026, 1, 1, 12, 0) # Violation: t_e > t_k
    t_d = datetime(2026, 1, 1, 14, 0)
    t_x = datetime(2026, 1, 1, 16, 0)

    result = evaluator.evaluate_proposal(
        authority="RECOMMEND_CAPITAL_ALLOCATION", # Violation
        t_e=t_e,
        t_k=t_k,
        t_d=t_d,
        t_x=t_x,
        wasserstein_div=0.1
    )

    print(json.dumps(result, indent=2))
