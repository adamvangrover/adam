import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DeterministicRiskJudge:
    """
    Evaluates dual-model divergence using a deterministic logic engine.
    This replaces probabilistic estimates with authoritative 2LOD assessments.
    """
    def __init__(self, divergence_threshold_bps: float = 35.0, penalty_lambda: float = 1.25):
        self.divergence_threshold_bps = divergence_threshold_bps
        self.penalty_lambda = penalty_lambda

    def evaluate_divergence(
        self,
        baseline_yield_t2: float,
        challenger_yield_t2: float,
        sim_hy_drawdown: float,
        bsl_spread_widening_bps: float,
        eval_id: str = "JUDGE-ADJUDICATION-20260923-0891"
    ) -> Dict[str, Any]:
        """
        Adjudicates the Baseline vs Challenger regimes to apply
        capital penalties if discrepancy exceeds bounds.
        """
        # Bidirectional disparity for circuit breaker
        delta_bps = abs(challenger_yield_t2 - baseline_yield_t2) * 100

        # One-sided downside spread for capital penalties
        downside_delta_bps = max(0.0, (challenger_yield_t2 - baseline_yield_t2) * 100)
        capital_buffer_penalty = downside_delta_bps * self.penalty_lambda

        severity = "NORMAL"
        circuit_breaker = "OPERATIONAL"
        resolution = "BASELINE_VALID"
        rationale = "Divergence within normal operating bounds."

        if delta_bps >= self.divergence_threshold_bps:
            severity = "CRITICAL"
            circuit_breaker = "TRIPPED_TIER_2_SOFT_STOP"
            resolution = "CHALLENGER_FLIGHT_TO_QUALITY_OVERRIDES_BASELINE"
            rationale = (f"The {delta_bps:.1f} bps divergence on Day 2 exceeds the "
                         f"{self.divergence_threshold_bps:.1f} bps threshold. The Challenger's "
                         "flight-to-quality dynamics correctly supersede the Baseline hawkish trajectory "
                         "during the initial shock phase. Quoting engines must widen bid-ask spreads "
                         "and enforce a 12.5% collateral haircut on unrated middle-market paper "
                         "while preserving core underwriting limits.")

        return {
            "evaluation_id": eval_id,
            "runtime_telemetry": {
                "baseline_10y_yield_t2": baseline_yield_t2,
                "challenger_10y_yield_t2": challenger_yield_t2,
                "absolute_yield_delta_bps": delta_bps,
                "divergence_threshold_bps": self.divergence_threshold_bps,
                "bsl_credit_spread_widening_bps": bsl_spread_widening_bps,
                "sim_hy_credit_drawdown_pct": sim_hy_drawdown
            },
            "adjudication_verdict": {
                "divergence_severity": severity,
                "circuit_breaker_status": circuit_breaker,
                "precedence_resolution": resolution,
                "underwriting_thesis_status": "FOUNDATIONAL_THESIS_INTACT",
                "capital_buffer_action": f"APPLY_DOWNSIDE_SPREAD_SURCHARGE (lambda = {self.penalty_lambda:.2f})",
                "adjudication_rationale": rationale,
                "calculated_penalty_bps": capital_buffer_penalty
            }
        }

if __name__ == "__main__":
    judge = DeterministicRiskJudge()
    result = judge.evaluate_divergence(
        baseline_yield_t2=5.18,
        challenger_yield_t2=4.72,
        sim_hy_drawdown=-7.50,
        bsl_spread_widening_bps=135.0
    )
    print(json.dumps(result, indent=2))
