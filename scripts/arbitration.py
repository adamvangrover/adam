"""Quantitative Credit Risk Neutrality & Arbitration Module."""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class RiskMetrics:
    obligor_id: str
    pd_alpha: float
    pd_beta: float
    bidirectional_divergence: float
    downside_spread: float
    requires_arbitration: bool
    capital_buffer_penalty: float


class ArbitrationEngine:
    """Evaluates dual obligor-level PD forecasts and applies arbitration rules."""

    def __init__(self, divergence_threshold: float = 0.05, penalty_multiplier: float = 1.25):
        if not (0.0 < divergence_threshold < 1.0):
            raise ValueError("Divergence threshold must reside in (0.0, 1.0)")
        self.theta = divergence_threshold
        self.penalty_multiplier = penalty_multiplier

    def compute_metrics(self, obligor_id: str, pd_alpha: float, pd_beta: float) -> RiskMetrics:
        for name, pd in [("Model Alpha", pd_alpha), ("Model Beta", pd_beta)]:
            if not (0.0 <= pd <= 1.0):
                raise ValueError(f"{name} PD must be bounded in [0.0, 1.0], received: {pd}")

        delta_bi = abs(pd_beta - pd_alpha)
        delta_downside = max(0.0, pd_beta - pd_alpha)
        requires_arbitration = delta_bi > self.theta
        capital_buffer_penalty = delta_downside * self.penalty_multiplier

        return RiskMetrics(
            obligor_id=obligor_id,
            pd_alpha=pd_alpha,
            pd_beta=pd_beta,
            bidirectional_divergence=round(delta_bi, 6),
            downside_spread=round(delta_downside, 6),
            requires_arbitration=requires_arbitration,
            capital_buffer_penalty=round(capital_buffer_penalty, 6),
        )

    def evaluate_facility_rating(self, obligor_metrics: RiskMetrics, lgd: float) -> Dict[str, Any]:
        """Facility rating contract: Derives facility metrics without overriding obligor PD."""
        if not (0.0 <= lgd <= 1.0):
            raise ValueError("LGD must reside within [0.0, 1.0]")

        base_el = obligor_metrics.pd_alpha * lgd
        adjusted_el = (obligor_metrics.pd_alpha + obligor_metrics.downside_spread) * lgd

        return {
            "obligor_id": obligor_metrics.obligor_id,
            "expected_loss_base": round(base_el, 6),
            "expected_loss_adjusted": round(adjusted_el, 6),
            "arbitration_status": "ESCALATED" if obligor_metrics.requires_arbitration else "RESOLVED",
        }
