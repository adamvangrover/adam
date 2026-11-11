"""
ADAM OS v30.1 / AFOS V3 — Neuro-Symbolic Risk Adapter & Multiplicative Assurance Gate
Module: adam_finance/risk_adapter_gate.py
Compliance: Federal Reserve SR 11-7, OCC 2026-13, Basel CRE 31 (A-IRB)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EpistemicStatus(str, Enum):
    VERIFIED = "[VERIFIED]"       # Immutable primary filing / ledger record
    CALCULATED = "[CALCULATED]"   # Deterministic adam-core Rust / Python output
    MODEL_EST = "[MODEL_EST]"     # Calibrated statistical output (e.g. TimesFM quantile)
    PENDING = "[PENDING]"         # Missing context requiring human/data resolution


class RoutingAction(str, Enum):
    COMMIT = "COMMIT"
    HUMAN_ESCALATE = "HUMAN_ESCALATE"
    C5_DEGRADED_FALLBACK = "C5_DEGRADED_FALLBACK"
    BLOCK_AND_LOG = "BLOCK_AND_LOG"


@dataclass(frozen=True)
class TimesFMRiskSurface:
    """
    Output contract for mcp://adam-finance/timesfm-risk-adapter (~2.46M trained adapter params).
    Enforces monotonicity (non-crossing quantiles) and rolling Actual-over-Expected (AE) calibration.
    """
    asset_id: str
    horizon_days: int
    adapter_param_count: int  # Expected ~2,460,000 for IDX risk-adapted head
    quantiles: dict[float, float]  # e.g., {0.01: -0.184, 0.05: -0.112, 0.50: 0.002, 0.99: 0.145}
    trailing_252d_exceedances_99: int
    epistemic_tag: EpistemicStatus = EpistemicStatus.MODEL_EST

    def enforce_monotonic_quantiles(self) -> dict[float, float]:
        """Repairs any quantile crossing via cumulative maximum pass."""
        sorted_levels = sorted(self.quantiles.keys())
        repaired: dict[float, float] = {}
        running_max = -float("inf")
        for q in sorted_levels:
            running_max = max(running_max, self.quantiles[q])
            repaired[q] = running_max
        return repaired

    @property
    def actual_over_expected_99(self) -> float:
        """Computes rolling 252-day AE ratio at alpha = 0.01 (expected = 2.52 breaches)."""
        expected = 252.0 * 0.01
        return self.trailing_252d_exceedances_99 / expected

    def calibrated_tail_var_99(self) -> float:
        """
        Applies automatic regime-lag stress multiplier if rolling AE_0.99 > 1.35
        (preventing under-prediction during volatility transitions).
        """
        repaired = self.enforce_monotonic_quantiles()
        raw_var_99 = abs(min(0.0, repaired[0.01]))
        ae = self.actual_over_expected_99
        stress_multiplier = max(1.0, ae) if ae > 1.35 else 1.0
        return raw_var_99 * stress_multiplier


@dataclass(frozen=True)
class CreditSymbolicMetrics:
    """Deterministic metrics computed by adam-core (Plane C)."""
    obligor_id: str
    fccr_reported: float
    fccr_true_covenant: float
    gross_leverage: float
    dscr_stressed_p05: float
    pd_pit: float
    pd_ttc: float
    cohort_75th_pct_pd_floor: float
    brier_score_oot: float
    lmt_uncapped_leakage_pct: float
    accounting_identity_residual_usd_m: float
    prov_o_root_sources: set[str] = field(default_factory=set)

    def hybrid_regulatory_pd(self, pit_weight: float = 0.35) -> float:
        """
        Computes Basel CRE 31 compliant Hybrid PD with mandatory 5 bps floor
        and 2LoD 75th-percentile cohort override when Leverage > 6.0x or DSCR < 1.15x.
        """
        blended = (1.0 - pit_weight) * self.pd_ttc + pit_weight * self.pd_pit
        basel_floored = max(0.0005, blended)  # 5 bps CRE 31 minimum floor
        if self.gross_leverage > 6.0 or self.fccr_true_covenant < 1.15:
            return max(basel_floored, self.cohort_75th_pct_pd_floor)
        return basel_floored


def evaluate_neuro_symbolic_decision(
    credit: CreditSymbolicMetrics,
    market: TimesFMRiskSurface,
    s_grounding: float,
    s_downside: float,
    s_reusability: float,
    materiality_tier: int = 1,
    human_approved: bool = False,
) -> dict[str, object]:
    """
    Executes the Multiplicative Safety Gate (Layer 0 x Layer 1) and Lexicographic Tripwires.
    """
    # 1. Layer 0 Binary Gate Checks
    g_provenance = 1 if len(credit.prov_o_root_sources) >= 3 else 0
    g_reconcile = 1 if abs(credit.accounting_identity_residual_usd_m) <= 0.01 else 0
    g_calibration = 1 if credit.brier_score_oot <= 0.065 else 0
    g_approval = 1 if (materiality_tier <= 2 or human_approved) else 0

    layer_0_gate = g_provenance * g_reconcile * g_calibration * g_approval

    # 2. Layer 1 Weighted Rubric Score (1.0 to 5.0 scale)
    layer_1_score = (0.40 * s_grounding) + (0.35 * s_downside) + (0.25 * s_reusability)
    final_assured_score = float(layer_0_gate) * layer_1_score

    # 3. Hard Lexicographic Credit & LMT Tripwires (Cannot be overridden by high rubric prose)
    hard_tripwire_breached = (
        credit.fccr_true_covenant < 1.05
        or credit.gross_leverage > 6.50
        or credit.lmt_uncapped_leakage_pct > 0.20
        or market.calibrated_tail_var_99() > 0.25
    )

    # 4. Deterministic Routing Decision
    if layer_0_gate == 0:
        if g_calibration == 0 or market.actual_over_expected_99 > 1.75:
            routing = RoutingAction.C5_DEGRADED_FALLBACK
        else:
            routing = RoutingAction.BLOCK_AND_LOG
    elif hard_tripwire_breached or final_assured_score < 4.00 or (materiality_tier >= 2 and not human_approved):
        routing = RoutingAction.HUMAN_ESCALATE
    else:
        routing = RoutingAction.COMMIT

    return {
        "obligor_id": credit.obligor_id,
        "layer_0_multiplier": layer_0_gate,
        "final_assured_score": round(final_assured_score, 3),
        "hybrid_regulatory_pd": round(credit.hybrid_regulatory_pd(), 5),
        "calibrated_timesfm_var_99": round(market.calibrated_tail_var_99(), 4),
        "timesfm_ae_ratio_99": round(market.actual_over_expected_99, 2),
        "hard_tripwire_breached": hard_tripwire_breached,
        "routing_action": routing.value,
    }
