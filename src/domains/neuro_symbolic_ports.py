"""Unified Minimal-Additive Ports for adamvangrover/adam (System 1 -> PDIL -> System 2 -> System 3)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

EpistemicTier = Literal["VERIFIED", "CALCULATED", "MODEL_EST", "HYPOTHETICAL", "THIN", "REJECTED"]


# --- 1. GOOGLE LANGEXTRACT + W3C PROV-O OFFSET GROUNDING ---
class GroundedSpan(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    source_doc_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    verbatim_text: str = Field(min_length=1)
    tier: EpistemicTier


# --- 2. TYPESAFE JEV (JEV-RS) SYSTEM 1 JUDGMENT & OPENAI/NEMO TRIPWIRE HANDOFF ---
class JevSystem1Judgment(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    intent_choice: Literal["UNDERWRITE_CREDIT", "SURVEILLANCE_LMT", "MARKET_MAKE", "NULL_INTENT"]
    conviction_score: float = Field(ge=0.0, le=1.0)
    next_step_metadata: dict[str, str]
    grounded_spans: tuple[GroundedSpan, ...]

    def enforce_tripwire(self, min_conviction: float = 0.85) -> JevSystem1Judgment:
        """NeMo / OpenAI-Agents input tripwire: demote to NULL_INTENT if ungrounded or <0.85."""
        if self.conviction_score < min_conviction or not self.grounded_spans:
            return JevSystem1Judgment(
                intent_choice="NULL_INTENT",
                conviction_score=self.conviction_score,
                next_step_metadata={"next_step": "human_escalate", "reason": "low_conviction_or_ungrounded"},
                grounded_spans=self.grounded_spans,
            )
        return self


JEV_ADAPTER = TypeAdapter(JevSystem1Judgment)  # Zero-copy Pydantic v2 Rust validation


# --- 3. OPENFISCA-STYLE VECTORIZED WATERFALL (ASC 842 LEASES + IRC 163(j)) ---
def compute_vectorized_cads_waterfall(
    reported_ebitda: np.ndarray,
    synergy_addbacks: np.ndarray,
    noncancelable_compute_lease_payments: np.ndarray,
    maint_capex: np.ndarray,
    cash_interest: np.ndarray,
    principal_amort: np.ndarray,
    tax_rate: float = 0.21,
    synergy_cap_pct: float = 0.20,
    irc_163j_ebitda_cap: float = 0.30,
) -> dict[str, np.ndarray]:
    """Vectorized OpenFisca-style parameter DAG across N Monte Carlo / QMC paths (zero iterrows)."""
    # Rule #41: Cap borrower synergy add-backs at 20% of base EBITDA
    capped_addbacks = np.minimum(synergy_addbacks, synergy_cap_pct * reported_ebitda)
    ebitdar = reported_ebitda + capped_addbacks + noncancelable_compute_lease_payments

    # IRC Section 163(j): Deductible interest capped at 30% of adjusted taxable base
    deductible_interest = np.minimum(cash_interest, irc_163j_ebitda_cap * np.maximum(0.0, reported_ebitda))
    taxable_income = np.maximum(0.0, reported_ebitda - maint_capex - deductible_interest)
    cash_taxes = taxable_income * tax_rate

    cads = ebitdar - noncancelable_compute_lease_payments - maint_capex - cash_taxes
    total_debt_service = cash_interest + principal_amort
    dscr = np.where(total_debt_service > 0.0, cads / total_debt_service, 999.0)
    return {"ebitdar": ebitdar, "cads": cads, "dscr": dscr}


# --- 4. CATALA DEFAULT-LOGIC + Z3-STYLE COVENANT & LMT BLOCKER VERIFIER ---
@dataclass(frozen=True)
class CatalaCovenantRule:
    """Catala-style default rule with prioritized Section 7.02 exceptions & J.Crew/Serta blockers."""

    base_max_leverage: float = 4.50
    min_dscr_floor: float = 1.05
    has_jcrew_ip_blocker: bool = True
    has_serta_prorata_blocker: bool = True


def verify_covenant_and_lmt_safety(
    rule: CatalaCovenantRule,
    stressed_dscr_p5: float,
    lease_capitalized_leverage: float,
    proposed_unrestricted_sub_ip_transfer: float,
    proposed_non_prorata_uptier: float,
) -> tuple[int, list[str]]:
    """Deterministic symbolic proof of Credit Agreement Section 7.02 & 9.02 invariants (Rule #42)."""
    breaches: list[str] = []
    if stressed_dscr_p5 < rule.min_dscr_floor:
        breaches.append(f"DSCR_BREACH:{stressed_dscr_p5:.2f}x<{rule.min_dscr_floor:.2f}x")
    if lease_capitalized_leverage > rule.base_max_leverage:
        breaches.append(f"LEVERAGE_BREACH:{lease_capitalized_leverage:.2f}x>{rule.base_max_leverage:.2f}x")
    if proposed_unrestricted_sub_ip_transfer > 0.0 and not rule.has_jcrew_ip_blocker:
        breaches.append("LMT_TRAP_UNRESTRICTED_SUB_IP_DROPDOWN")
    if proposed_non_prorata_uptier > 0.0 and not rule.has_serta_prorata_blocker:
        breaches.append("LMT_TRAP_NON_PRORATA_UPTIER")
    return (0 if breaches else 1), breaches


# --- 5. STRIX / CBFKIT CONTROL BARRIER FUNCTION (CBF) SWARM DEFLECTION SHIELD ---
def cbf_safety_filter_order_size(
    nominal_order_qty: float,
    vpin_toxicity: float,
    vpin_limit: float = 0.82,
    alpha_gain: float = 2.5,
) -> tuple[int, float]:
    """
    Control Barrier Function h(x) = vpin_limit - vpin_toxicity >= 0 (Rule #43).
    When h(x) <= 0, Layer 0 locks to 0 and clamps linear inventory absorption to 0.0.
    """
    h_x = vpin_limit - vpin_toxicity
    if h_x <= 0.0:
        return 0, 0.0
    safe_scale = min(1.0, alpha_gain * h_x)
    return 1, nominal_order_qty * safe_scale


# --- 6. MULTIPLICATIVE ASSURANCE ENVELOPE (W3C PROV-O CANONICAL RECORD) ---
def evaluate_and_seal_prov_o(
    payload: dict,
    layer_0_checks: list[int],
    grounding_score: float,
    reasoning_score: float,
    learning_score: float,
) -> dict:
    """Enforces Final Score = Layer 0 (0 or 1) * (0.40*G + 0.35*R + 0.25*L) + SHA-256."""
    layer_0 = 1 if all(c == 1 for c in layer_0_checks) else 0
    raw_l1 = (0.40 * grounding_score) + (0.35 * reasoning_score) + (0.25 * learning_score)
    final_score = round(layer_0 * raw_l1, 4)
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sha256_hash = hashlib.sha256(canonical_bytes).hexdigest()
    route = (
        "BLOCK_AND_LOG"
        if (layer_0 == 0 or final_score < 3.0)
        else ("COMMIT_CANDIDATE" if final_score >= 4.0 else "HUMAN_ESCALATE")
    )
    return {
        "layer_0": layer_0,
        "raw_layer_1": round(raw_l1, 4),
        "final_assured_score": final_score,
        "route": route,
        "w3c_prov_o_sha256": sha256_hash,
    }
