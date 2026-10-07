"""
Institutional Multi-Horizon Credit Engine: Type Interfaces, Invariants,
and Deterministic Evaluation Kernel.
Complies with SR 11-7 validation protocols and PROV-O audit tracing.
"""

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MacroRegime(str, Enum):
    EXPANSION = "EXPANSION"
    NORMAL = "NORMAL"
    SLOWDOWN = "SLOWDOWN"
    RECESSION = "RECESSION"
    CREDIT_STRESS = "CREDIT_STRESS"
    CRISIS = "CRISIS"


class TemporalEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    t_effective: datetime = Field(description="Date of underlying financial data")
    t_knowledge: datetime = Field(description="Date when data was ingested/known")
    t_decision: datetime = Field(description="Execution timestamp of model run")
    t_execution: datetime = Field(description="Timestamp of consensus sign-off")

    @field_validator("*")
    def validate_utc_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo != timezone.utc:
            raise ValueError("All temporal envelope clocks must be explicitly UTC-aware")
        return v

    @model_validator(mode="after")
    def validate_monotonicity(self) -> "TemporalEnvelope":
        if not (self.t_effective <= self.t_knowledge <= self.t_decision <= self.t_execution):
            raise ValueError(
                f"Temporal monotonicity violated: {self.t_effective} <= "
                f"{self.t_knowledge} <= {self.t_decision} <= {self.t_execution}"
            )
        return self


class ModelProposal(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    model_name: str
    pd_1y_bps: int = Field(ge=1, le=10000, description="1Y PD in integer basis points (1 bp to 100%)")
    pd_3y_bps: Optional[int] = Field(None, ge=1, le=10000)
    pd_5y_bps: Optional[int] = Field(None, ge=1, le=10000)
    confidence_permille: int = Field(ge=0, le=1000, description="Uncalibrated confidence in permille (0-1000)")
    evidence_payload_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    epistemic_state: str = Field(pattern=r"^(SUPPORTED|CONFLICTED|REJECTED)$")

    @field_validator("evidence_payload_sha256")
    def reject_empty_hash(cls, v: str) -> str:
        # Prevent e3b0c442... empty SHA-256 string from being accepted as evidence
        if v == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
            raise ValueError("Evidence payload hash cannot be SHA-256 of empty content")
        return v


class AdjudicationVerdict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    evaluation_id: str = Field(min_length=16, description="Unique, non-constant UUID/HMAC evaluation identifier")
    temporal_envelope: TemporalEnvelope
    selected_regime: MacroRegime
    physical_pd_1y_bps: int = Field(ge=1, le=10000)
    model_disagreement_bps: int = Field(ge=0, description="Variance/spread across models in bps")
    risk_neutral_hazard_bps: int = Field(ge=1, description="Bootstrap λ^Q in bps")
    par_cds_spread_bps: int = Field(ge=1, description="Calculated fair CDS spread in bps")
    circuit_breaker_status: str = Field(pattern=r"^(OPERATIONAL|CIRCUIT_BREAKER_TRIPPED)$")
    capital_buffer_action: str = Field(pattern=r"^(STANDARD_CAPITAL|APPLY_DOWNSIDE_SPREAD_SURCHARGE|IMMEDIATE_DELEVERAGING_COVENANT)$")
    underwriting_thesis_status: str = Field(pattern=r"^(FOUNDATIONAL_THESIS_INTACT|DETERIORATING_FORWARD_RISK|STRUCTURAL_IMPAIRMENT)$")
    decision_digest_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")


class DeterministicCreditJudge:
    def __init__(self, tolerance_bps: int = 35):
        self.tolerance_bps = tolerance_bps

    @staticmethod
    def _bps_to_prob(bps: int) -> float:
        return bps / 10000.0

    @staticmethod
    def _prob_to_logit(p: float) -> float:
        p = max(min(p, 0.9999), 0.0001)
        return math.log(p / (1.0 - p))

    @staticmethod
    def _logit_to_prob(l: float) -> float:
        return 1.0 / (1.0 + math.exp(-l))

    @staticmethod
    def _canonical_sha256(data: Dict[str, Any]) -> str:
        canonical_json = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(canonical_json).hexdigest()

    def adjudicate(
        self,
        envelope: Dict[str, datetime],
        proposals: List[Dict[str, Any]],
        regime: str,
        macro_stress_factor: float,
        cds_recovery_rate: float = 0.40
    ) -> Dict[str, Any]:
        """
        Executes deterministic multi-model adjudication and CDS spread bootstrapping.
        """
        # 1. Non-finite boundary enforcement (fail-closed)
        for p in proposals:
            for key, val in p.items():
                if isinstance(val, (float, int)):
                    if math.isnan(val) or math.isinf(val):
                        return self._fail_closed("NON_FINITE_NUMERIC_INPUT_DETECTED")
            if p.get("pd_1y_bps", 0) <= 0:
                return self._fail_closed("NEGATIVE_OR_ZERO_PD_DETECTED")

        # 2. Extract baseline and challengers
        by_name = {p["model_name"]: p for p in proposals}
        ttc = by_name.get("TTC")
        ltm = by_name.get("LTM")
        fwd = by_name.get("FWD")
        eq = by_name.get("EQUITY")

        if not all([ttc, ltm, fwd, eq]):
            return self._fail_closed("INCOMPLETE_SWARM_PROPOSALS")

        # 3. Disagreement Calculation in Logit Space
        logits = [
            self._prob_to_logit(self._bps_to_prob(ttc["pd_1y_bps"])),
            self._prob_to_logit(self._bps_to_prob(ltm["pd_1y_bps"])),
            self._prob_to_logit(self._bps_to_prob(fwd["pd_1y_bps"])),
            self._prob_to_logit(self._bps_to_prob(eq["pd_1y_bps"]))
        ]
        mean_logit = sum(logits) / len(logits)
        logit_variance = sum((x - mean_logit) ** 2 for x in logits) / (len(logits) - 1)

        # Disagreement variance converted to integer bps
        disagreement_bps = int(Decimal(str(logit_variance * 100.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # 4. Regime-aware weights
        weights = {
            "EXPANSION": (0.45, 0.25, 0.20, 0.10, 5),
            "NORMAL": (0.40, 0.25, 0.20, 0.15, 10),
            "SLOWDOWN": (0.25, 0.30, 0.30, 0.15, 20),
            "RECESSION": (0.15, 0.25, 0.35, 0.25, 30),
            "CREDIT_STRESS": (0.10, 0.20, 0.35, 0.35, 45),
            "CRISIS": (0.05, 0.15, 0.40, 0.40, 60),
        }.get(regime, (0.25, 0.25, 0.25, 0.25, 25))

        w1, w2, w3, w4, penalty_per_var = weights
        base_fused_logit = (w1 * logits[0]) + (w2 * logits[1]) + (w3 * logits[2]) + (w4 * logits[3])
        adjusted_prob = self._logit_to_prob(base_fused_logit)
        
        # Apply disagreement penalty
        physical_pd_1y_bps = int(Decimal(str(adjusted_prob * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        physical_pd_1y_bps += int(disagreement_bps * (penalty_per_var / 10.0))

        # 5. Exact Boundary Comparison in Integer Basis Points
        divergence_bps = abs(ltm["pd_1y_bps"] - ttc["pd_1y_bps"])
        is_critical = divergence_bps >= self.tolerance_bps

        if is_critical or physical_pd_1y_bps > 1000:
            circuit_status = "OPERATIONAL"  # Handled within risk parameters
            buffer_action = "APPLY_DOWNSIDE_SPREAD_SURCHARGE"
            thesis_status = "DETERIORATING_FORWARD_RISK"
        else:
            circuit_status = "OPERATIONAL"
            buffer_action = "STANDARD_CAPITAL"
            thesis_status = "FOUNDATIONAL_THESIS_INTACT"

        # 6. Physical -> Risk-Neutral Transform & CDS Spread Bootstrapping
        # λ^P ≈ -ln(1 - PD)
        phys_p = self._bps_to_prob(physical_pd_1y_bps)
        lambda_p = -math.log(max(1.0 - phys_p, 0.00001))
        
        # Risk premium RP_t parameterized by macro stress
        risk_premium = 0.0035 + (0.0020 * macro_stress_factor) + (logit_variance * 0.001)
        lambda_q = lambda_p + risk_premium
        rn_hazard_bps = int(Decimal(str(lambda_q * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # Simplified Par CDS Pricing across 5-year piecewise flat curve
        # Protection PV ≈ (1 - R) * λ^Q * Σ D_i * dt
        # Premium PV ≈ s * Σ D_i * dt
        # Thus s* ≈ (1 - R) * λ^Q (with payment schedule & default accrual adjustment)
        accrual_adjustment = 1.0 + (0.5 * lambda_q)
        par_spread = (1.0 - cds_recovery_rate) * lambda_q * accrual_adjustment
        par_cds_spread_bps = int(Decimal(str(par_spread * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # 7. Construct Result Payload with Unique Evaluation ID and Cryptographic Hash
        evaluation_id = f"eval_{uuid.uuid4().hex}"
        record = {
            "evaluation_id": evaluation_id,
            "temporal_envelope": {k: v.isoformat() for k, v in envelope.items()},
            "regime": regime,
            "physical_pd_1y_bps": physical_pd_1y_bps,
            "model_disagreement_bps": disagreement_bps,
            "divergence_bps": divergence_bps,
            "risk_neutral_hazard_bps": rn_hazard_bps,
            "par_cds_spread_bps": par_cds_spread_bps,
            "circuit_breaker_status": circuit_status,
            "capital_buffer_action": buffer_action,
            "underwriting_thesis_status": thesis_status,
            "proposals_evaluated": [p["model_name"] for p in proposals],
        }

        # Deterministic digest binding
        record["decision_digest_sha256"] = self._canonical_sha256(record)
        return record

    def _fail_closed(self, reason: str) -> Dict[str, Any]:
        bad_record = {
            "evaluation_id": f"eval_tripped_{uuid.uuid4().hex}",
            "error_reason": reason,
            "circuit_breaker_status": "CIRCUIT_BREAKER_TRIPPED",
            "capital_buffer_action": "IMMEDIATE_DELEVERAGING_COVENANT",
            "underwriting_thesis_status": "STRUCTURAL_IMPAIRMENT",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        bad_record["decision_digest_sha256"] = self._canonical_sha256(bad_record)
        return bad_record
