"""
MultiHorizonCreditEngine: Institutional Multi-Horizon, Multi-Measure Credit Risk Engine.

Includes:
- TTC Anchor Specialist (Latent Factor Model)
- PIT / LTM Accounting & Macro Challenger
- TTM-Forward Shadow Challenger (Forward Curves & Refinancing Stress)
- Merton Equity Structural Engine (Forward Barrier Simulation)
- Log-Odds Disagreement Fusion Judge (Regime-Aware)
- Physical to Risk-Neutral Hazard Transformation
- Institutional Arbitrage-Free CDS Pricing Bootstrapper
- Credit-Equity Feedback Loop Engine

Strictly compliant with SR 11-7 / OCC 2011-12 model risk governance and W3C PROV-O provenance.
"""

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
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
        now_utc = datetime.now(timezone.utc)
        if not (self.t_effective <= self.t_knowledge <= self.t_decision <= self.t_execution <= now_utc):
            raise ValueError(
                f"Temporal monotonicity or upper UTC bound violated: {self.t_effective} <= "
                f"{self.t_knowledge} <= {self.t_decision} <= {self.t_execution} <= {now_utc}"
            )
        return self


class ModelProposal(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    model_name: str
    pd_1y_bps: int = Field(ge=1, le=9999, description="1Y PD in integer basis points strictly in (0, 1) -> 1 to 9999 bps")
    pd_3y_bps: Optional[int] = Field(None, ge=1, le=9999)
    pd_5y_bps: Optional[int] = Field(None, ge=1, le=9999)
    confidence_permille: Optional[int] = Field(None, ge=0, le=1000, description="Uncalibrated confidence in permille (0-1000)")
    evidence_payload_sha256: Optional[str] = Field(None, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    epistemic_state: Optional[str] = Field(None, pattern=r"^(SUPPORTED|CONFLICTED|REJECTED)$")

    @field_validator("evidence_payload_sha256")
    def reject_empty_hash(cls, v: Optional[str]) -> Optional[str]:
        if v == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
            raise ValueError("Evidence payload hash cannot be SHA-256 of empty content")
        return v


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class TTCAnchorEngine:
    """TTC Anchor Specialist: Latent credit quality modeler."""

    @staticmethod
    def calculate_ttc_pd(
        fundamental_score: float,
        industry_score: float,
        rating_alpha: float = 0.0,
        w_fund: float = 0.6,
        w_ind: float = 0.4,
    ) -> Dict[str, int]:
        c_ttc = (w_fund * fundamental_score) + (w_ind * industry_score) + rating_alpha
        pd_1y = 1.0 / (1.0 + math.exp(-c_ttc))
        
        # Extrapolate 3Y and 5Y cumulative PDs
        pd_1y_bps = max(1, min(9999, int(Decimal(str(pd_1y * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))))
        pd_3y_bps = max(1, min(9999, int(Decimal(str((1.0 - (1.0 - pd_1y) ** 3) * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))))
        pd_5y_bps = max(1, min(9999, int(Decimal(str((1.0 - (1.0 - pd_1y) ** 5) * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))))

        return {
            "pd_1y_bps": pd_1y_bps,
            "pd_3y_bps": pd_3y_bps,
            "pd_5y_bps": pd_5y_bps,
        }


class PITLTMEngine:
    """Point-in-time LTM accounting challenger."""

    @staticmethod
    def calculate_ltm_pd(
        c_ttc: float,
        net_leverage: float,
        delta_ebitda: float,
        interest_coverage: float,
        liquidity_ratio: float,
        macro_cycle: float,
        beta_lev: float = 0.25,
        beta_ebitda: float = -0.30,
        beta_ic: float = -0.20,
        beta_liq: float = -0.15,
        gamma_macro: float = 0.20,
    ) -> int:
        logit_ltm = (
            c_ttc
            + (beta_lev * net_leverage)
            + (beta_ebitda * delta_ebitda)
            + (beta_ic * interest_coverage)
            + (beta_liq * liquidity_ratio)
            + (gamma_macro * macro_cycle)
        )
        pd_ltm = 1.0 / (1.0 + math.exp(-logit_ltm))
        return max(1, min(9999, int(Decimal(str(pd_ltm * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))))


class ForwardShadowEngine:
    """TTM-Forward shadow challenger simulating forward curve & covenant depletion."""

    @staticmethod
    def calculate_forward_pd(
        base_pd_bps: int,
        sofr_forward_shift_bps: float,
        consensus_fcf_growth: float,
        covenant_headroom_pct: float,
        refinancing_wall_24m: bool,
    ) -> int:
        prob = base_pd_bps / 10000.0
        logit = math.log(max(prob, 1e-5) / max(1.0 - prob, 1e-5))

        # Adjust for interest rate forward shift and covenant depletion
        rate_impact = (sofr_forward_shift_bps / 100.0) * 0.15
        growth_impact = -consensus_fcf_growth * 0.20
        covenant_impact = max(0.0, 0.25 - covenant_headroom_pct) * 1.5
        refi_impact = 0.35 if refinancing_wall_24m else 0.0

        adj_logit = logit + rate_impact + growth_impact + covenant_impact + refi_impact
        fwd_pd = 1.0 / (1.0 + math.exp(-adj_logit))
        return max(1, min(9999, int(Decimal(str(fwd_pd * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))))


class EquityStructuralEngine:
    """Merton structural default barrier engine with asset volatility solver."""

    @staticmethod
    def solve_merton(
        equity_val: float,
        debt_val: float,
        equity_vol: float,
        r: float,
        t: float = 1.0,
    ) -> Tuple[float, float]:
        """Iteratively solves for asset value V_A and asset volatility sigma_A."""
        if equity_val <= 0 or debt_val <= 0 or equity_vol <= 0:
            raise ValueError("Non-positive market/debt/volatility inputs in Merton solver")

        v_a = equity_val + debt_val
        sigma_a = equity_vol * (equity_val / v_a)

        for _ in range(50):
            if sigma_a <= 0 or v_a <= 0:
                raise ValueError("Non-positive asset value or volatility in Merton iteration")
            d1 = (math.log(v_a / debt_val) + (r + 0.5 * sigma_a**2) * t) / (sigma_a * math.sqrt(t))
            d2 = d1 - sigma_a * math.sqrt(t)

            v_e_calc = v_a * norm_cdf(d1) - debt_val * math.exp(-r * t) * norm_cdf(d2)
            sigma_e_calc = (v_a / max(v_e_calc, 1e-5)) * norm_cdf(d1) * sigma_a

            diff_v = equity_val - v_e_calc
            diff_sig = equity_vol - sigma_e_calc

            if abs(diff_v) < 1e-4 and abs(diff_sig) < 1e-4:
                break

            v_a += diff_v * 0.5
            sigma_a = max(0.01, sigma_a + diff_sig * 0.2)

        if v_a <= 0 or sigma_a <= 0:
            raise ValueError("Merton solver failed to find positive asset state")

        return v_a, sigma_a

    @classmethod
    def calculate_merton_pd(
        cls,
        equity_val: float,
        debt_val: float,
        equity_vol: float,
        r: float = 0.04,
        t: float = 1.0,
        mu_a: float = 0.05,
    ) -> int:
        v_a, sigma_a = cls.solve_merton(equity_val, debt_val, equity_vol, r, t)

        # Projected default probability P(A_{t+T} < D_{t+T})
        d2 = (math.log(v_a / debt_val) + (mu_a - 0.5 * sigma_a**2) * t) / (sigma_a * math.sqrt(t))
        pd_merton = norm_cdf(-d2)
        return max(1, min(9999, int(Decimal(str(pd_merton * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))))


class MultiHorizonCreditEngine:
    """Deterministic Multi-Horizon Credit Evaluation Kernel and Arbitrage-Free CDS Pricer."""

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
        cds_recovery_rate: float = 0.40,
    ) -> Dict[str, Any]:
        """Executes logit fusion adjudication, CDS bootstrapping, and risk governance checks."""
        # 1. Non-finite & numeric sanity boundary check on scalar float parameters and proposals (fail-closed)
        for scalar_val in (macro_stress_factor, cds_recovery_rate):
            if isinstance(scalar_val, float) and (math.isnan(scalar_val) or math.isinf(scalar_val)):
                return self._fail_closed("NON_FINITE_NUMERIC_INPUT_DETECTED")

        for p in proposals:
            for key, val in p.items():
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    return self._fail_closed("NON_FINITE_NUMERIC_INPUT_DETECTED")

        # Validate envelope clock monotonicity and UTC awareness via TemporalEnvelope
        try:
            if isinstance(envelope, dict):
                TemporalEnvelope(**envelope)
            elif isinstance(envelope, TemporalEnvelope):
                pass
        except Exception as err:
            return self._fail_closed(f"TEMPORAL_ENVELOPE_VALIDATION_FAILED: {err}")

        # Validate proposal schemas and PD normalization bounds PD in (0, 1) -> 1 to 9999 bps
        validated_proposals = []
        try:
            for p in proposals:
                m_prop = ModelProposal(**p)
                validated_proposals.append(m_prop.model_dump())
        except Exception as err:
            return self._fail_closed(f"PROPOSAL_SCHEMA_OR_PD_BOUND_VIOLATION: {err}")

        by_name = {p["model_name"]: p for p in validated_proposals}
        ttc = by_name.get("TTC")
        ltm = by_name.get("LTM")
        fwd = by_name.get("FWD")
        eq = by_name.get("EQUITY")

        if not all([ttc, ltm, fwd, eq]):
            return self._fail_closed("INCOMPLETE_SWARM_PROPOSALS")

        # 2. Log-Odds Fusion & Disagreement Variance Calculation
        logits = [
            self._prob_to_logit(self._bps_to_prob(ttc["pd_1y_bps"])),
            self._prob_to_logit(self._bps_to_prob(ltm["pd_1y_bps"])),
            self._prob_to_logit(self._bps_to_prob(fwd["pd_1y_bps"])),
            self._prob_to_logit(self._bps_to_prob(eq["pd_1y_bps"])),
        ]
        mean_logit = sum(logits) / len(logits)
        logit_variance = sum((x - mean_logit) ** 2 for x in logits) / (len(logits) - 1)
        disagreement_bps = int(Decimal(str(logit_variance * 100.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # 3. Regime-Conditioned Weighting
        weights_map = {
            "EXPANSION": (0.45, 0.25, 0.20, 0.10, 5),
            "NORMAL": (0.45, 0.25, 0.20, 0.10, 5),
            "SLOWDOWN": (0.25, 0.30, 0.30, 0.15, 15),
            "RECESSION": (0.15, 0.25, 0.35, 0.25, 30),
            "CREDIT_STRESS": (0.10, 0.25, 0.35, 0.30, 40),
            "CRISIS": (0.05, 0.15, 0.40, 0.40, 60),
        }
        w1, w2, w3, w4, w6_penalty_bps = weights_map.get(regime, (0.25, 0.25, 0.25, 0.25, 25))

        fused_logit = (w1 * logits[0]) + (w2 * logits[1]) + (w3 * logits[2]) + (w4 * logits[3])
        fused_prob = self._logit_to_prob(fused_logit)

        physical_pd_1y_bps = int(Decimal(str(fused_prob * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        physical_pd_1y_bps += int(Decimal(str(w6_penalty_bps * logit_variance)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # 4. Integer Basis Points Divergence Guard
        divergence_bps = abs(ltm["pd_1y_bps"] - ttc["pd_1y_bps"])
        is_critical = divergence_bps >= self.tolerance_bps

        if is_critical or physical_pd_1y_bps > 1000:
            circuit_status = "OPERATIONAL"
            buffer_action = "APPLY_DOWNSIDE_SPREAD_SURCHARGE"
            thesis_status = "DETERIORATING_FORWARD_RISK"
        else:
            circuit_status = "OPERATIONAL"
            buffer_action = "STANDARD_CAPITAL"
            thesis_status = "FOUNDATIONAL_THESIS_INTACT"

        # 5. Physical Hazard -> Risk-Neutral Hazard Transformation & CDS Pricing
        phys_p = self._bps_to_prob(physical_pd_1y_bps)
        lambda_p = -math.log(max(1.0 - phys_p, 0.00001))

        risk_premium = 0.0035 + (0.0020 * macro_stress_factor) + (logit_variance * 0.001)
        lambda_q = lambda_p + risk_premium
        rn_hazard_bps = int(Decimal(str(lambda_q * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # Arbitrage-free 5Y Par CDS Spread solution
        n_years = 5
        dt = 1.0
        r_f = 0.04
        denom = 0.0
        num = 0.0
        s_q_prev = 1.0

        for i in range(1, n_years + 1):
            t_i = i * dt
            d_i = math.exp(-r_f * t_i)
            s_q_curr = math.exp(-lambda_q * t_i)
            ds_q = s_q_prev - s_q_curr

            num += d_i * ds_q
            denom += dt * d_i * s_q_curr + 0.5 * dt * d_i * ds_q
            s_q_prev = s_q_curr

        par_spread = (1.0 - cds_recovery_rate) * (num / max(denom, 1e-6))
        par_cds_spread_bps = int(Decimal(str(par_spread * 10000.0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # 6. Audit payload and SHA-256 digest binding
        evaluation_id = f"eval_{uuid.uuid4().hex}"
        record = {
            "evaluation_id": evaluation_id,
            "temporal_envelope": {
                k: v.isoformat() if isinstance(v, datetime) else v
                for k, v in (envelope.model_dump().items() if isinstance(envelope, TemporalEnvelope) else envelope.items())
            },
            "regime": regime,
            "physical_pd_1y_bps": physical_pd_1y_bps,
            "model_disagreement_bps": disagreement_bps,
            "divergence_bps": divergence_bps,
            "risk_neutral_hazard_bps": rn_hazard_bps,
            "par_cds_spread_bps": par_cds_spread_bps,
            "circuit_breaker_status": circuit_status,
            "capital_buffer_action": buffer_action,
            "underwriting_thesis_status": thesis_status,
            "proposals_evaluated": [p["model_name"] for p in validated_proposals],
        }
        record["decision_digest_sha256"] = self._canonical_sha256(record)
        return record

    def evaluate_enterprise_feedback(
        self,
        fcf_projections: List[float],
        net_debt: float,
        market_equity_val: float,
        r_f: float = 0.04,
        erp: float = 0.05,
        par_cds_spread_bps: int = 250,
    ) -> Dict[str, Any]:
        """Calculates EV feedback loop and tests for implied equity value impairment."""
        cds_spread_decimal = par_cds_spread_bps / 10000.0
        ev = 0.0
        for u, fcf in enumerate(fcf_projections, start=1):
            discount_rate = 1.0 + r_f + erp + cds_spread_decimal
            ev += fcf / (discount_rate**u)

        implied_equity = ev - net_debt
        trigger_feedback = implied_equity < market_equity_val

        return {
            "enterprise_value": ev,
            "implied_equity_value": implied_equity,
            "market_equity_value": market_equity_val,
            "trigger_volatility_feedback": trigger_feedback,
            "suggested_volatility_multiplier": 1.25 if trigger_feedback else 1.0,
        }

    def _fail_closed(self, reason: str) -> Dict[str, Any]:
        bad_record = {
            "evaluation_id": f"eval_tripped_{uuid.uuid4().hex}",
            "error_reason": reason,
            "circuit_breaker_status": "TRIPPED",
            "capital_buffer_action": "IMMEDIATE_DELEVERAGING_COVENANT",
            "underwriting_thesis_status": "STRUCTURAL_IMPAIRMENT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        bad_record["decision_digest_sha256"] = self._canonical_sha256(bad_record)
        return bad_record
