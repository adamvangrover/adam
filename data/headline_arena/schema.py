"""
ADAM-Macro-Sentinel Schema Definitions
======================================
Pydantic models enforcing strict JSON schema compliance for all I/O boundaries.
Implements the Response Contract from the hardened directive.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Direction(str, Enum):
    """Market directional bias."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RationaleBlock(BaseModel):
    """
    Structured 4-sub-dimension rationale architecture.
    Each field maps to a scoring sub-dimension evaluated by the LLM Judge.
    """
    causal_grounding: str = Field(
        ...,
        min_length=100,
        description=(
            "Primary catalyst and structural macro driver. Must identify "
            "the exogenous catalyst or liquidity impulse — reject superficial "
            "correlation, generic price momentum, or circular claims."
        ),
    )
    transmission_mechanism: str = Field(
        ...,
        min_length=100,
        description=(
            "Step-by-step causal path from catalyst to asset settlement. "
            "Trace: Catalyst → Rate/Spread/Flow Transmission → Dealer "
            "Inventory/Risk Capacity → Terminal Price Settlement."
        ),
    )
    counterfactual_falsification: str = Field(
        ...,
        min_length=80,
        description=(
            "Asymmetric risks and explicit invalidation thresholds. "
            "Must state what observation nullifies the thesis prior to "
            "settlement. Include a testable invalidation condition."
        ),
    )
    calibration_and_sizing: str = Field(
        ...,
        min_length=80,
        description=(
            "Mathematical coherence between confidence level and the "
            "volatility regime. High confidence requires multi-engine "
            "signal concurrence. Low confidence must document parameter "
            "volatility or binary event risk."
        ),
    )

    def sub_dimension_lengths(self) -> dict[str, int]:
        """Return character counts per sub-dimension for quality audit."""
        return {
            "causal_grounding": len(self.causal_grounding),
            "transmission_mechanism": len(self.transmission_mechanism),
            "counterfactual_falsification": len(self.counterfactual_falsification),
            "calibration_and_sizing": len(self.calibration_and_sizing),
        }


class ForecastSubmission(BaseModel):
    """
    Complete forecast submission adhering to the strict JSON schema
    from the Response Contract.
    """
    challenge_id: str = Field(..., description="Headline Arena challenge UUID")
    target_asset: str = Field(..., description="Asset ticker or identifier")
    direction: Direction
    confidence: float = Field(
        ...,
        ge=0.50,
        le=1.00,
        description="Subjective probability c ∈ [0.50, 1.00]",
    )
    point_forecast: Optional[float] = Field(
        None, description="μ — point estimate for CRPS-scored challenges"
    )
    std_deviation: Optional[float] = Field(
        None,
        gt=0.0,
        description="σ — dispersion for CRPS-scored challenges",
    )
    rationale: RationaleBlock
    summary_statement: str = Field(
        ...,
        min_length=40,
        max_length=500,
        description="Concise 2-sentence institutional synthesis",
    )

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return round(max(0.50, min(1.00, v)), 4)

    @model_validator(mode="after")
    def validate_confidence_direction_coherence(self) -> "ForecastSubmission":
        """
        Neutral direction should not have extreme confidence.
        Bullish/bearish below 0.55 is suspicious but allowed.
        """
        if self.direction == Direction.NEUTRAL and self.confidence > 0.70:
            raise ValueError(
                f"Neutral direction with confidence={self.confidence:.2f} "
                "is incoherent. Neutral should have c <= 0.70."
            )
        return self

    def to_api_payload(self) -> dict:
        """Serialize to the Headline Arena API submission format."""
        payload = {
            "challenge_id": self.challenge_id,
            "target_asset": self.target_asset,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "point_forecast": self.point_forecast,
            "std_deviation": self.std_deviation,
            "rationale": {
                "causal_grounding": self.rationale.causal_grounding,
                "transmission_mechanism": self.rationale.transmission_mechanism,
                "counterfactual_falsification": self.rationale.counterfactual_falsification,
                "calibration_and_sizing": self.rationale.calibration_and_sizing,
            },
            "summary_statement": self.summary_statement,
        }
        return payload

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON for audit trail."""
        canonical = json.dumps(self.to_api_payload(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


class SettlementResult(BaseModel):
    """Post-settlement record linking forecast to market outcome."""
    challenge_id: str
    target_asset: str
    predicted_direction: Direction
    actual_direction: Direction
    confidence: float
    point_forecast: Optional[float] = None
    std_deviation: Optional[float] = None
    settlement_price: Optional[float] = None
    reference_price: Optional[float] = None
    s_dir_score: Optional[float] = None
    brier_score: Optional[float] = None
    crps_score: Optional[float] = None
    settled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    forecast_hash: str = ""


class EpistemicMemoryEntry(BaseModel):
    """
    Single entry in the Epistemic Memory Ledger.
    Tracks calibration history for Platt scaling parameter updates.
    """
    entry_id: str
    challenge_id: str
    target_asset: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_confidence: float = Field(..., ge=0.0, le=1.0)
    calibrated_confidence: float = Field(..., ge=0.50, le=1.00)
    predicted_direction: Direction
    actual_direction: Optional[Direction] = None
    correct: Optional[bool] = None
    s_dir_score: Optional[float] = None
    brier_residual: Optional[float] = None
    platt_a: float = Field(default=1.0, description="Platt scaling parameter a")
    platt_b: float = Field(default=0.0, description="Platt scaling parameter b")
    rationale_score: Optional[float] = None
    arbitration_winner: Optional[str] = None  # "champion" | "challenger"
    notes: str = ""


class GateVerdict(BaseModel):
    """Pre-submission gate validation result."""
    passed: bool
    s_rat_score: float = Field(..., ge=0.0, le=100.0)
    sub_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores per sub-dimension: d_j ∈ {1..5}",
    )
    violations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    gate_threshold: float = 75.0

    @model_validator(mode="after")
    def check_threshold(self) -> "GateVerdict":
        self.passed = self.s_rat_score >= self.gate_threshold
        return self
