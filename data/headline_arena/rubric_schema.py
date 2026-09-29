"""rubric_schema.py - Deterministic Verification and Adversarial Gate Schemas."""

from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class RationaleComponentScore(BaseModel):
  model_config = ConfigDict(strict=True, extra='forbid')
  dimension: str
  score: int = Field(..., ge=1, le=5, description="1 (Deficient) to 5 (Exemplary)")
  strengths: str
  deficiencies: str
  passed_gate: bool


class EvaluationRubricResult(BaseModel):
  model_config = ConfigDict(strict=True, extra='forbid')
  challenge_id: str
  causal_grounding: RationaleComponentScore
  transmission_mechanics: RationaleComponentScore
  counterfactual_falsification: RationaleComponentScore
  epistemic_calibration: RationaleComponentScore
  aggregate_score: float = Field(0.0, ge=0.0, le=100.0)
  audit_verdict: Literal["APPROVE", "REJECT_FOR_REFINEMENT"] = "APPROVE"
  rejection_reasons: List[str] = Field(default_factory=list)

  @model_validator(mode="after")
  def calculate_verdict(self) -> "EvaluationRubricResult":
    raw_sum = (
        self.causal_grounding.score
        + self.transmission_mechanics.score
        + self.counterfactual_falsification.score
        + self.epistemic_calibration.score
    )
    # Scaled percentage formula: ((Sum - 4) / 16) * 100
    calculated_aggregate = ((raw_sum - 4) / 16.0) * 100.0
    self.aggregate_score = round(calculated_aggregate, 2)

    failures = []
    for dim in [
        self.causal_grounding,
        self.transmission_mechanics,
        self.counterfactual_falsification,
        self.epistemic_calibration,
    ]:
      if dim.score < 3:
        failures.append(
            f"Dimension '{dim.dimension}' scored below acceptable floor"
            f" ({dim.score}/5): {dim.deficiencies}"
        )

    if self.aggregate_score < 75.0:
      failures.append(
          f"Aggregate score {self.aggregate_score:.1f}% below minimum 75.0%"
          " barrier."
      )

    if failures:
      self.audit_verdict = "REJECT_FOR_REFINEMENT"
      self.rejection_reasons = failures
    else:
      self.audit_verdict = "APPROVE"

    return self


class MacroForecastSubmission(BaseModel):
  model_config = ConfigDict(strict=True, extra='forbid')
  challenge_id: str
  target_asset: str
  direction: Literal["bullish", "bearish", "neutral"]
  confidence: float = Field(..., ge=0.50, le=1.00)
  point_forecast: Optional[float] = None
  std_deviation: Optional[float] = None
  rationale: Dict[str, str]
  summary_statement: str

  @field_validator("confidence")
  @classmethod
  def check_confidence_precision(cls, v: float) -> float:
    return round(v, 4)

  @model_validator(mode="after")
  def validate_epistemic_depth(self) -> "MacroForecastSubmission":
    required_keys = {
        "causal_grounding",
        "transmission_mechanism",
        "counterfactual_falsification",
        "calibration_and_sizing",
    }
    missing = required_keys - set(self.rationale.keys())
    if missing:
      raise ValueError(f"Rationale missing required scoring sections: {missing}")

    for key, text in self.rationale.items():
      word_count = len(text.strip().split())
      if word_count < 15:
        raise ValueError(
            f"Rationale section '{key}' failed depth check ({word_count} words"
            " < minimum 15 words)"
        )
    return self


JUDGE_RUBRIC_PROMPT_TEMPLATE = """
You are the Headline Arena Adversarial Rationale Judge. Evaluate this macroeconomic prediction across four objective dimensions.
Output ONLY a JSON payload conforming to the EvaluationRubricResult schema.

EVALUATION AXES:
1. Causal Grounding: Does it isolate balance-sheet, regulatory, flow-of-funds, or liquidity catalysts? (1 = Pure price action / chatter; 5 = Rigorous structural drivers).
2. Transmission Mechanics: Is the step-by-step pathway from catalyst to settlement price explicit? (1 = Hand-wavy jump; 5 = Unbroken mechanical transmission).
3. Counterfactual Falsification: Are explicit, falsifiable conditions and tail risks defined? (1 = Vague/None; 5 = Actionable quantitative invalidation triggers).
4. Epistemic Calibration: Does confidence level ({confidence}) mathematically reflect volatility, binary event risks, and sizing? (1 = Massive over/underconfidence; 5 = Perfectly calibrated).

SUBMISSION PAYLOAD:
Asset: {target_asset}
Direction: {direction} (Confidence: {confidence})
Point Forecast: {point_forecast} (Sigma: {std_deviation})
Rationale:
{rationale_json}
"""
