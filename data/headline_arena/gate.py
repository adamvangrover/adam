"""
ADAM-Macro-Sentinel Pre-Submission Gate
========================================
Validates forecasts against all quality gates before API submission:

1. Schema validation (Pydantic enforcement)
2. S_rat >= 75.0 (Adversarial Epistemic Rationale Score)
3. CRPS confidence-distribution coherence
4. Negative constraint enforcement (zero fluff, no narrative momentum)
5. Champion-Challenger rebuttal completeness
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import Direction, ForecastSubmission, GateVerdict
from .scoring import (
    compute_s_rat,
    validate_crps_confidence_coherence,
)


@dataclass
class GateResult:
    """Aggregated gate validation result."""
    forecast: ForecastSubmission
    schema_valid: bool = True
    s_rat_verdict: GateVerdict | None = None
    crps_coherent: bool = True
    crps_message: str = ""
    negative_constraint_violations: list[str] = field(default_factory=list)
    arbitration_complete: bool = True
    overall_pass: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_summary(self) -> dict:
        return {
            "challenge_id": self.forecast.challenge_id,
            "overall_pass": self.overall_pass,
            "s_rat_score": self.s_rat_verdict.s_rat_score if self.s_rat_verdict else 0.0,
            "s_rat_pass": self.s_rat_verdict.passed if self.s_rat_verdict else False,
            "schema_valid": self.schema_valid,
            "crps_coherent": self.crps_coherent,
            "negative_violations": len(self.negative_constraint_violations),
            "failure_reasons": self.failure_reasons,
            "recommendations": self.recommendations,
        }


# ─── Negative Constraint Patterns ────────────────────────────────────────────

FLUFF_PATTERNS = [
    r"(?i)^(hello|hi|hey|greetings|good\s(morning|afternoon|evening))",
    r"(?i)as an ai",
    r"(?i)as a language model",
    r"(?i)i'm just an",
    r"(?i)let me (think|consider|analyze)",
    r"(?i)in conclusion",
    r"(?i)to summarize",
    r"(?i)it('s| is) (important|worth) (to note|noting)",
    r"(?i)it should be noted",
    r"(?i)disclaimer",
    r"(?i)this is not financial advice",
]

NARRATIVE_MOMENTUM_PATTERNS = [
    # Generic "sentiment" without positioning proxy
    r"(?i)market sentiment(?!.*(?:CFTC|CoT|positioning|gamma|dealer|put.?call))",
    r"(?i)investor sentiment(?!.*(?:CFTC|CoT|positioning|flow))",
    # Circular momentum claims
    r"(?i)it has been (rallying|falling|rising|declining) (?!.*because)",
    r"(?i)momentum suggests",
    r"(?i)the trend (is|seems|appears) (likely|probable)",
    r"(?i)markets (believe|think|feel|expect)(?!.*because)",
]


class PreSubmissionGate:
    """
    Multi-stage validation gate that must be passed before any forecast
    is submitted to the Headline Arena API.
    """

    def __init__(
        self,
        s_rat_threshold: float = 75.0,
        enforce_crps: bool = True,
        enforce_negative_constraints: bool = True,
    ):
        self.s_rat_threshold = s_rat_threshold
        self.enforce_crps = enforce_crps
        self.enforce_negative_constraints = enforce_negative_constraints

    def validate(
        self,
        forecast: ForecastSubmission,
        reference_price: float | None = None,
        arbitration_complete: bool = True,
    ) -> GateResult:
        """
        Run all validation gates on a forecast submission.
        Returns a GateResult indicating pass/fail with detailed diagnostics.
        """
        result = GateResult(forecast=forecast)

        # Gate 1: Schema validation (already enforced by Pydantic, but check)
        try:
            forecast.to_api_payload()
            result.schema_valid = True
        except Exception as e:
            result.schema_valid = False
            result.failure_reasons.append(f"Schema validation failed: {e}")

        # Gate 2: S_rat scoring
        s_rat_verdict = compute_s_rat(forecast)
        result.s_rat_verdict = s_rat_verdict
        if not s_rat_verdict.passed:
            result.failure_reasons.append(
                f"S_rat score {s_rat_verdict.s_rat_score:.1f} < "
                f"threshold {self.s_rat_threshold:.1f}"
            )
            result.recommendations.extend(s_rat_verdict.recommendations)

        # Gate 3: CRPS confidence-distribution coherence
        if (
            self.enforce_crps
            and forecast.point_forecast is not None
            and forecast.std_deviation is not None
            and reference_price is not None
        ):
            coherent, msg = validate_crps_confidence_coherence(
                forecast.confidence,
                forecast.point_forecast,
                forecast.std_deviation,
                reference_price,
                forecast.direction,
            )
            result.crps_coherent = coherent
            result.crps_message = msg
            if not coherent:
                result.failure_reasons.append(f"CRPS coherence failure: {msg}")
                result.recommendations.append(
                    "Adjust σ so that P(Settlement > Strike) matches "
                    "the binary directional confidence c."
                )
        else:
            result.crps_coherent = True

        # Gate 4: Negative constraints
        if self.enforce_negative_constraints:
            violations = self._check_negative_constraints(forecast)
            result.negative_constraint_violations = violations
            if violations:
                result.failure_reasons.extend(violations)

        # Gate 5: Arbitration completeness
        result.arbitration_complete = arbitration_complete
        if not arbitration_complete:
            result.failure_reasons.append(
                "Champion-Challenger arbitration was not completed."
            )

        # Overall determination
        result.overall_pass = len(result.failure_reasons) == 0

        return result

    def _check_negative_constraints(
        self, forecast: ForecastSubmission
    ) -> list[str]:
        """
        Check for forbidden patterns across all text fields.
        Enforces: ZERO FLUFF, NO NARRATIVE MOMENTUM.
        """
        violations = []

        # Collect all text fields
        text_fields = {
            "causal_grounding": forecast.rationale.causal_grounding,
            "transmission_mechanism": forecast.rationale.transmission_mechanism,
            "counterfactual_falsification": forecast.rationale.counterfactual_falsification,
            "calibration_and_sizing": forecast.rationale.calibration_and_sizing,
            "summary_statement": forecast.summary_statement,
        }

        for field_name, text in text_fields.items():
            # Fluff check
            for pattern in FLUFF_PATTERNS:
                if re.search(pattern, text):
                    violations.append(
                        f"FLUFF detected in {field_name}: "
                        f"matches pattern '{pattern}'"
                    )

            # Narrative momentum check
            for pattern in NARRATIVE_MOMENTUM_PATTERNS:
                if re.search(pattern, text):
                    violations.append(
                        f"NARRATIVE MOMENTUM in {field_name}: "
                        f"cites sentiment without positioning proxy"
                    )

        return violations

    def validate_batch(
        self,
        forecasts: list[ForecastSubmission],
        reference_prices: dict[str, float] | None = None,
    ) -> list[GateResult]:
        """Validate a batch of forecasts. Returns list of GateResults."""
        results = []
        for forecast in forecasts:
            ref_price = None
            if reference_prices:
                ref_price = reference_prices.get(forecast.challenge_id)
            result = self.validate(forecast, reference_price=ref_price)
            results.append(result)
        return results

    def summary_report(self, results: list[GateResult]) -> str:
        """Generate a human-readable gate validation report."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║  ADAM-Macro-Sentinel Pre-Submission Gate Report             ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
        ]

        passed = sum(1 for r in results if r.overall_pass)
        failed = len(results) - passed

        lines.append(f"  Total forecasts:  {len(results)}")
        lines.append(f"  Passed:           {passed} ✓")
        lines.append(f"  Failed:           {failed} ✗")
        lines.append("")

        for r in results:
            status = "✓ PASS" if r.overall_pass else "✗ FAIL"
            s_rat = r.s_rat_verdict.s_rat_score if r.s_rat_verdict else 0
            lines.append(
                f"  [{status}] {r.forecast.challenge_id[:12]}… "
                f"({r.forecast.target_asset}) "
                f"S_rat={s_rat:.1f} "
                f"{'📊' if r.crps_coherent else '⚠️'}"
            )

            if not r.overall_pass:
                for reason in r.failure_reasons:
                    lines.append(f"         ↳ {reason}")

        lines.append("")
        lines.append(f"  Gate threshold: S_rat >= {self.s_rat_threshold}")
        return "\n".join(lines)
