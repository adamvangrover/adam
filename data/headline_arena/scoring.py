"""
ADAM-Macro-Sentinel Scoring Module
===================================
Implements the four mathematical scoring functions from the specification:

1. Confidence-Weighted Directional Settlement Score (S_dir)
2. Brier Calibration Loss (BS)
3. Continuous Ranked Probability Score (CRPS)
4. Adversarial Epistemic Rationale Score (S_rat)
"""

from __future__ import annotations

import math
from typing import Optional

from .schema import Direction, ForecastSubmission, GateVerdict


# ─── 1. Confidence-Weighted Directional Settlement Score ──────────────────────

def compute_s_dir(
    confidence: float,
    predicted_direction: Direction,
    actual_direction: Direction,
) -> float:
    """
    S_dir = 50 + 50 * c  if correct
    S_dir = 50 - 50 * c  if incorrect

    Where c ∈ [0.50, 1.00] is the declared subjective confidence.
    Overconfidence on incorrect predictions penalizes returns toward 0.
    Underconfidence on correct calls caps upside.

    Returns:
        Score in [0, 100].
    """
    c = max(0.50, min(1.00, confidence))
    correct = _directions_match(predicted_direction, actual_direction)
    if correct:
        return 50.0 + 50.0 * c
    else:
        return 50.0 - 50.0 * c


def _directions_match(predicted: Direction, actual: Direction) -> bool:
    """Check if predicted direction matches actual settlement direction."""
    if predicted == Direction.NEUTRAL:
        # Neutral is "correct" only if actual is also neutral
        return actual == Direction.NEUTRAL
    return predicted == actual


# ─── 2. Brier Calibration Loss ────────────────────────────────────────────────

def compute_brier_score(
    confidence: float,
    predicted_direction: Direction,
    actual_direction: Direction,
) -> float:
    """
    BS = (f - o)²

    Where:
        f = predicted probability of the event occurring
        o = 1 if event occurred, 0 otherwise

    For directional forecasting:
        f = confidence if predicting the correct direction
        o = 1 if the predicted direction matches actual, else 0

    Returns:
        Brier score in [0, 1]. Lower is better.
    """
    correct = _directions_match(predicted_direction, actual_direction)

    if predicted_direction == Direction.NEUTRAL:
        # For neutral predictions, the "event" is whether the market moves
        # We score against the probability of being correct
        f = confidence
        o = 1.0 if correct else 0.0
    else:
        f = confidence
        o = 1.0 if correct else 0.0

    return (f - o) ** 2


def compute_rolling_brier(
    scores: list[float],
    window: int = 20,
) -> list[float]:
    """
    Compute rolling average Brier score over a sliding window.
    Used for multi-day calibration assessment.
    """
    if not scores:
        return []
    rolling = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        window_scores = scores[start : i + 1]
        rolling.append(sum(window_scores) / len(window_scores))
    return rolling


# ─── 3. Continuous Ranked Probability Score (CRPS) ────────────────────────────

def compute_crps_gaussian(
    mu: float,
    sigma: float,
    settlement_price: float,
) -> float:
    """
    CRPS for a Gaussian predictive distribution F ~ N(μ, σ²):

    CRPS(F, x) = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]

    Where:
        z = (x - μ) / σ
        Φ = standard normal CDF
        φ = standard normal PDF

    Lower is better. Units match the asset's price units.
    """
    if sigma <= 0:
        # Degenerate: point forecast with no uncertainty
        return abs(settlement_price - mu)

    z = (settlement_price - mu) / sigma
    phi_z = _standard_normal_pdf(z)
    big_phi_z = _standard_normal_cdf(z)

    crps = sigma * (z * (2.0 * big_phi_z - 1.0) + 2.0 * phi_z - 1.0 / math.sqrt(math.pi))
    return crps


def _standard_normal_pdf(z: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _standard_normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def validate_crps_confidence_coherence(
    confidence: float,
    mu: float,
    sigma: float,
    reference_price: float,
    direction: Direction,
) -> tuple[bool, str]:
    """
    Validate that P(Settlement > Strike) matches the binary directional
    confidence score c.

    For bullish: P(X > reference) should approximate c
    For bearish: P(X < reference) should approximate c
    For neutral: P(|X - reference| < sigma) should be reasonable

    Returns:
        (is_coherent, explanation)
    """
    if sigma <= 0:
        return False, "σ must be positive for CRPS coherence check."

    z = (reference_price - mu) / sigma
    p_above = 1.0 - _standard_normal_cdf(z)
    p_below = _standard_normal_cdf(z)

    tolerance = 0.15  # Allow 15% deviation

    if direction == Direction.BULLISH:
        if abs(p_above - confidence) > tolerance:
            return False, (
                f"Bullish with c={confidence:.2f} but P(X > ref)={p_above:.2f}. "
                f"Gap of {abs(p_above - confidence):.2f} exceeds tolerance {tolerance}."
            )
    elif direction == Direction.BEARISH:
        if abs(p_below - confidence) > tolerance:
            return False, (
                f"Bearish with c={confidence:.2f} but P(X < ref)={p_below:.2f}. "
                f"Gap of {abs(p_below - confidence):.2f} exceeds tolerance {tolerance}."
            )

    return True, "Distributional coherence validated."


# ─── 4. Adversarial Epistemic Rationale Score (S_rat) ─────────────────────────

RATIONALE_SUB_DIMENSIONS = [
    "causal_grounding",
    "transmission_mechanism",
    "counterfactual_falsification",
    "calibration_and_sizing",
]

# Scoring rubric thresholds (character length proxies + structural checks)
QUALITY_RUBRIC = {
    "causal_grounding": {
        "min_length": 150,
        "required_patterns": [
            # Must reference specific catalysts, not vague "sentiment"
        ],
        "forbidden_patterns": [
            "market sentiment",
            "it has been rallying",
            "it has been falling",
            "momentum suggests",
            "As an AI",
        ],
    },
    "transmission_mechanism": {
        "min_length": 150,
        "required_patterns": [],
        "forbidden_patterns": [
            "should go up",
            "should go down",
            "likely to",
        ],
    },
    "counterfactual_falsification": {
        "min_length": 100,
        "required_patterns": [],
        "forbidden_patterns": [],
    },
    "calibration_and_sizing": {
        "min_length": 100,
        "required_patterns": [],
        "forbidden_patterns": [],
    },
}

GATE_THRESHOLD = 75.0


def score_rationale_subdimension(
    dimension: str,
    text: str,
) -> tuple[int, list[str]]:
    """
    Score a single rationale sub-dimension on a 1-5 scale.
    Returns (score, list_of_issues).

    Scoring:
        5 — Exceptional: specific, quantitative, structurally sound
        4 — Strong: well-reasoned with minor gaps
        3 — Adequate: passes minimum bar but lacks depth
        2 — Weak: superficial or partially circular
        1 — Failing: generic, fluff-filled, or incoherent
    """
    issues: list[str] = []
    rubric = QUALITY_RUBRIC.get(dimension, {})
    min_len = rubric.get("min_length", 100)
    forbidden = rubric.get("forbidden_patterns", [])

    score = 3  # Start at adequate

    # Length assessment
    text_len = len(text.strip())
    if text_len < min_len * 0.5:
        score = 1
        issues.append(f"{dimension}: critically short ({text_len} chars, need {min_len})")
    elif text_len < min_len:
        score = max(score - 1, 1)
        issues.append(f"{dimension}: below minimum length ({text_len}/{min_len})")
    elif text_len > min_len * 2:
        score = min(score + 1, 5)

    # Forbidden pattern check
    text_lower = text.lower()
    for pattern in forbidden:
        if pattern.lower() in text_lower:
            score = max(score - 1, 1)
            issues.append(f"{dimension}: contains forbidden pattern '{pattern}'")

    # Structural quality heuristics
    if dimension == "causal_grounding":
        # Should reference specific data points or events
        has_specifics = any(
            marker in text
            for marker in [
                "bps", "basis point", "%", "billion", "million",
                "hike", "cut", "auction", "inventory", "payrolls",
                "CPI", "PCE", "GDP", "PMI", "FOMC", "ECB", "BoJ",
                "BoE", "PBOC", "CFTC", "CoT", "VIX", "MOVE",
            ]
        )
        if has_specifics:
            score = min(score + 1, 5)
        else:
            issues.append(f"{dimension}: lacks specific data references")

    elif dimension == "transmission_mechanism":
        # Should describe a causal chain with multiple steps
        chain_markers = ["→", "->", "leads to", "resulting in", "which causes",
                         "transmits", "flows through", "propagates"]
        has_chain = any(m in text for m in chain_markers)
        if has_chain:
            score = min(score + 1, 5)
        else:
            issues.append(f"{dimension}: lacks explicit causal chain markers")

    elif dimension == "counterfactual_falsification":
        # Should contain explicit invalidation conditions
        invalidation_markers = [
            "invalidated if", "nullified if", "falsified if",
            "breaks above", "breaks below", "exceeds", "fails to hold",
            "if instead", "contrary scenario", "risk of",
        ]
        has_invalidation = any(m.lower() in text_lower for m in invalidation_markers)
        if has_invalidation:
            score = min(score + 1, 5)
        else:
            issues.append(f"{dimension}: lacks explicit invalidation condition")

    elif dimension == "calibration_and_sizing":
        # Should reference confidence level and volatility
        cal_markers = [
            "confidence", "sigma", "volatility", "implied vol",
            "VIX", "MOVE", "OVX", "dispersion", "calibrat",
            "brier", "0.5", "0.6", "0.7", "0.8", "0.9",
        ]
        has_calibration = any(m.lower() in text_lower for m in cal_markers)
        if has_calibration:
            score = min(score + 1, 5)
        else:
            issues.append(f"{dimension}: lacks calibration/volatility references")

    return max(1, min(5, score)), issues


def compute_s_rat(forecast: ForecastSubmission) -> GateVerdict:
    """
    Adversarial Epistemic Rationale Score:

    S_rat = (25 / N) * Σ d_j

    Where d_j ∈ {1, 2, 3, 4, 5} across the four mandatory sub-dimensions.
    N = 4 (number of sub-dimensions)
    Maximum score = 25/4 * (4*5) = 125 → but capped at 100.

    The formula simplifies to: S_rat = 25 * mean(d_j)
    With 4 dimensions at max 5: S_rat_max = 25 * 5 = 125
    But practical max is 100 (capped).

    Pre-submission gate enforces S_rat >= 75.0.
    """
    sub_scores: dict[str, float] = {}
    all_issues: list[str] = []
    recommendations: list[str] = []

    rationale_fields = {
        "causal_grounding": forecast.rationale.causal_grounding,
        "transmission_mechanism": forecast.rationale.transmission_mechanism,
        "counterfactual_falsification": forecast.rationale.counterfactual_falsification,
        "calibration_and_sizing": forecast.rationale.calibration_and_sizing,
    }

    for dim, text in rationale_fields.items():
        score, issues = score_rationale_subdimension(dim, text)
        sub_scores[dim] = float(score)
        all_issues.extend(issues)

    # S_rat = 25 * mean(d_j), capped at 100
    mean_d = sum(sub_scores.values()) / len(sub_scores)
    s_rat = min(100.0, 25.0 * mean_d)

    # Generate recommendations for failing dimensions
    for dim, score in sub_scores.items():
        if score <= 2:
            recommendations.append(
                f"CRITICAL: {dim} scored {score}/5. "
                "Requires deeper structural analysis with specific data points."
            )
        elif score == 3:
            recommendations.append(
                f"IMPROVE: {dim} scored {score}/5. "
                "Add quantitative evidence or explicit causal chain."
            )

    return GateVerdict(
        passed=s_rat >= GATE_THRESHOLD,
        s_rat_score=round(s_rat, 2),
        sub_scores=sub_scores,
        violations=all_issues,
        recommendations=recommendations,
        gate_threshold=GATE_THRESHOLD,
    )
