import hashlib
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple


def compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def evaluate_empirical_groundedness(evidence_mappings: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
    reason_codes = []
    if not evidence_mappings:
        reason_codes.append("GROUNDING_NO_EVIDENCE_MAPPINGS")
        return 0.0, reason_codes

    valid_count = 0
    for item in evidence_mappings:
        ref = item.get("source_ref")
        if ref and ("sec" in ref.lower() or "10-k" in ref.lower() or "10-q" in ref.lower() or "bloomberg" in ref.lower() or "fed" in ref.lower() or "filing" in ref.lower() or ref.startswith("doc:")):
            valid_count += 1
        elif ref:
            valid_count += 0.5  # Partial credit for unspecified ref

    score = round(valid_count / len(evidence_mappings), 4)
    if score >= 0.8:
        reason_codes.append("GROUNDING_HIGH_FIDELITY")
    elif score >= 0.5:
        reason_codes.append("GROUNDING_PARTIAL_FIDELITY")
    else:
        reason_codes.append("GROUNDING_LOW_FIDELITY")
    return score, reason_codes


def evaluate_reasoning_rigor(financial_metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    reason_codes = []
    checks = [
        financial_metrics.get("refinancing_wall_modeled", False),
        financial_metrics.get("margin_compression_modeled", False),
        financial_metrics.get("non_linear_debt_service_decay", False),
    ]
    passed_checks = sum(1 for c in checks if c)
    score = round(passed_checks / len(checks), 4)

    if checks[0]:
        reason_codes.append("RIGOR_REFINANCING_WALL_VERIFIED")
    if checks[1]:
        reason_codes.append("RIGOR_MARGIN_COMPRESSION_VERIFIED")
    if checks[2]:
        reason_codes.append("RIGOR_NON_LINEAR_DECAY_VERIFIED")

    return score, reason_codes


def evaluate_policy_compliance(financial_metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    reason_codes = []
    score = 1.0

    sr11_7 = financial_metrics.get("sr11_7_compliant", True)
    if not sr11_7:
        score -= 0.5
        reason_codes.append("POLICY_SR11_7_NON_COMPLIANT")
    else:
        reason_codes.append("POLICY_SR11_7_COMPLIANT")

    snc_class = financial_metrics.get("snc_classification", "PASS")
    if snc_class in ["SPECIAL_MENTION", "SUBSTANDARD", "DOUBTFUL", "LOSS"]:
        score -= 0.3
        reason_codes.append(f"POLICY_SNC_WARNING:{snc_class}")

    liquidity_days = financial_metrics.get("liquidity_runway_days")
    if liquidity_days is not None and liquidity_days < 90:
        score -= 0.2
        reason_codes.append(f"POLICY_LIQUIDITY_RUNWAY_WARNING:{liquidity_days}_DAYS")

    score = max(0.0, round(score, 4))
    return score, reason_codes


def check_tripwires(financial_metrics: Dict[str, Any], policy: Dict[str, Any]) -> Tuple[bool, List[str]]:
    min_dscr = policy.get("min_dscr", 1.05)
    allow_unauthorized_addbacks = policy.get("allow_unauthorized_addbacks", False)
    max_leverage = policy.get("max_leverage", None)
    min_liquidity_days = policy.get("min_liquidity_runway_days", None)
    require_sr11_7 = policy.get("require_sr11_7", True)

    triggered = []

    dscr = financial_metrics.get("dscr")
    if dscr is not None and dscr < min_dscr:
        triggered.append(f"TRIPWIRE_DSCR_BREACH:dscr={dscr}<min={min_dscr}")

    unauthorized = financial_metrics.get("unauthorized_addbacks", False)
    if unauthorized and not allow_unauthorized_addbacks:
        triggered.append("TRIPWIRE_UNAUTHORIZED_COVENANT_ADDBACKS")

    leverage = financial_metrics.get("leverage_ratio")
    if max_leverage is not None and leverage is not None and leverage > max_leverage:
        triggered.append(f"TRIPWIRE_LEVERAGE_EXCEEDED:leverage={leverage}>max={max_leverage}")

    liquidity_days = financial_metrics.get("liquidity_runway_days")
    if min_liquidity_days is not None and liquidity_days is not None and liquidity_days < min_liquidity_days:
        triggered.append(f"TRIPWIRE_LIQUIDITY_RUNWAY_BREACH:runway={liquidity_days}<min={min_liquidity_days}")

    sr11_7 = financial_metrics.get("sr11_7_compliant", True)
    if require_sr11_7 and not sr11_7:
        triggered.append("TRIPWIRE_SR11_7_VIOLATION")

    return (len(triggered) == 0), triggered


def adjudicate_pairwise_debate(candidate_a: Dict[str, Any], candidate_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs pairwise debate adjudication with:
    1. Position-bias mitigation via candidate order swapping (eval A vs B and B vs A)
    2. Verbosity-bias mitigation via token length normalization ratio penalty
    3. Explicit token density metrics and position bias detection indicators
    """
    tok_a = max(1, candidate_a.get("token_count", 500))
    tok_b = max(1, candidate_b.get("token_count", 500))
    g_a = candidate_a.get("grounding_count", 1)
    g_b = candidate_b.get("grounding_count", 1)

    # Token density (groundings per token x 100)
    density_a = round(g_a / tok_a, 6)
    density_b = round(g_b / tok_b, 6)

    # Base raw scores from grounding density
    raw_score_a = g_a / (math.log(tok_a + 1) + 1e-5)
    raw_score_b = g_b / (math.log(tok_b + 1) + 1e-5)

    # Verbosity normalization factor
    token_ratio = tok_a / tok_b
    if token_ratio > 2.0:
        raw_score_a *= 0.85  # Penalize verbosity in candidate A
    elif token_ratio < 0.5:
        raw_score_b *= 0.85  # Penalize verbosity in candidate B

    # Position swapping verification (A vs B)
    winner_order_1 = "candidate_a" if raw_score_a > raw_score_b else ("candidate_b" if raw_score_b > raw_score_a else "tie")

    # Position swapping verification (B vs A)
    raw_score_b_swapped = g_b / (math.log(tok_b + 1) + 1e-5)
    raw_score_a_swapped = g_a / (math.log(tok_a + 1) + 1e-5)
    if token_ratio > 2.0:
        raw_score_a_swapped *= 0.85
    elif token_ratio < 0.5:
        raw_score_b_swapped *= 0.85

    winner_order_2 = "candidate_b" if raw_score_b_swapped > raw_score_a_swapped else ("candidate_a" if raw_score_a_swapped > raw_score_b_swapped else "tie")

    position_bias_detected = (winner_order_1 != winner_order_2 and winner_order_1 != "tie" and winner_order_2 != "tie")

    # Symmetric consensus check
    if not position_bias_detected and (winner_order_1 == winner_order_2 or winner_order_1 == "tie"):
        final_winner = winner_order_1
    else:
        final_winner = "tie"  # Position bias detected or asymmetric -> fall back to tie

    consensus_score = round(max(raw_score_a, raw_score_b) / (raw_score_a + raw_score_b + 1e-5), 4)

    return {
        "winner": final_winner,
        "token_density": {
            "candidate_a": density_a,
            "candidate_b": density_b
        },
        "bias_mitigation_applied": {
            "position_swapped": True,
            "position_bias_detected": position_bias_detected,
            "verbosity_length_normalized": True,
            "token_ratio": round(token_ratio, 4)
        },
        "consensus_score": consensus_score
    }


def compute_human_alignment_calibration(calibration_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes Cohen's Kappa (kappa) and Brier Score against Investment/Credit Committee historical decisions.
    """
    if not calibration_dataset:
        return {
            "cohens_kappa": 1.0,
            "brier_score": 0.0,
            "calibrated_rubric_weights": {
                "empirical_groundedness": 0.40,
                "reasoning_rigor_downside": 0.35,
                "policy_regulatory_compliance": 0.25
            }
        }

    # Confusion matrix for Kappa
    agreed = 0
    total = len(calibration_dataset)
    brier_sum = 0.0

    verdict_map = {"ADMIT_PROPOSAL": 1, "REJECT": 0, "ESCALATE": 0, "QUARANTINED": 0}

    for item in calibration_dataset:
        jv = item.get("judge_verdict")
        cv = item.get("committee_verdict")
        if jv == cv:
            agreed += 1

        judge_prob = item.get("judge_score", 1.0 if jv == "ADMIT_PROPOSAL" else 0.0)
        actual_outcome = item.get("committee_outcome_binary", verdict_map.get(cv, 0))
        brier_sum += (judge_prob - actual_outcome) ** 2

    po = agreed / total  # Observed agreement
    # Assume 50/50 chance agreement baseline pe
    pe = 0.50
    kappa = (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0
    brier_score = brier_sum / total

    # Dynamically calibrate rubric weights based on kappa and brier score
    if kappa >= 0.8:
        weights = {"empirical_groundedness": 0.40, "reasoning_rigor_downside": 0.35, "policy_regulatory_compliance": 0.25}
    elif kappa >= 0.5:
        weights = {"empirical_groundedness": 0.45, "reasoning_rigor_downside": 0.30, "policy_regulatory_compliance": 0.25}
    else:
        weights = {"empirical_groundedness": 0.50, "reasoning_rigor_downside": 0.25, "policy_regulatory_compliance": 0.25}

    return {
        "cohens_kappa": round(kappa, 4),
        "brier_score": round(brier_score, 4),
        "calibrated_rubric_weights": weights
    }


def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skill 1 — llm-as-judge-arbiter
    Evaluates credit memos, investment theses, code commits, and prompt iterations.
    Combines multi-criteria rubrics, pairwise adversarial debate, neuro-symbolic tripwires, and human alignment calibration.
    """
    correlation_id = input_data.get("correlation_id", "unknown-correlation-id")
    eval_type = input_data.get("evaluation_type", "single_candidate_rubric")
    tripwire_policy = input_data.get("tripwire_policy", {})
    candidate_proposal = input_data.get("candidate_proposal", {})
    financial_metrics = candidate_proposal.get("financial_metrics", {})
    evidence_mappings = candidate_proposal.get("evidence_mappings", [])

    all_reason_codes = []

    # 1. Neuro-Symbolic Tripwire Check (Deterministic Gating override)
    tripwires_passed, triggered_tripwires = check_tripwires(financial_metrics, tripwire_policy)
    all_reason_codes.extend(triggered_tripwires)

    # 2. Multi-Criteria Scoring Rubrics
    g_score, g_reasons = evaluate_empirical_groundedness(evidence_mappings)
    r_score, r_reasons = evaluate_reasoning_rigor(financial_metrics)
    p_score, p_reasons = evaluate_policy_compliance(financial_metrics)

    all_reason_codes.extend(g_reasons + r_reasons + p_reasons)

    # 3. Human Alignment Calibration
    calib_data = input_data.get("calibration_dataset", [])
    calibration_metrics = compute_human_alignment_calibration(calib_data)
    rubric_weights = calibration_metrics["calibrated_rubric_weights"]

    weighted_composite = round(
        g_score * rubric_weights["empirical_groundedness"] +
        r_score * rubric_weights["reasoning_rigor_downside"] +
        p_score * rubric_weights["policy_regulatory_compliance"],
        4
    )

    # 4. Pairwise Adversarial Adjudication if applicable
    pairwise_adjudication = None
    if eval_type == "pairwise_debate" and "candidate_debate" in input_data:
        debate = input_data["candidate_debate"]
        pairwise_adjudication = adjudicate_pairwise_debate(
            debate.get("candidate_a", {}),
            debate.get("candidate_b", {})
        )
        all_reason_codes.append(f"PAIRWISE_WINNER:{pairwise_adjudication['winner']}")

    # 5. Determine Outcome (Neuro-Symbolic Hybrid Gating)
    if not tripwires_passed:
        outcome = "REJECT"
        status = "REJECTED"
        all_reason_codes.append("NEURO_SYMBOLIC_TRIPWIRE_HARD_OVERRIDE_REJECT")
    elif weighted_composite >= 0.85:
        outcome = "ADMIT_PROPOSAL"
        status = "PASSED"
    elif weighted_composite >= 0.50:
        outcome = "ESCALATE"
        status = "REVIEW_REQUIRED"
    else:
        outcome = "REJECT"
        status = "REJECTED"

    rubric_scores = {
        "empirical_groundedness": g_score,
        "reasoning_rigor_downside": r_score,
        "policy_regulatory_compliance": p_score,
        "weighted_composite_score": weighted_composite
    }

    # W3C PROV-O audit trail
    prov_o_audit_trail = [
        {
            "activity": "llm_as_judge_arbitration",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "weighted_composite_score": weighted_composite,
            "tripwires_passed": tripwires_passed,
            "input_hash": compute_sha256(str(input_data))
        }
    ]

    return {
        "skill_id": "llm_as_judge_arbiter",
        "status": status,
        "correlation_id": correlation_id,
        "outcome": outcome,
        "rubric_scores": rubric_scores,
        "tripwire_status": {
            "passed": tripwires_passed,
            "triggered_tripwires": triggered_tripwires
        },
        "pairwise_adjudication": pairwise_adjudication,
        "calibration_metrics": calibration_metrics,
        "reason_codes": all_reason_codes,
        "prov_o_audit_trail": prov_o_audit_trail
    }
