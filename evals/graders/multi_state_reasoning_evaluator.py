import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Expected standard execution lifecycle steps for base state machine
STANDARD_EXECUTION_STEPS = ["init", "retrieve", "reason", "validate", "synthesize", "complete"]

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "is", "are", "was", "were",
    "by", "with", "from", "as", "it", "this", "that", "be", "has", "have", "had", "due", "or",
}


class BaseStateMachineEvaluator:
    """
    Evaluates the first-stage state machine (Task Extraction & Deep Research).
    Focuses on State-Trace Fidelity and Adversarial Resiliency.
    """

    def __init__(self, expected_steps: Optional[List[str]] = None):
        self.expected_steps = expected_steps or STANDARD_EXECUTION_STEPS

    def eval_state_trace_fidelity(self, state_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pass state execution logs to evaluate if state transitions were logically sound
        or if vital research steps were skipped or looped improperly.
        """
        if not state_logs:
            return {
                "fidelity_score": 0.0,
                "skipped_steps": self.expected_steps,
                "loop_detected": False,
                "step_count": 0,
                "feedback": "Empty state log provided.",
            }

        executed_steps = [log.get("step") or log.get("state") or log.get("node") for log in state_logs]
        executed_steps = [str(s).lower() for s in executed_steps if s is not None]

        # 1. Check skipped mandatory steps
        skipped_steps = [step for step in self.expected_steps if step not in executed_steps]

        # 2. Check transition ordering fidelity
        order_correct_count = 0
        last_idx = -1
        for step in executed_steps:
            if step in self.expected_steps:
                curr_idx = self.expected_steps.index(step)
                if curr_idx >= last_idx:
                    order_correct_count += 1
                    last_idx = curr_idx

        order_ratio = order_correct_count / max(len(executed_steps), 1)

        # 3. Detect duplicate/repeated step loops (> 3 repeated occurrences)
        step_counts: Dict[str, int] = {}
        loop_detected = False
        for step in executed_steps:
            step_counts[step] = step_counts.get(step, 0) + 1
            if step_counts[step] > 3:
                loop_detected = True

        # Calculate fidelity score
        step_coverage = (len(self.expected_steps) - len(skipped_steps)) / len(self.expected_steps)
        fidelity_score = round(0.5 * step_coverage + 0.3 * order_ratio + (0.2 if not loop_detected else 0.0), 4)

        return {
            "fidelity_score": max(0.0, min(1.0, fidelity_score)),
            "executed_steps": executed_steps,
            "skipped_steps": skipped_steps,
            "loop_detected": loop_detected,
            "step_count": len(executed_steps),
            "feedback": (
                f"Fidelity score {fidelity_score}. Skipped steps: {skipped_steps}. "
                f"Loop detected: {loop_detected}."
            ),
        }

    def eval_adversarial_resiliency(
        self,
        baseline_trace: List[Dict[str, Any]],
        adversarial_trace: List[Dict[str, Any]],
        max_allowed_steps: int = 15,
        min_required_steps: int = 3,
    ) -> Dict[str, Any]:
        """
        Compares state traces when fed baseline expert data versus adversarial data.
        Checks if adversarial inputs cause infinite loops or premature exits.
        """
        baseline_fidelity = self.eval_state_trace_fidelity(baseline_trace)
        adv_fidelity = self.eval_state_trace_fidelity(adversarial_trace)

        adv_step_count = adv_fidelity["step_count"]
        infinite_loop = adv_step_count > max_allowed_steps or adv_fidelity["loop_detected"]

        # Premature exit if adversarial trace terminated with fewer steps than min required or exited on error
        has_error_exit = any(
            log.get("status") in ["error", "failed"] or log.get("error") is not None
            for log in adversarial_trace
        )
        premature_exit = adv_step_count < min_required_steps or (has_error_exit and adv_step_count < len(baseline_trace))

        # Calculate resiliency score
        resiliency_score = 1.0
        penalties = 0.0
        if infinite_loop:
            penalties += 0.5
        if premature_exit:
            penalties += 0.4
        if len(adv_fidelity["skipped_steps"]) > len(baseline_fidelity["skipped_steps"]):
            penalties += 0.2

        resiliency_score = max(0.0, round(resiliency_score - penalties, 4))

        return {
            "resiliency_score": resiliency_score,
            "baseline_step_count": baseline_fidelity["step_count"],
            "adversarial_step_count": adv_step_count,
            "infinite_loop_detected": infinite_loop,
            "premature_exit_detected": premature_exit,
            "feedback": (
                f"Resiliency score {resiliency_score}. Infinite loop: {infinite_loop}, "
                f"Premature exit: {premature_exit}."
            ),
        }


class AggregatorStateMachineEvaluator:
    """
    Evaluates the Aggregator State Machine (Synthesis).
    Verifies fact insertion rate (hallucination detection) and conflict resolution.
    """

    def eval_fact_insertion_rate(
        self, aggregator_summary: str, base_state_outputs: List[str]
    ) -> Dict[str, Any]:
        """
        Verifies if the aggregator injected facts not present in the base state machine's outputs.
        Fact Insertion Rate = (Injected Facts) / (Total Summary Sentences/Facts).
        Fact Fidelity Score = 1.0 - Fact Insertion Rate.
        """
        if not aggregator_summary.strip():
            return {
                "insertion_rate": 0.0,
                "fact_fidelity_score": 1.0,
                "injected_facts": [],
                "grounded_facts": [],
                "feedback": "Aggregator summary is empty.",
            }

        # Combine all base outputs into normalized context
        base_context = " ".join(base_state_outputs).lower()
        base_words = set(re.findall(r"\b\w+\b", base_context)) - STOPWORDS
        base_roots = {w[:5] if len(w) >= 5 else w for w in base_words}

        # Split summary into distinct claim/fact sentences
        summary_claims = [
            s.strip()
            for s in re.split(r"[.!?]\s+", aggregator_summary)
            if len(s.strip()) > 10
        ]

        if not summary_claims:
            return {
                "insertion_rate": 0.0,
                "fact_fidelity_score": 1.0,
                "injected_facts": [],
                "grounded_facts": [],
                "feedback": "No actionable claims found in aggregator summary.",
            }

        injected_facts = []
        grounded_facts = []

        for claim in summary_claims:
            claim_words = [w for w in re.findall(r"\b\w+\b", claim.lower()) if w not in STOPWORDS]

            if not claim_words:
                grounded_facts.append(claim)
                continue

            # Count how many content words or roots in claim are present in base_words/base_roots
            matched_words = [
                w for w in claim_words
                if w in base_words or (w[:5] if len(w) >= 5 else w) in base_roots
            ]
            match_ratio = len(matched_words) / len(claim_words)

            # Check for 2-gram overlap as well
            if len(claim_words) >= 2:
                bigrams = [" ".join(claim_words[i : i + 2]) for i in range(len(claim_words) - 1)]
                bigram_matches = sum(1 for bg in bigrams if bg in base_context)
                bigram_ratio = bigram_matches / len(bigrams)
            else:
                bigram_ratio = match_ratio

            is_grounded = match_ratio >= 0.40 or bigram_ratio >= 0.20

            if is_grounded:
                grounded_facts.append(claim)
            else:
                injected_facts.append(claim)

        insertion_rate = round(len(injected_facts) / len(summary_claims), 4)
        fact_fidelity_score = round(1.0 - insertion_rate, 4)

        return {
            "insertion_rate": insertion_rate,
            "fact_fidelity_score": fact_fidelity_score,
            "injected_facts": injected_facts,
            "grounded_facts": grounded_facts,
            "total_claims_evaluated": len(summary_claims),
            "feedback": (
                f"Fact Insertion Rate: {insertion_rate} ({len(injected_facts)} injected out of "
                f"{len(summary_claims)} claims)."
            ),
        }

    def eval_conflict_resolution(
        self, aggregator_summary: str, conflicting_inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates whether the aggregator correctly flagged, reconciled, or rejected
        conflicting information fed from base machine test cases.
        """
        if not conflicting_inputs:
            return {
                "conflict_resolution_score": 1.0,
                "conflicts_handled": 0,
                "conflicts_ignored": 0,
                "feedback": "No conflicting inputs provided to test.",
            }

        summary_lower = aggregator_summary.lower()

        conflict_indicators = [
            "conflict",
            "contradict",
            "discrepancy",
            "diverge",
            "reconcil",
            "however",
            "whereas",
            "on the other hand",
            "in contrast",
            "inconsistent",
            "disputed",
            "rejected",
            "flagged",
        ]

        conflicts_handled = 0
        conflicts_ignored = 0

        for item in conflicting_inputs:
            topic = str(item.get("topic", "")).lower()
            fact_a = str(item.get("fact_a", "")).lower()
            fact_b = str(item.get("fact_b", "")).lower()

            # Check if topic or facts appear in summary alongside conflict indicator
            has_topic_ref = (topic in summary_lower) or (fact_a in summary_lower or fact_b in summary_lower)
            has_conflict_flag = any(ind in summary_lower for ind in conflict_indicators)

            if has_topic_ref and has_conflict_flag:
                conflicts_handled += 1
            else:
                conflicts_ignored += 1

        total = len(conflicting_inputs)
        conflict_resolution_score = round(conflicts_handled / total, 4)

        return {
            "conflict_resolution_score": conflict_resolution_score,
            "conflicts_handled": conflicts_handled,
            "conflicts_ignored": conflicts_ignored,
            "feedback": (
                f"Conflict Resolution Score: {conflict_resolution_score} "
                f"({conflicts_handled}/{total} conflicts addressed)."
            ),
        }


class ResearchReasoningRubrics:
    """
    Decoupled LLM Judge Rubrics for Research & Reasoning.
    Evaluates Premise Monotonicity, Evidence Grounding, and Gap Identification.
    """

    def eval_premise_monotonicity(
        self, reasoning_steps: List[str], final_conclusion: str
    ) -> Dict[str, Any]:
        """
        Evaluates if the reasoning holds true monotonically from step 1 to the final conclusion,
        or if it contradicts itself mid-trace or reverses premises without justification.
        """
        if not reasoning_steps or not final_conclusion:
            return {
                "monotonicity_score": 0.0,
                "contradictions_found": ["Missing reasoning steps or conclusion"],
                "feedback": "Insufficient trace input for monotonicity evaluation.",
            }

        contradiction_pairs = [
            ("increase", "decrease"),
            ("bullish", "bearish"),
            ("positive", "negative"),
            ("solvent", "insolvent"),
            ("overvalued", "undervalued"),
            ("approve", "reject"),
            ("low risk", "high risk"),
        ]

        trace_tokens = [s.lower() for s in reasoning_steps]
        conclusion_lower = final_conclusion.lower()

        contradictions_found = []

        # Compare step-by-step sentiment/stance shifts
        for idx in range(len(trace_tokens) - 1):
            step1 = trace_tokens[idx]
            step2 = trace_tokens[idx + 1]

            for term_a, term_b in contradiction_pairs:
                if (term_a in step1 and term_b in step2) or (term_b in step1 and term_a in step2):
                    contradictions_found.append(
                        f"Step {idx + 1} ('{term_a}'/'{term_b}') contradicts Step {idx + 2}."
                    )

        # Check final step vs conclusion contradiction
        last_step = trace_tokens[-1]
        for term_a, term_b in contradiction_pairs:
            if (term_a in last_step and term_b in conclusion_lower) or (
                term_b in last_step and term_a in conclusion_lower
            ):
                contradictions_found.append(
                    f"Final reasoning step contradicts conclusion: '{term_a}' vs '{term_b}'."
                )

        penalty = 0.3 * len(contradictions_found)
        monotonicity_score = max(0.0, round(1.0 - penalty, 4))

        return {
            "monotonicity_score": monotonicity_score,
            "contradictions_found": contradictions_found,
            "total_reasoning_steps": len(reasoning_steps),
            "feedback": (
                f"Premise Monotonicity Score: {monotonicity_score}. "
                f"Contradictions found: {len(contradictions_found)}."
            ),
        }

    def eval_evidence_grounding(self, research_output: str) -> Dict[str, Any]:
        """
        Counts the ratio of explicit source citations to generalized claims in the final research output.
        """
        if not research_output.strip():
            return {
                "grounding_ratio": 0.0,
                "evidence_score": 0.0,
                "citation_count": 0,
                "total_claims": 0,
                "feedback": "Research output is empty.",
            }

        sentences = [
            s.strip()
            for s in re.split(r"[.!?]\s+", research_output)
            if len(s.strip()) > 8
        ]

        if not sentences:
            return {
                "grounding_ratio": 0.0,
                "evidence_score": 0.0,
                "citation_count": 0,
                "total_claims": 0,
                "feedback": "No sentences found to evaluate.",
            }

        citation_patterns = [
            r"\[ Source:? [^\]]+ \]",
            r"\[ Ref:? [^\]]+ \]",
            r"\[\d+\]",
            r"https?://\S+",
            r"according to [^,.!?]+",
            r"as reported in [^,.!?]+",
            r"sec filing [^,.!?]+",
            r"10-k|10-q|8-k",
            r"exhibit \d+",
        ]

        cited_sentences = 0
        total_citations = 0

        for sentence in sentences:
            sentence_has_citation = False
            for pattern in citation_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                if matches:
                    total_citations += len(matches)
                    sentence_has_citation = True
            if sentence_has_citation:
                cited_sentences += 1

        grounding_ratio = round(cited_sentences / len(sentences), 4)

        # Evidence score maps grounding ratio to 0.0-1.0 scale with baseline expectations (>0.50 ratio = 1.0)
        evidence_score = min(1.0, round(grounding_ratio * 1.5, 4))

        return {
            "grounding_ratio": grounding_ratio,
            "evidence_score": evidence_score,
            "citation_count": total_citations,
            "cited_sentences": cited_sentences,
            "total_claims": len(sentences),
            "feedback": (
                f"Evidence Grounding Score: {evidence_score} (Grounding ratio: {grounding_ratio}, "
                f"Citations found: {total_citations})."
            ),
        }

    def eval_gap_identification(self, research_output: str) -> Dict[str, Any]:
        """
        Evaluates whether the system explicitly stated what it could not find or what remains uncertain.
        """
        if not research_output.strip():
            return {
                "gap_identification_score": 0.0,
                "gaps_found": [],
                "feedback": "Research output is empty.",
            }

        gap_phrase_patterns = [
            r"unable to (?:verify|confirm|locate|find|determine)",
            r"data (?:unavailable|missing|incomplete|lacking)",
            r"remains (?:uncertain|unclear|unverified|ambiguous)",
            r"knowledge gap[s]?",
            r"limitation[s]?:",
            r"further research is required",
            r"not explicitly stated in",
            r"information was not provided",
        ]

        gaps_found = []
        for pattern in gap_phrase_patterns:
            matches = re.findall(pattern, research_output, re.IGNORECASE)
            if matches:
                gaps_found.extend(matches)

        has_explicit_gap_section = "gap" in research_output.lower() or "limitation" in research_output.lower() or "uncertain" in research_output.lower()

        if len(gaps_found) >= 2 or (len(gaps_found) >= 1 and has_explicit_gap_section):
            score = 1.0
        elif len(gaps_found) == 1:
            score = 0.7
        elif has_explicit_gap_section:
            score = 0.5
        else:
            score = 0.0

        return {
            "gap_identification_score": score,
            "gaps_found": gaps_found,
            "feedback": f"Gap Identification Score: {score}. Gaps detected: {len(gaps_found)}.",
        }


class ConfidenceCalibrationEvaluator:
    """
    Evaluates Expected Calibration Error (ECE) and penalizes Adversarial Traps
    (high confidence outputs with low reasoning accuracy).
    """

    def calculate_ece(
        self, accuracy_scores: List[float], confidence_scores: List[float], num_bins: int = 5
    ) -> float:
        """
        Calculates Expected Calibration Error (ECE).
        ECE = sum_b (|acc(b) - conf(b)| * |b| / N)
        """
        if not accuracy_scores or len(accuracy_scores) != len(confidence_scores):
            return 0.0

        n = len(accuracy_scores)
        bin_boundaries = [i / num_bins for i in range(num_bins + 1)]

        ece = 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Indices belonging to bin
            bin_indices = [
                idx
                for idx, conf in enumerate(confidence_scores)
                if (conf >= bin_lower and conf < bin_upper) or (i == num_bins - 1 and conf == bin_upper)
            ]

            bin_size = len(bin_indices)
            if bin_size > 0:
                avg_acc = sum(accuracy_scores[idx] for idx in bin_indices) / bin_size
                avg_conf = sum(confidence_scores[idx] for idx in bin_indices) / bin_size
                ece += (bin_size / n) * abs(avg_acc - avg_conf)

        return round(ece, 4)

    def eval_confidence_calibration(
        self,
        accuracy_scores: List[float],
        confidence_scores: List[float],
        is_adversarial: bool = False,
    ) -> Dict[str, Any]:
        """
        Gather judge's accuracy scores across test suite, compare against system generated
        confidence scores to calculate ECE and heavily penalize hallucinated certainty.
        """
        if not accuracy_scores or len(accuracy_scores) != len(confidence_scores):
            return {
                "calibration_score": 0.0,
                "ece": 1.0,
                "adversarial_trap_count": 0,
                "feedback": "Empty or mismatched accuracy and confidence arrays provided.",
            }

        ece = self.calculate_ece(accuracy_scores, confidence_scores)

        # Detect Adversarial Traps: High confidence (>=0.80) when accuracy is low (<=0.40)
        adversarial_traps = 0
        for acc, conf in zip(accuracy_scores, confidence_scores):
            if conf >= 0.80 and acc <= 0.40:
                adversarial_traps += 1

        # Calibration score calculation
        # Base calibration = 1.0 - ECE
        # Heavy penalty for each adversarial trap (0.25 per trap)
        trap_penalty = 0.25 * adversarial_traps
        calibration_score = max(0.0, round(1.0 - ece - trap_penalty, 4))

        return {
            "calibration_score": calibration_score,
            "ece": ece,
            "adversarial_trap_count": adversarial_traps,
            "is_adversarial_suite": is_adversarial,
            "sample_count": len(accuracy_scores),
            "feedback": (
                f"Calibration Score: {calibration_score} (ECE: {ece}, "
                f"Adversarial Traps: {adversarial_traps})."
            ),
        }


class MultiStateReasoningEvaluator:
    """
    Unified Master Evaluator combining all 4 evaluation dimensions:
    1. Base State Machine (Task Extraction)
    2. Aggregator State Machine (Synthesis)
    3. Research & Reasoning Rubrics
    4. Confidence Calibration & ECE
    """

    def __init__(self):
        self.base_evaluator = BaseStateMachineEvaluator()
        self.aggregator_evaluator = AggregatorStateMachineEvaluator()
        self.reasoning_rubrics = ResearchReasoningRubrics()
        self.calibration_evaluator = ConfidenceCalibrationEvaluator()

    def evaluate_multi_state_system(
        self,
        base_state_logs: List[Dict[str, Any]],
        adversarial_state_logs: Optional[List[Dict[str, Any]]],
        aggregator_summary: str,
        base_state_outputs: List[str],
        conflicting_inputs: Optional[List[Dict[str, Any]]],
        reasoning_steps: List[str],
        final_conclusion: str,
        accuracy_scores: List[float],
        confidence_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Executes complete end-to-end evaluation across all four state machine and calibration axes.
        """
        # 1. Base Machine Evaluation
        base_fidelity = self.base_evaluator.eval_state_trace_fidelity(base_state_logs)
        adv_resiliency = (
            self.base_evaluator.eval_adversarial_resiliency(base_state_logs, adversarial_state_logs)
            if adversarial_state_logs
            else {"resiliency_score": 1.0, "feedback": "No adversarial trace provided."}
        )

        # 2. Aggregator Evaluation
        fact_insertion = self.aggregator_evaluator.eval_fact_insertion_rate(aggregator_summary, base_state_outputs)
        conflict_res = self.aggregator_evaluator.eval_conflict_resolution(
            aggregator_summary, conflicting_inputs or []
        )

        # 3. Reasoning Rubrics
        monotonicity = self.reasoning_rubrics.eval_premise_monotonicity(reasoning_steps, final_conclusion)
        grounding = self.reasoning_rubrics.eval_evidence_grounding(aggregator_summary)
        gap_id = self.reasoning_rubrics.eval_gap_identification(aggregator_summary)

        # 4. Calibration Evaluation
        calibration = self.calibration_evaluator.eval_confidence_calibration(
            accuracy_scores, confidence_scores, is_adversarial=bool(adversarial_state_logs)
        )

        # Aggregate Overall Composite Score
        composite_score = round(
            0.20 * base_fidelity["fidelity_score"]
            + 0.15 * adv_resiliency.get("resiliency_score", 1.0)
            + 0.20 * fact_insertion["fact_fidelity_score"]
            + 0.15 * conflict_res["conflict_resolution_score"]
            + 0.10 * monotonicity["monotonicity_score"]
            + 0.10 * grounding["evidence_score"]
            + 0.10 * calibration["calibration_score"],
            4,
        )

        return {
            "composite_score": composite_score,
            "axis_1_base_state_machine": {
                "fidelity": base_fidelity,
                "adversarial_resiliency": adv_resiliency,
            },
            "axis_2_aggregator_state_machine": {
                "fact_insertion": fact_insertion,
                "conflict_resolution": conflict_res,
            },
            "axis_3_research_reasoning": {
                "premise_monotonicity": monotonicity,
                "evidence_grounding": grounding,
                "gap_identification": gap_id,
            },
            "axis_4_confidence_calibration": calibration,
        }
