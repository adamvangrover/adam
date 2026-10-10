import math
import numpy as np
from typing import List, Dict, Any, Tuple

class MultiStateReasoningEvaluator:
    """
    Evaluator for multi-state reasoning, swarm trace fidelity, fact insertion,
    and calibration error across agent reasoning pipelines.
    """

    @staticmethod
    def evaluate_trace_fidelity(predicted_trace: List[str], ground_truth_trace: List[str]) -> float:
        """Calculates normalized Levenshtein-based sequence similarity between trace steps."""
        if not predicted_trace and not ground_truth_trace:
            return 1.0
        if not predicted_trace or not ground_truth_trace:
            return 0.0

        str_pred = " -> ".join(predicted_trace)
        str_gt = " -> ".join(ground_truth_trace)

        m, n = len(str_pred), len(str_gt)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str_pred[i - 1] == str_gt[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        max_len = max(m, n)
        return max(0.0, round(1.0 - (dp[m][n] / max_len), 4))

    @staticmethod
    def evaluate_fact_insertion(inserted_facts: List[Dict[str, Any]], expected_facts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculates Precision, Recall, and F1 for fact insertion and conflict resolution."""
        if not expected_facts:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not inserted_facts:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        expected_keys = {f.get("fact_id", f.get("key")) for f in expected_facts if f.get("fact_id") or f.get("key")}
        inserted_keys = {f.get("fact_id", f.get("key")) for f in inserted_facts if f.get("fact_id") or f.get("key")}

        tp = len(expected_keys.intersection(inserted_keys))
        fp = len(inserted_keys - expected_keys)
        fn = len(expected_keys - inserted_keys)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    @staticmethod
    def evaluate_reasoning_rubric(reasoning_payload: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates research reasoning quality across Monotonicity, Evidence Grounding,
        and Gap Identification.
        """
        evidence = reasoning_payload.get("evidence", [])
        gaps = reasoning_payload.get("gaps_identified", [])
        monotonic_steps = reasoning_payload.get("monotonic_confidence_steps", [])

        grounding_score = min(1.0, len(evidence) / 5.0) if isinstance(evidence, list) else 0.0
        gap_score = min(1.0, len(gaps) / 3.0) if isinstance(gaps, list) else 0.0

        if monotonic_steps and len(monotonic_steps) > 1:
            diffs = [monotonic_steps[i] - monotonic_steps[i-1] for i in range(1, len(monotonic_steps))]
            monotonicity_score = 1.0 if all(d >= 0 for d in diffs) else max(0.0, 1.0 - (sum(1 for d in diffs if d < 0) / len(diffs)))
        else:
            monotonicity_score = 1.0

        composite_score = round(0.4 * grounding_score + 0.3 * gap_score + 0.3 * monotonicity_score, 4)

        return {
            "evidence_grounding": round(grounding_score, 4),
            "gap_identification": round(gap_score, 4),
            "monotonicity": round(monotonicity_score, 4),
            "composite_rubric_score": composite_score
        }

    @staticmethod
    def calculate_expected_calibration_error(confidences: List[float], accuracies: List[int], num_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE) across confidence bins.
        """
        if not confidences or len(confidences) != len(accuracies):
            return 0.0

        conf_arr = np.array(confidences)
        acc_arr = np.array(accuracies)
        bin_boundaries = np.linspace(0, 1, num_bins + 1)

        ece = 0.0
        total_samples = len(confidences)

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (conf_arr > bin_lower) & (conf_arr <= bin_upper) if i > 0 else (conf_arr >= bin_lower) & (conf_arr <= bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                avg_confidence = np.mean(conf_arr[in_bin])
                avg_accuracy = np.mean(acc_arr[in_bin])
                ece += (bin_size / total_samples) * abs(avg_confidence - avg_accuracy)

        return round(float(ece), 4)

    def evaluate_multi_state_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Complete evaluation harness combining trace fidelity, fact insertion, rubrics, and ECE."""
        trace_score = self.evaluate_trace_fidelity(
            payload.get("predicted_trace", []),
            payload.get("ground_truth_trace", [])
        )

        fact_metrics = self.evaluate_fact_insertion(
            payload.get("inserted_facts", []),
            payload.get("expected_facts", [])
        )

        rubric_metrics = self.evaluate_reasoning_rubric(payload.get("reasoning_payload", {}))

        ece_score = self.calculate_expected_calibration_error(
            payload.get("confidences", []),
            payload.get("accuracies", [])
        )

        overall_quality = round(
            0.3 * trace_score +
            0.3 * fact_metrics["f1"] +
            0.3 * rubric_metrics["composite_rubric_score"] +
            0.1 * (1.0 - min(1.0, ece_score)),
            4
        )

        return {
            "trace_fidelity": trace_score,
            "fact_insertion": fact_metrics,
            "reasoning_rubrics": rubric_metrics,
            "expected_calibration_error": ece_score,
            "overall_quality_score": overall_quality
        }
