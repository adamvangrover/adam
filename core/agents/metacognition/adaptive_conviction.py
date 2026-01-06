# core/agents/metacognition/adaptive_conviction.py
from typing import Optional, Dict, List, Any, Tuple
import logging
import math

class ConvictionRouter:
    """
    Decides between 'DIRECT' (Natural Language) and 'MCP' (Structured Schema)
    interaction modes based on the Conviction-to-Complexity Ratio (CCR).
    """

    # Default Thresholds
    DEFAULT_CONVICTION_THRESHOLD_MCP = 0.85
    DEFAULT_COMPLEXITY_THRESHOLD_DIRECT = 0.4

    @staticmethod
    def estimate_complexity(task_description: str) -> float:
        """
        Estimates the 'Intrinsic Load' of a task.
        Simple heuristic: length, number of variables, keywords.
        0.0 = "What is the date?"
        1.0 = "Calculate the Sharpe Ratio of this 50-asset portfolio and rebalance."
        """
        complexity = 0.0

        # Length factor
        length = len(task_description.split())
        complexity += min(length / 100.0, 0.5)

        # Keywords
        complex_terms = ["calculate", "analyze", "optimize", "simulate", "integrate", "schema", "json", "protocol"]
        for term in complex_terms:
            if term in task_description.lower():
                complexity += 0.1

        # Structural indicators
        if "{" in task_description or "[" in task_description:
            complexity += 0.2

        return min(complexity, 1.0)

    @staticmethod
    def decide_mode(conviction: float, complexity: float,
                   conviction_threshold: float = DEFAULT_CONVICTION_THRESHOLD_MCP) -> str:
        """
        Returns 'DIRECT' or 'MCP'.
        """
        if complexity < 0.2:
            return "DIRECT" # Simplicity Bias aligned

        if conviction < conviction_threshold:
            # Conviction Gap detected.
            return "DIRECT" # Trigger Elicitation / Clarification

        return "MCP"

class AdaptiveConvictionMixin:
    """
    Mixin to add Adaptive Conviction capabilities to an Agent.
    """

    def __init__(self):
        self._conviction_history: List[float] = []
        self._conviction_threshold = ConvictionRouter.DEFAULT_CONVICTION_THRESHOLD_MCP

    def configure_conviction(self, threshold: float):
        self._conviction_threshold = threshold

    def measure_conviction(self,
                          text_output: str,
                          logit_probs: Optional[List[float]] = None,
                          semantic_entropy: Optional[float] = None) -> float:
        """
        Calculates a normalized conviction score (0.0 - 1.0).
        Prioritizes Logits > Semantic Entropy > Verbalized (simulated).
        """
        score = 0.0

        # 1. Logit-Based (The Gold Standard)
        if logit_probs:
            # Product Method (avg log prob)
            avg_log_prob = sum(logit_probs) / len(logit_probs)
            # Convert to probability space for score
            score = math.exp(avg_log_prob)

        # 2. Semantic Entropy (The Silver Standard)
        elif semantic_entropy is not None:
            # Low entropy = High conviction
            # Normalize entropy (0 to ~3ish) to 0-1 confidence
            score = max(0.0, 1.0 - (semantic_entropy / 2.0))

        # 3. Heuristic / Verbalized Proxy (The Bronze Standard)
        else:
            # Fallback: Analyze the text for uncertainty markers
            uncertainty_markers = ["maybe", "perhaps", "could", "might", "unsure", "estimate", "assuming"]
            count = sum(1 for marker in uncertainty_markers if marker in text_output.lower())

            # Base confidence
            score = 0.95

            # Penalize for uncertainty words
            score -= (count * 0.1)

            # Penalize for extreme brevity or verbosity (U-curve)
            length = len(text_output.split())
            if length < 5: score -= 0.1
            if length > 500: score -= 0.1

        score = max(0.0, min(1.0, score))
        self._conviction_history.append(score)
        return score

    def get_interaction_strategy(self, task_description: str) -> Tuple[str, Dict[str, Any]]:
        """
        Determines the optimal interaction strategy.
        Returns (Mode, Metadata).
        """
        complexity = ConvictionRouter.estimate_complexity(task_description)
        current_conviction = self._conviction_history[-1] if self._conviction_history else 0.9

        # Adjust conviction based on complexity (Simulating the "Conviction Gap")
        adjusted_conviction = current_conviction - (complexity * 0.2)

        mode = ConvictionRouter.decide_mode(adjusted_conviction, complexity, self._conviction_threshold)

        return mode, {
            "complexity": complexity,
            "conviction": adjusted_conviction,
            "raw_conviction": current_conviction
        }

    def generate_metacognitive_response(self, content: Any, conviction: float) -> Dict[str, Any]:
        """
        Wraps response in Metacognitive Metadata for A2A auditability.
        """
        return {
            "content": content,
            "meta": {
                "conviction": conviction,
                "drift_warning": False, # Placeholder
                "router_verdict": "SAFE" if conviction > 0.8 else "CAUTION"
            }
        }
