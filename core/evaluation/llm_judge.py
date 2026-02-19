import re
from typing import Dict, Any

class ConvictionScorer:
    """
    Simulated LLM-as-a-Judge that assigns reliability scores (0.0-1.0) to agent outputs based on heuristics.
    """
    def score(self, text: str) -> Dict[str, Any]:
        """
        Assigns a conviction score to the given text based on heuristics.
        Returns a dictionary with 'score', 'rationale', and 'judge_model'.
        """
        if not text:
            return {
                "score": 0.0,
                "rationale": "Empty input text.",
                "judge_model": "HeuristicConvictionScorer-v1"
            }

        # Evidence patterns: digits, %, $, specific years
        evidence_boost = 0.0
        if re.search(r'\d+%', text): evidence_boost += 0.2
        if re.search(r'\$\d+', text): evidence_boost += 0.2
        if re.search(r'\b(19|20)\d{2}\b', text): evidence_boost += 0.1  # Years like 1998, 2026

        # Hedging patterns
        hedging_penalty = 0.0
        if re.search(r'\b(maybe|perhaps|possibly|might|could)\b', text, re.IGNORECASE): hedging_penalty += 0.2
        if re.search(r'\b(rumor|speculation|suggests)\b', text, re.IGNORECASE): hedging_penalty += 0.1

        # Base score is 0.5, adjusted by evidence and hedging
        raw_score = 0.5 + evidence_boost - hedging_penalty
        final_score = max(0.0, min(1.0, raw_score))

        return {
            "score": round(final_score, 2),
            "rationale": "Based on heuristic analysis of quantitative data presence vs hedging language.",
            "judge_model": "HeuristicConvictionScorer-v1"
        }
