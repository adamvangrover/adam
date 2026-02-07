import math
from typing import Dict, Any, List, Tuple

class BayesianReasoningEngine:
    """
    A probabilistic engine that estimates the likelihood of correctness
    for AI-generated financial analysis without relying on a live LLM.

    It uses a naive Bayesian approach to update 'Confidence' based on
    observed 'Signals' (Consistency, Ontology Alignment, Numerical Accuracy).
    """

    def __init__(self):
        # Priors
        self.base_hallucination_rate = 0.15 # 15% chance an LLM hallucinates complex debt
        self.base_logic_error_rate = 0.20   # 20% chance of math errors

    def calculate_confidence(self, state: Dict[str, Any], verification_flags: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Derives a final Confidence Score and Reasoning Trace.
        """
        # 1. Feature Extraction
        quant_text = state.get("quant_analysis", "")
        financials = state.get("balance_sheet", {})
        ebitda = state.get("income_statement", {}).get("consolidated_ebitda", 0)
        debt = financials.get("total_debt", 0)

        # 2. Signals
        signals = []

        # Signal A: Numerical Consistency (Leverage)
        calc_leverage = debt / ebitda if ebitda > 0 else 0
        stated_leverage = self._extract_leverage(quant_text)

        leverage_delta = 0
        if stated_leverage is not None:
            leverage_delta = abs(calc_leverage - stated_leverage)
            if leverage_delta > 0.5:
                # Strong signal of error
                signals.append({
                    "type": "NUMERICAL_MISMATCH",
                    "weight": 0.40,
                    "impact": -1.0, # Negative impact
                    "desc": f"Calculated leverage ({calc_leverage:.2f}x) differs significantly from stated ({stated_leverage:.2f}x)."
                })
            elif leverage_delta < 0.1:
                signals.append({
                    "type": "NUMERICAL_MATCH",
                    "weight": 0.20,
                    "impact": 1.0,
                    "desc": "Leverage calculation is accurate."
                })
        else:
             signals.append({
                "type": "MISSING_DATA",
                "weight": 0.10,
                "impact": -0.5,
                "desc": "No leverage figure extracted from text."
            })

        # Signal B: Ontology Alignment (from SymbolicVerifier)
        if verification_flags:
            count = len(verification_flags)
            severity = sum(1 for f in verification_flags if f.get("severity") == "HIGH")
            signals.append({
                "type": "ONTOLOGY_VIOLATION",
                "weight": 0.50, # High importance
                "impact": -1.0 * (0.5 + (0.5 * min(count, 3)/3)), # Scales with count
                "desc": f"Found {count} semantic contradictions (Seniority/Structure)."
            })
        else:
            signals.append({
                "type": "ONTOLOGY_ALIGNMENT",
                "weight": 0.15,
                "impact": 0.5,
                "desc": "Entities align with Knowledge Graph."
            })

        # Signal C: Sentiment/Financials Consistency
        # If High Leverage (>6x) but text says "Safe/Strong" -> Optimism Bias
        if calc_leverage > 6.0:
            positive_words = ["strong", "safe", "resilient", "healthy", "low risk"]
            if any(w in quant_text.lower() for w in positive_words):
                signals.append({
                    "type": "OPTIMISM_BIAS",
                    "weight": 0.30,
                    "impact": -0.8,
                    "desc": f"High leverage ({calc_leverage:.2f}x) contradicts positive sentiment keywords."
                })

        # 3. Bayesian Update
        # Start with a neutral confidence (e.g., 0.8 for a good model)
        confidence = 0.85

        trace = []
        for signal in signals:
            # Simple linear adjustment for this "Static" demo version
            # In a full version, this would be Log-Odds updates
            adjustment = signal["weight"] * signal["impact"]
            confidence += adjustment
            trace.append(f"{signal['desc']} (Impact: {adjustment:+.2f})")

        confidence = max(0.01, min(0.99, confidence))

        return {
            "score": confidence,
            "signals": signals,
            "trace": trace
        }

    def _extract_leverage(self, text: str) -> float | None:
        import re
        match = re.search(r"Leverage.*?:\s*(\d+\.\d+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None
