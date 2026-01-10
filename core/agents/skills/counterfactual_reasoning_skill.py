from typing import Dict, Any, List
from datetime import datetime

class CounterfactualReasoningSkill:
    """
    Implements the 'Counterfactual Reasoning' skill for the Red Team Agent.
    Inverts assumptions to generate 'Bear Case' scenarios.
    """
    def __init__(self, llm_client=None):
        self.llm = llm_client # In a real agent, this would be the LLM interface

    def invert_assumptions(self, credit_memo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses a credit memo and inverts key assumptions.
        """
        original_assumptions = credit_memo.get("assumptions", {})
        inverted = {}

        # Heuristic inversion logic (can be enhanced with LLM)
        for key, value in original_assumptions.items():
            if isinstance(value, (int, float)):
                # Simple heuristic: flip growth to decline, stability to volatility
                if "growth" in key.lower():
                     inverted[key] = value * -1.0
                elif "inflation" in key.lower():
                     inverted[key] = value * 1.5 # Sticky inflation
                elif "rate" in key.lower():
                     inverted[key] = value + 0.02 # Rate hike shock
                else:
                     inverted[key] = value * 0.8 # Generic stress
            else:
                inverted[key] = f"Inverse of {value}"

        return inverted

    def generate_bear_case(self, credit_memo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a Bear Case narrative and simulates impact.
        """
        inverted_assumptions = self.invert_assumptions(credit_memo)

        # Identify Failure Catalyst
        catalyst = "Liquidity Crunch" # Default placeholder
        if inverted_assumptions.get("revenue_growth", 0) < 0:
            catalyst = "Revenue Contraction"
        if inverted_assumptions.get("interest_rate", 0) > 0.05:
            catalyst = "Debt Service Coverage Failure"

        return {
            "scenario": "Bear Case / Stress Test",
            "inverted_assumptions": inverted_assumptions,
            "failure_catalyst": catalyst,
            "simulated_impact_score": self._simulate_impact(inverted_assumptions)
        }

    def _simulate_impact(self, assumptions: Dict[str, Any]) -> float:
        """
        Calculates a synthetic impact score (0-100).
        Higher score = Higher Risk.
        """
        score = 50.0
        # Logic to adjust score based on assumptions
        if assumptions.get("revenue_growth", 0) < 0:
            score += 20
        return min(score, 100.0)

    def review_memo(self, credit_memo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point. Reviews a memo and returns a Red Team Report.
        """
        bear_case = self.generate_bear_case(credit_memo)

        passed = bear_case["simulated_impact_score"] < 80 # Threshold

        return {
            "timestamp": datetime.now().isoformat(),
            "status": "APPROVED" if passed else "REJECTED",
            "red_team_score": bear_case["simulated_impact_score"],
            "critique": bear_case
        }
