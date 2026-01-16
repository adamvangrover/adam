from typing import Dict, List, Any
import logging

class ImpactAnalysisAgent:
    """
    Analyzes cross-sector correlations and systemic risks.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_impact(self, sector_outlooks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a cross-impact matrix based on sector outlooks.
        """
        self.logger.info("Analyzing cross-sector impacts...")

        sectors = list(sector_outlooks.keys())
        matrix = {s: {} for s in sectors}

        # Define Base Correlations (Static Knowledge Base)
        # 1.0 = Perfect Positive, -1.0 = Perfect Negative
        base_correlations = {
            ("Technology", "Energy"): 0.8,   # AI Energy Demand
            ("Technology", "Financials"): -0.4, # Rate sensitivity vs Value
            ("Energy", "Financials"): 0.3,   # Inflation trade alignment
            ("Energy", "Technology"): 0.8,
            ("Financials", "Technology"): -0.4,
            ("Financials", "Energy"): 0.3
        }

        # Dynamic Adjustment based on Outlooks
        # If both are Bullish, correlation increases (Sync)
        # If divergent, correlation decreases

        for s1 in sectors:
            for s2 in sectors:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                    continue

                # Get Base
                corr = base_correlations.get((s1, s2), 0.1) # Default weak correlation

                # Adjust based on 'Overweight' (Bullish) vs 'Underweight' (Bearish)
                r1 = sector_outlooks[s1].get("rating", "Neutral")
                r2 = sector_outlooks[s2].get("rating", "Neutral")

                score_map = {"OVERWEIGHT": 1, "NEUTRAL": 0, "UNDERWEIGHT": -1}
                v1 = score_map.get(r1, 0)
                v2 = score_map.get(r2, 0)

                # Narrative Generation
                narrative = "Standard correlation."
                if s1 == "Technology" and s2 == "Energy":
                    narrative = "High correlation driven by AI Datacenter power constraints."
                elif s1 == "Technology" and s2 == "Financials":
                    narrative = "Inverse relationship due to rate sensitivity vs cyclical value."

                matrix[s1][s2] = {
                    "value": corr,
                    "narrative": narrative
                }

        return matrix
