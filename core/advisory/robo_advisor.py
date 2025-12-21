"""
Module 3: Robo-Advisor Logic & Intake Engine
============================================

Architect Notes:
----------------
1.  **Psychology vs. Math**: This engine distinguishes between 'Risk Capacity' (Financial ability to take loss)
    and 'Risk Tolerance' (Emotional ability to take loss). Capacity is a hard constraint; Tolerance is a soft preference.
    The score is weighted: Capacity (60%) > Tolerance (40%).
2.  **Scoring Algorithm**: We use a weighted sum model. Each answer maps to a normalized score (0-100).
    The final score determines the `RiskBand`.
3.  **Safety First**: If Risk Capacity is "Low" (e.g., short timeline or need for liquidity), the system
    forces a Conservative allocation regardless of the user's high Risk Tolerance. This is a "Suitability Check"
    required by fiduciary standards.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class RiskBand(Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"

@dataclass
class ClientProfile:
    time_horizon_years: int
    liquidity_needs: str  # 'High', 'Medium', 'Low'
    risk_capacity_score: int # 0-100
    risk_tolerance_score: int # 0-100

class IntakeForm:
    """
    Simulates the client questionnaire.
    In a web app, this would process JSON payload from a frontend form.
    """

    @staticmethod
    def calculate_score(answers: Dict[str, any]) -> ClientProfile:
        # --- Section 1: Time Horizon & Liquidity (Risk Capacity) ---
        # Q1: When do you need this money?
        years = answers.get('time_horizon', 5)

        # Q2: How much of this investment might you need to withdraw in < 2 years?
        liquidity = answers.get('liquidity_needs', 'Low') # High, Medium, Low

        # Logic:
        # > 15 years = 100 capacity
        # < 3 years = 0 capacity
        capacity_score = min(100, max(0, (years - 3) * 8))
        if liquidity == 'High':
            capacity_score *= 0.5  # Penalize heavy liquidity needs

        # --- Section 2: Emotional Risk (Risk Tolerance) ---
        # Q3: The market drops 20% in a month. You:
        # A: Sell everything (0)
        # B: Sell some (30)
        # C: Do nothing (60)
        # D: Buy more (100)
        reaction_score = answers.get('market_drop_reaction', 50)

        return ClientProfile(
            time_horizon_years=years,
            liquidity_needs=liquidity,
            risk_capacity_score=int(capacity_score),
            risk_tolerance_score=int(reaction_score)
        )

class RoboAdvisor:
    def __init__(self, base_portfolio_path: str):
        self.base_portfolio_path = base_portfolio_path
        # Weights for the final score
        self.w_capacity = 0.60
        self.w_tolerance = 0.40

    def analyze_market_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes market data (Intra-Year and Long-Term) to adjust risk profile.
        This framework method allows advanced functionality to be plugged in.

        Args:
            market_data: A dictionary containing 'long_term' and 'intra_year' data keys,
                         following the Gold Standard format.
        """
        # Framework placeholder for advanced market analysis
        # Could implement moving averages, volatility calculations, etc.
        return {
            "status": "Analysis Not Implemented",
            "modifier": 0.0
        }

    def generate_recommendation(self, answers: Dict[str, any], market_context: Optional[Dict] = None) -> Dict:
        """
        Main entry point for advisory logic.
        Accepts optional market_context to refine recommendation.
        """
        profile = IntakeForm.calculate_score(answers)

        # Weighted Score
        final_score = (profile.risk_capacity_score * self.w_capacity) + \
                      (profile.risk_tolerance_score * self.w_tolerance)

        # Suitability Override: If Capacity is critically low, cap the risk
        if profile.risk_capacity_score < 30:
            final_score = min(final_score, 30)
            warning = "Risk Tolerance overridden by low Risk Capacity (Time Horizon/Liquidity)."
        else:
            warning = None

        # Mapping Logic
        if final_score < 40:
            band = RiskBand.CONSERVATIVE
            modifier = {"equities": 0.20, "fixed_income": 0.60, "alternatives": 0.20}
        elif final_score < 75:
            band = RiskBand.MODERATE
            modifier = {"equities": 0.40, "fixed_income": 0.40, "alternatives": 0.20} # The Base "Gold Standard"
        else:
            band = RiskBand.AGGRESSIVE
            modifier = {"equities": 0.70, "fixed_income": 0.20, "alternatives": 0.10}

        return {
            "client_profile": {
                "capacity": profile.risk_capacity_score,
                "tolerance": profile.risk_tolerance_score,
                "final_score": final_score
            },
            "recommendation": {
                "band": band.value,
                "target_weights": modifier,
                "warning": warning
            }
        }

# --- Example Usage Logic ---
if __name__ == "__main__":
    advisor = RoboAdvisor("data/strategies/gold_standard_portfolio.json")

    # Test Case: Young Professional
    client_a = {
        'time_horizon': 25,
        'liquidity_needs': 'Low',
        'market_drop_reaction': 100
    }

    # Test Case: Retiree
    client_b = {
        'time_horizon': 3,
        'liquidity_needs': 'High',
        'market_drop_reaction': 30
    }

    print("--- Client A (Aggressive) ---")
    print(json.dumps(advisor.generate_recommendation(client_a), indent=2))

    print("\n--- Client B (Conservative) ---")
    print(json.dumps(advisor.generate_recommendation(client_b), indent=2))
