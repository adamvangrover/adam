"""
Module 3: Robo-Advisor Logic & Intake Engine
============================================

Architectural Blueprint:
------------------------
1.  **Dual-Dimension Risk Framework**:
    - **Risk Capacity** (The "Can"): Objective, financial ability to bear loss (Time Horizon, Liquidity, Net Worth).
    - **Risk Tolerance** (The "Want"): Subjective, psychological willingness to bear loss (Questionnaire, Behavior).
2.  **Constraint Principle**: Risk Capacity always acts as a hard ceiling on Risk Tolerance.
3.  **Mapping Matrix**: A 5x5 coordinate system maps scores to specific portfolio variants.

Portfolio Variants:
- **Defensive Dragon** ("The Bunker"): Capital Preservation.
- **Aggressive Dragon** ("The Hunter"): Geometric Compounding.
- **Standard Dragon**: Balanced 5-Pillar Approach.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json

class PortfolioVariant(Enum):
    DEFENSIVE_DRAGON = "Defensive Dragon"
    STANDARD_DRAGON = "Standard Dragon"
    AGGRESSIVE_DRAGON = "Aggressive Dragon"
    BALANCED_DRAGON = "Balanced Dragon" # Used for "Nervous Wealthy"

@dataclass
class ClientProfile:
    risk_capacity_score: int    # 0-100
    risk_tolerance_score: int   # 0-100
    variant: PortfolioVariant
    rationale: str

class IntakeForm:
    """
    Simulates the client questionnaire processing logic.
    """
    
    @staticmethod
    def calculate_capacity(answers: Dict[str, any]) -> int:
        """
        Calculates Risk Capacity (Financial).
        Inputs: Time Horizon, Liquidity, Income Stability, Liabilities.
        """
        score = 50 # Start neutral
        
        # 1. Time Horizon (Years)
        years = answers.get('time_horizon_years', 5)
        if years > 15: score += 40
        elif years > 10: score += 20
        elif years < 3: score -= 40
        elif years < 5: score -= 20
        
        # 2. Liquidity Needs
        liquidity = answers.get('liquidity_needs', 'Medium')
        if liquidity == 'Low': score += 10
        elif liquidity == 'High': score -= 30
        
        # 3. Net Worth / Stability
        net_worth_tier = answers.get('net_worth_tier', 'Medium')
        if net_worth_tier == 'High': score += 10
        elif net_worth_tier == 'Low': score -= 10
        
        return max(0, min(100, score))

    @staticmethod
    def calculate_tolerance(answers: Dict[str, any]) -> int:
        """
        Calculates Risk Tolerance (Psychological).
        Inputs: Questionnaire, Historical Behavior.
        """
        score = 50
        
        # 1. Market Drop Reaction (-20%)
        # Options: Sell All (0), Sell Some (30), Nothing (60), Buy More (100)
        reaction = answers.get('market_drop_reaction_score', 50)
        
        # Weight reaction heavily
        score = reaction
        
        return max(0, min(100, score))

class RoboAdvisor:
    def __init__(self):
        pass

    def map_portfolio(self, capacity: int, tolerance: int) -> Tuple[PortfolioVariant, str]:
        """
        Implements the 5x5 Mapping Matrix logic.
        """
        # Logic Table based on Blueprint
        
        # 1. Low Capacity (<40)
        if capacity < 40:
            if tolerance < 40:
                return (PortfolioVariant.DEFENSIVE_DRAGON, "Aligned Conservative. User is fragile and scared.")
            else: # Tolerance >= 40 (High/Medium)
                return (PortfolioVariant.DEFENSIVE_DRAGON, "Daredevil Conflict. User wants risk but cannot afford it. Capacity overrides.")

        # 2. High Capacity (>60)
        elif capacity > 60:
            if tolerance < 40:
                return (PortfolioVariant.BALANCED_DRAGON, "Nervous Wealthy Conflict. User can afford risk but fears it. System nudges slightly, but respects fear.")
            elif tolerance > 60:
                return (PortfolioVariant.AGGRESSIVE_DRAGON, "Aligned Aggressive. User is robust and brave.")
            else:
                return (PortfolioVariant.STANDARD_DRAGON, "High Capacity but Moderate Tolerance.")

        # 3. Medium Capacity (40-60)
        else:
            if tolerance > 60:
                 # Medium Cap, High Tol -> Lean Aggressive but capped? 
                 # Blueprint says "Medium/Medium = Aligned Moderate". 
                 # Let's default mixed cases to Standard.
                 return (PortfolioVariant.STANDARD_DRAGON, "Capacity constraints moderate the Aggressive Tolerance.")
            elif tolerance < 40:
                 return (PortfolioVariant.DEFENSIVE_DRAGON, "Low Tolerance drags down Moderate Capacity.")
            else:
                 return (PortfolioVariant.STANDARD_DRAGON, "Aligned Moderate.")

    def generate_recommendation(self, answers: Dict[str, any]) -> Dict:
        capacity = IntakeForm.calculate_capacity(answers)
        tolerance = IntakeForm.calculate_tolerance(answers)
        
        variant, rationale = self.map_portfolio(capacity, tolerance)
        
        # Define Allocations based on Variant
        if variant == PortfolioVariant.DEFENSIVE_DRAGON:
            weights = {
                "Equities": 0.10, "TIPS/Short-Term": 0.40, "Gold": 0.20, "Commodities": 0.10, "Cash": 0.20
            }
        elif variant == PortfolioVariant.AGGRESSIVE_DRAGON:
            weights = {
                "Equities": 0.30, "Long Treasuries": 0.10, "Gold": 0.20, "Commodity Trend": 0.25, "Long Volatility": 0.15
            }
        else: # Standard / Balanced
            weights = {
                "Equities": 0.20, "Fixed Income": 0.20, "Gold": 0.20, "Commodity Trend": 0.20, "Long Volatility": 0.20
            }

        return {
            "profile": {
                "risk_capacity": capacity,
                "risk_tolerance": tolerance,
                "variant": variant.value,
                "rationale": rationale
            },
            "allocation": weights
        }

# --- Example Usage ---
if __name__ == "__main__":
    advisor = RoboAdvisor()
    
    # Test Case 1: The "Daredevil" (High Tolerance, Low Capacity)
    # E.g., Young person with no job and high debt, but loves crypto.
    client_daredevil = {
        'time_horizon_years': 2,    # Needs money soon -> Low Capacity
        'liquidity_needs': 'High',  # Low Capacity
        'market_drop_reaction_score': 100 # Wants to buy the dip -> High Tolerance
    }
    
    # Test Case 2: The "Nervous Wealthy" (High Capacity, Low Tolerance)
    # E.g., Retired CEO with $10M but terrified of losing it.
    client_nervous = {
        'time_horizon_years': 20, # Long horizon (legacy) -> High Capacity
        'liquidity_needs': 'Low', 
        'net_worth_tier': 'High',
        'market_drop_reaction_score': 0 # Panic sells -> Low Tolerance
    }

    print("--- Client: Daredevil ---")
    print(json.dumps(advisor.generate_recommendation(client_daredevil), indent=2))
    
    print("\n--- Client: Nervous Wealthy ---")
    print(json.dumps(advisor.generate_recommendation(client_nervous), indent=2))
