"""
Module 3: Robo-Advisor Logic & Intake Engine
============================================
Architecture: Constraints-Based Advisory System
Objective: Map client profiles to "Gold Standard" portfolio variants.
"""

from typing import Dict, Any, Union, List

class IntakeForm:
    """
    Manages the questions and scoring logic.
    """
    @staticmethod
    def get_questions() -> List[Dict]:
        return [
            {
                "id": "Q1",
                "text": "When do you need to withdraw >20% of these funds?",
                "options": {
                    "<3 Years": 0,
                    "3-5 Years": 25,
                    "6-10 Years": 65,
                    "11+ Years": 100
                }
            },
            {
                "id": "Q2",
                "text": "What represents your liquidity status?",
                "options": {
                    "No other savings": 0,
                    "3 months expenses": 50,
                    "6+ months expenses": 100
                }
            },
            {
                "id": "Q3",
                "text": "In 2008, markets fell 50%. If you owned stocks, you would have:",
                "options": {
                    "Sold everything": 0,
                    "Sold some": 30,
                    "Held steady": 70,
                    "Bought more": 100
                }
            },
            {
                "id": "Q4",
                "text": "You prefer a portfolio that:",
                "options": {
                    "Avoids loss at all costs": 0,
                    "Balances safety/growth": 50,
                    "Maximizes growth despite swings": 100
                }
            }
        ]

    @staticmethod
    def calculate_risk_profile(user_answers: Dict[str, int]) -> Dict[str, Any]:
        """
        Core logic for determining client suitability.
        Implements the 'Capacity Ceiling' safety protocol.
        
        Args:
            user_answers: Dict where keys are 'Q1_score', 'Q2_score', etc.
        """
        
        # --- STEP 1: Calculate Financial Capacity (Objective) ---
        # Time Horizon is weighted heavily (75%) as it dictates recovery time.
        # Liquidity is secondary (25%).
        # Default to 0 if key missing for safety
        q1 = user_answers.get('Q1_score', 0)
        q2 = user_answers.get('Q2_score', 0)
        raw_capacity = (q1 * 0.75) + (q2 * 0.25)
        
        # --- STEP 2: Calculate Psychological Tolerance (Subjective) ---
        # Averaged equally across behavioral questions.
        q3 = user_answers.get('Q3_score', 0)
        q4 = user_answers.get('Q4_score', 0)
        raw_tolerance = (q3 + q4) / 2.0
        
        # --- STEP 3: The Constraint Logic (The Safety Valve) ---
        # The Final Score is the MINIMUM of Capacity and Tolerance.
        # Example: High Tolerance (90) but Low Capacity (20) -> Final Score 20.
        # This prevents high-risk portfolios for clients who cannot afford losses.
        final_risk_score = min(raw_capacity, raw_tolerance)
        
        # --- STEP 4: Portfolio Mapping ---
        portfolio_variant = RoboAdvisor.map_score_to_portfolio(final_risk_score)
        
        return {
            "capacity_score": raw_capacity,
            "tolerance_score": raw_tolerance,
            "final_risk_score": final_risk_score,
            "recommended_portfolio": portfolio_variant,
            "warning_flag": "Capacity Constraint Triggered" if raw_capacity < raw_tolerance else "None"
        }

class RoboAdvisor:
    @staticmethod
    def map_score_to_portfolio(score: float) -> str:
        if score < 30:
            return "Conservative_Income_Preservation"
        elif 30 <= score < 50:
            return "Balanced_Defensive"
        elif 50 <= score < 80:
            return "Hybrid_All_Weather_Strategic" # The Gold Standard Portfolio
        else:
            return "Aggressive_Growth_Equity_Focus"

    @staticmethod
    def get_portfolio_details(variant: str) -> Dict[str, str]:
        """
        Returns the allocation summary for a variant.
        """
        table = {
            "Conservative_Income_Preservation": {
                "Equity": "20%", "Fixed_Income": "70%", "Alternatives": "10%",
                "Objective": "Capital Preservation"
            },
            "Balanced_Defensive": {
                "Equity": "30%", "Fixed_Income": "50%", "Alternatives": "20%",
                "Objective": "Inflation Protection"
            },
            "Hybrid_All_Weather_Strategic": {
                "Equity": "35%", "Fixed_Income": "35%", "Alternatives": "30%",
                "Objective": "Strategic Growth & Resilience"
            },
            "Aggressive_Growth_Equity_Focus": {
                "Equity": "80%", "Fixed_Income": "10%", "Alternatives": "10%",
                "Objective": "Maximum Accumulation"
            }
        }
        return table.get(variant, {})

if __name__ == "__main__":
    import json
    # Example usage based on prompt
    
    # Example 1: Young Professional (High Capacity, High Tolerance)
    # Q1: 11+ Years (100)
    # Q2: 6+ months (100)
    # Q3: Bought more (100)
    # Q4: Maximize growth (100)
    user_a = {'Q1_score': 100, 'Q2_score': 100, 'Q3_score': 100, 'Q4_score': 100}
    
    # Example 2: Retiree (Low Capacity, High Tolerance)
    # Q1: <3 Years (0)
    # Q2: 3 months (50)
    # Q3: Bought more (100)
    # Q4: Balances (50)
    user_b = {'Q1_score': 0, 'Q2_score': 50, 'Q3_score': 100, 'Q4_score': 50}
    
    print("--- Client A (Young Pro) ---")
    res_a = IntakeForm.calculate_risk_profile(user_a)
    print(json.dumps(res_a, indent=2))
    
    print("\n--- Client B (Retiree) ---")
    res_b = IntakeForm.calculate_risk_profile(user_b)
    print(json.dumps(res_b, indent=2))
