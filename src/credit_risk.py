# src/credit_risk.py
from typing import Dict, Any, Tuple
import json
import os
from json_logic import jsonLogic
from .config import RATING_MAP
_RATING_RULE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'covenants', 'rating_rules.json')
with open(_RATING_RULE_PATH, 'r') as f:
    _RATING_RULES = json.load(f)

_SNC_RULE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'covenants', 'snc_rules.json')
with open(_SNC_RULE_PATH, 'r') as f:
    _SNC_RULES = json.load(f)


class CreditSponsorModel:
    """
    Evaluates credit risk metrics, assigns regulatory ratings, and performs stress testing.
    """
    def __init__(self, enterprise_value: float, total_debt: float, ebitda: float, interest_expense: float):
        if enterprise_value <= 0:
            raise ValueError("Enterprise Value must be greater than zero.")
        if total_debt < 0:
            raise ValueError("Total Debt cannot be negative.")
        if ebitda <= 0:
            raise ValueError("EBITDA must be greater than zero for valid leverage calculation.")
        if interest_expense <= 0:
            raise ValueError("Interest Expense must be greater than zero for valid FCCR calculation.")

        self.ev = float(enterprise_value)
        self.debt = float(total_debt)
        self.ebitda = float(ebitda)
        self.interest = float(interest_expense)

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculates key credit metrics: Leverage, Loan-to-Value (LTV), and Fixed Charge Coverage Ratio (FCCR).
        """
        leverage = self.debt / self.ebitda
        ltv = self.debt / self.ev
        fccr = (self.ebitda - (0.1 * self.ebitda)) / self.interest # Simplified (EBITDA-Capex)/Int

        return {
            "Leverage (x)": round(leverage, 2),
            "LTV (%)": round(ltv * 100, 2),
            "FCCR (x)": round(fccr, 2)
        }

    def determine_regulatory_rating(self, metrics: Dict[str, float]) -> str:
        """
        Heuristic logic to assign a Regulatory Rating based on FCCR and Leverage.
        """
        if 'Leverage (x)' not in metrics or 'FCCR (x)' not in metrics:
            raise KeyError("Metrics dictionary must contain 'Leverage (x)' and 'FCCR (x)'.")

        try:
            rating_key = jsonLogic(_RATING_RULES, metrics)
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate regulatory rating rule: {e}")
        return RATING_MAP[rating_key]

    def perform_downside_stress(self, stress_factor: float = 0.20) -> Tuple[Dict[str, float], str]:
        """
        Stresses EBITDA by stress_factor (e.g., 20% decline) to see impact on PD/Rating.
        """
        if not (0 <= stress_factor < 1):
            raise ValueError("stress_factor must be between 0 and less than 1.")

        stressed_ebitda = self.ebitda * (1 - stress_factor)
        stressed_ev = self.ev * (1 - stress_factor) # EV typically correlates

        if stressed_ebitda <= 0:
            raise ValueError("Stress factor resulted in non-positive EBITDA, cannot compute valid metrics.")
        if stressed_ev <= 0:
            raise ValueError("Stress factor resulted in non-positive EV, cannot compute valid metrics.")

        # Re-calc metrics
        lev = self.debt / stressed_ebitda
        ltv = self.debt / stressed_ev
        fccr = (stressed_ebitda * 0.9) / self.interest

        metrics = {"Leverage (x)": round(lev, 2), "LTV (%)": round(ltv * 100, 2), "FCCR (x)": round(fccr, 2)}
        rating = self.determine_regulatory_rating(metrics)

        return metrics, rating

    def snc_check(self) -> str:
        """
        Shared National Credit Logic: Debt > $100MM check.
        """
        try:
            return jsonLogic(_SNC_RULES, {"debt": self.debt})
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate SNC rule: {e}")
