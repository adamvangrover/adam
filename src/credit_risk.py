# src/credit_risk.py
from .config import RATING_MAP


class CreditSponsorModel:
    def __init__(self, enterprise_value, total_debt, ebitda, interest_expense):
        self.ev = enterprise_value
        self.debt = total_debt
        self.ebitda = ebitda
        self.interest = interest_expense

    def calculate_metrics(self):
        leverage = self.debt / self.ebitda
        ltv = self.debt / self.ev
        fccr = (self.ebitda - (0.1 * self.ebitda)) / self.interest # Simplified (EBITDA-Capex)/Int

        return {
            "Leverage (x)": round(leverage, 2),
            "LTV (%)": round(ltv * 100, 2),
            "FCCR (x)": round(fccr, 2)
        }

    def determine_regulatory_rating(self, metrics):
        """
        Heuristic logic to assign a Regulatory Rating based on FCCR and Leverage.
        """
        lev = metrics['Leverage (x)']
        fccr = metrics['FCCR (x)']

        if lev < 3.0 and fccr > 2.5: return RATING_MAP[3.0] # IG
        if lev < 4.5 and fccr > 1.5: return RATING_MAP[4.0] # BB
        if lev < 6.0 and fccr > 1.1: return RATING_MAP[5.0] # B+
        if lev < 7.5 and fccr > 1.0: return RATING_MAP[6.0] # B-
        return RATING_MAP[8.0] # Substandard

    def perform_downside_stress(self, stress_factor=0.20):
        """
        Stresses EBITDA by stress_factor (e.g., 20% decline) to see impact on PD/Rating.
        """
        stressed_ebitda = self.ebitda * (1 - stress_factor)
        stressed_ev = self.ev * (1 - stress_factor) # EV typically correlates

        # Re-calc metrics
        lev = self.debt / stressed_ebitda
        ltv = self.debt / stressed_ev
        fccr = (stressed_ebitda * 0.9) / self.interest

        metrics = {"Leverage (x)": lev, "LTV (%)": ltv * 100, "FCCR (x)": fccr}
        rating = self.determine_regulatory_rating(metrics)

        return metrics, rating

    def snc_check(self):
        """
        Shared National Credit Logic: Debt > $100MM check.
        """
        if self.debt > 100_000_000:
            return "SNC REPORTING REQUIRED: Flag for Regulatory Review"
        return "Non-SNC"
