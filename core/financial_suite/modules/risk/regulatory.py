from typing import Any, Dict, Tuple

from core.financial_suite.schemas.workstream_context import WorkstreamContext


class RegulatoryEngine:
    @staticmethod
    def get_rating_from_metrics(pd: float, fccr: float, ltv: float) -> Tuple[int, str]:
        """
        Determines Regulatory Rating based on PD, FCCR, and LTV.
        Uses the 'Most Severe' logic (max rating number).
        """
        # PD Rating
        if pd < 0.005: pd_rating = 3
        elif pd < 0.012: pd_rating = 4
        elif pd < 0.025: pd_rating = 5
        elif pd < 0.050: pd_rating = 6
        elif pd < 0.150: pd_rating = 7
        elif pd < 0.500: pd_rating = 8
        else: pd_rating = 9

        # FCCR Rating
        if fccr > 2.0: fccr_rating = 3
        elif fccr > 1.5: fccr_rating = 4
        elif fccr > 1.25: fccr_rating = 5
        elif fccr > 1.10: fccr_rating = 6
        elif fccr > 1.00: fccr_rating = 7 # Implicit, < 1.1
        elif fccr > 0: fccr_rating = 8
        else: fccr_rating = 9

        # LTV Rating
        if ltv < 0.40: ltv_rating = 3
        elif ltv < 0.55: ltv_rating = 4
        elif ltv < 0.65: ltv_rating = 5
        elif ltv < 0.80: ltv_rating = 6
        elif ltv < 1.00: ltv_rating = 7 # > 80%
        else: ltv_rating = 8 # > 100%

        final_rating = max(pd_rating, fccr_rating, ltv_rating)

        descriptions = {
            1: "Pass (Strong)", 2: "Pass (Strong)", 3: "Pass (Strong)",
            4: "Pass",
            5: "Pass / Watch",
            6: "Special Mention",
            7: "Substandard",
            8: "Doubtful",
            9: "Loss"
        }

        return final_rating, descriptions.get(final_rating, "Unknown")

    @staticmethod
    def analyze_snc_compliance(ctx: WorkstreamContext, pd: float, ev: float, ebitda: float) -> Dict[str, Any]:
        """
        Performs full SNC regulatory analysis.
        """
        total_debt = sum(s.balance for s in ctx.capital_structure.securities if s.security_type in ["REVOLVER", "TERM_LOAN", "MEZZANINE"])
        interest_expense = sum(s.balance * s.interest_rate for s in ctx.capital_structure.securities)

        # Assuming 5% mandatory amortization for Term Loans
        mandatory_principal = sum(s.balance * 0.05 for s in ctx.capital_structure.securities if s.security_type == "TERM_LOAN")

        fixed_charges = interest_expense + mandatory_principal
        fccr = ebitda / fixed_charges if fixed_charges > 0 else 99.0
        ltv = total_debt / ev if ev > 0 else 0.0

        rating, description = RegulatoryEngine.get_rating_from_metrics(pd, fccr, ltv)

        # Cost of Debt Spread Logic (SNC Repricing)
        # If Rating >= 6 (Special Mention), spread widens
        spread_adjustment = 0.0
        if rating >= 6:
            spread_adjustment = 0.0400 # +400 bps

        return {
            "rating": rating,
            "description": description,
            "fccr": fccr,
            "ltv": ltv,
            "spread_adjustment": spread_adjustment,
            "metrics": {
                "pd": pd,
                "total_debt": total_debt,
                "ebitda": ebitda
            }
        }
