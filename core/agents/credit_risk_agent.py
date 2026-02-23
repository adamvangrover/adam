from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import numpy as np
from core.agents.agent_base import AgentBase

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class CreditRiskAgent(AgentBase):
    """
    Agent responsible for assessing Credit Risk (Default Risk).
    Calculates Altman Z-Score (Manufacturing & Non-Manufacturing),
    Merton Distance to Default, and implied Credit Ratings.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the credit risk assessment.

        Args:
            financial_data: Dictionary containing:
                - working_capital
                - retained_earnings
                - ebit
                - total_assets
                - market_value_equity (or book value)
                - total_liabilities
                - sales
                - industry_type (optional, default "Manufacturing")
                - volatility (optional, for Merton model)
                - risk_free_rate (optional, default 0.04)

        Returns:
            Dict containing credit risk metrics.
        """
        logger.info("Starting Credit Risk Assessment...")

        try:
            # Extract inputs
            wc = financial_data.get("working_capital", 0)
            re = financial_data.get("retained_earnings", 0)
            ebit = financial_data.get("ebit", 0)
            ta = financial_data.get("total_assets", 1)
            mve = financial_data.get("market_value_equity", 0)
            tl = financial_data.get("total_liabilities", 1)
            sales = financial_data.get("sales", 0)
            industry = financial_data.get("industry_type", "Manufacturing")
            volatility = financial_data.get("volatility", 0.30) # Default 30% vol
            rf_rate = financial_data.get("risk_free_rate", 0.04)

            if ta <= 0: ta = 1.0
            if tl <= 0: tl = 1.0

            # 1. Altman Z-Score
            # A = Working Capital / Total Assets
            # B = Retained Earnings / Total Assets
            # C = EBIT / Total Assets
            # D = Market Value of Equity / Total Liabilities
            # E = Sales / Total Assets

            A = wc / ta
            B = re / ta
            C = ebit / ta
            D = mve / tl
            E = sales / ta

            z_score = self._calculate_z_score(A, B, C, D, E, industry)
            z_score_interpretation = self._interpret_z_score(z_score, industry)

            # 2. Merton Model (Distance to Default)
            merton_dd = None
            implied_pd = None
            credit_rating = "N/A"

            if SCIPY_AVAILABLE:
                # Need mve and tl to be non-zero for meaningful Merton
                if mve > 0 and tl > 0:
                    merton_dd = self._calculate_merton_dd(mve, tl, volatility, rf_rate)
                    implied_pd = self._pd_from_dd(merton_dd)
                    credit_rating = self._map_pd_to_rating(implied_pd)

            # 3. Fundamental Credit Ratios
            # DSCR, Interest Coverage, Debt/EBITDA
            ebitda = financial_data.get("ebitda", ebit) # Fallback to EBIT if no EBITDA
            interest_expense = financial_data.get("interest_expense", 0)
            debt_service = financial_data.get("debt_service", interest_expense) # Fallback
            total_debt = financial_data.get("total_debt", tl) # Fallback to TL

            ratios = self._calculate_fundamental_ratios(ebitda, ebit, interest_expense, debt_service, total_debt)

            result = {
                "z_score": float(z_score),
                "z_score_model": "Manufacturing (Z)" if industry == "Manufacturing" else "Non-Manufacturing (Z'')",
                "zone": z_score_interpretation["zone"],
                "z_score_distress_level": z_score_interpretation["message"],
                "merton_distance_to_default": float(merton_dd) if merton_dd is not None else None,
                "implied_probability_of_default": float(implied_pd) if implied_pd is not None else None,
                "estimated_credit_rating": credit_rating,
                "fundamental_ratios": ratios,
                "components": {
                    "A_Liquidity": float(A),
                    "B_Leverage": float(B),
                    "C_Profitability": float(C),
                    "D_Solvency": float(D),
                    "E_Activity": float(E)
                }
            }

            logger.info(f"Credit Risk Assessment Complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Error calculating credit risk: {e}")
            return {"error": str(e)}

    def _calculate_z_score(self, A, B, C, D, E, industry_type):
        """Calculates Altman Z-Score based on industry type."""
        if industry_type == "Manufacturing":
            # Original Z-Score
            return 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
        else:
            # Altman Z''-Score (Emerging Markets / Non-Manufacturing)
            # Coefficients: 6.56, 3.26, 6.72, 1.05
            # Note: The coefficients are significantly higher because the cutoffs are different.
            # Sales/Assets (E) is omitted.
            return 6.56 * A + 3.26 * B + 6.72 * C + 1.05 * D

    def _interpret_z_score(self, z_score, industry_type):
        """Interprets the Z-Score based on the model used."""
        if industry_type == "Manufacturing":
            if z_score > 2.99:
                return {"zone": "Safe", "message": "Low Default Probability"}
            elif z_score > 1.81:
                return {"zone": "Grey", "message": "Moderate Default Probability"}
            else:
                return {"zone": "Distress", "message": "High Default Probability"}
        else:
            # Z'' Score thresholds are different
            # Safe > 2.60
            # Grey 1.10 - 2.60
            # Distress < 1.10
            if z_score > 2.60:
                return {"zone": "Safe", "message": "Low Default Probability"}
            elif z_score > 1.10:
                return {"zone": "Grey", "message": "Moderate Default Probability"}
            else:
                return {"zone": "Distress", "message": "High Default Probability"}

    def _calculate_merton_dd(self, equity: float, debt: float, vol: float, r: float, T: float = 1.0) -> float:
        """
        Calculates Distance to Default (DD) using a simplified Merton Model.
        Assuming 'vol' is Equity Volatility.
        """
        # Estimate Asset Value (V) roughly as Equity + Debt
        # Ideally, we solve for V and Asset Volatility iteratively.
        # Approximation:
        V = equity + debt

        # Unlevering volatility: sigma_A approx sigma_E * (E/V)
        sigma_A = vol * (equity / V)

        # Avoid division by zero
        if sigma_A == 0: sigma_A = 0.01

        # Distance to Default (d2 in Black-Scholes context relative to debt face value)
        # Using the standard KMV-like definition:
        # DD = (Asset Value - Default Point) / (Asset Value * Asset Volatility)
        # Default Point is often Short Term Debt + 0.5 * Long Term Debt. Here we just use 'debt'.

        numerator = np.log(V / debt) + (r - 0.5 * sigma_A**2) * T
        denominator = sigma_A * np.sqrt(T)

        dd = numerator / denominator
        return float(dd)

    def _pd_from_dd(self, dd: float) -> float:
        """Probability of Default from Distance to Default."""
        # PD = N(-DD)
        return float(norm.cdf(-dd))

    def _map_pd_to_rating(self, pd: float) -> str:
        """Maps Probability of Default to approximate Credit Rating (S&P scale)."""
        if pd <= 0.0005: return "AAA"    # <= 0.05%
        if pd <= 0.0010: return "AA+"
        if pd <= 0.0020: return "AA"
        if pd <= 0.0030: return "AA-"
        if pd <= 0.0050: return "A+"
        if pd <= 0.0070: return "A"
        if pd <= 0.0100: return "A-"
        if pd <= 0.0150: return "BBB+"
        if pd <= 0.0250: return "BBB"
        if pd <= 0.0500: return "BBB-"   # Investment Grade Cutoff (~5%)
        if pd <= 0.0750: return "BB+"
        if pd <= 0.1000: return "BB"
        if pd <= 0.1500: return "BB-"
        if pd <= 0.2000: return "B+"
        if pd <= 0.2500: return "B"
        if pd <= 0.3000: return "B-"
        if pd <= 0.4000: return "CCC+"
        if pd <= 0.5000: return "CCC"
        if pd <= 0.6000: return "CCC-"
        return "D"

    def _calculate_fundamental_ratios(self, ebitda, ebit, interest_expense, debt_service, total_debt):
        """Calculates key credit ratios."""
        ratios = {}

        # Interest Coverage Ratio = EBIT / Interest Expense
        if interest_expense > 0:
            icr = ebit / interest_expense
            ratios["interest_coverage_ratio"] = float(icr)
            ratios["icr_assessment"] = "Healthy" if icr > 3.0 else "Weak" if icr < 1.5 else "Adequate"
        else:
            ratios["interest_coverage_ratio"] = None

        # DSCR = EBITDA / Debt Service
        if debt_service > 0:
            dscr = ebitda / debt_service
            ratios["dscr"] = float(dscr)
            ratios["dscr_assessment"] = "Strong" if dscr > 1.25 else "Breakeven" if dscr >= 1.0 else "Deficit"
        else:
            ratios["dscr"] = None

        # Leverage Ratio = Total Debt / EBITDA
        if ebitda > 0:
            lev = total_debt / ebitda
            ratios["debt_to_ebitda"] = float(lev)
            ratios["leverage_assessment"] = "Low" if lev < 2.0 else "High" if lev > 4.0 else "Moderate"
        else:
             ratios["debt_to_ebitda"] = None

        return ratios
