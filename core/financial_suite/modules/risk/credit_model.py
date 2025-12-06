import numpy as np
from scipy.stats import norm
from core.financial_suite.schemas.workstream_context import WorkstreamContext

class CreditEngine:
    @staticmethod
    def calculate_merton_pd(ev: float, total_debt: float, volatility: float, risk_free_rate: float, time_horizon: float = 1.0) -> float:
        """
        Calculates Probability of Default (PD) using Merton Structural Model.
        """
        if total_debt <= 0:
            return 0.0001 # Non-zero small probability
        if ev <= 0:
            return 1.0

        V = ev
        D = total_debt
        sigma = volatility
        r = risk_free_rate
        T = time_horizon

        d1 = (np.log(V / D) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # PD = N(-d2)
        pd = norm.cdf(-d2)
        return float(pd)

    @staticmethod
    def calculate_logistic_pd(ctx: WorkstreamContext, ev: float, total_debt: float, ebitda: float) -> float:
        """
        Calculates PD using a Logistic Regression (Z-Score) approach.
        Simplified for architectural demonstration.
        """
        # Feature Engineering
        leverage = total_debt / ebitda if ebitda > 0 else 100.0
        ltv = total_debt / ev if ev > 0 else 1.0
        interest_coverage = ebitda / (total_debt * 0.08) # Approximation if interest not available locally

        # Coefficients (Mock calibrated)
        intercept = -3.5
        coef_leverage = 0.5
        coef_ltv = 2.0

        z = intercept + (coef_leverage * leverage) + (coef_ltv * ltv)

        # Sigmoid
        pd = 1 / (1 + np.exp(-z))
        return float(pd)

    @staticmethod
    def calculate_pd(ctx: WorkstreamContext, ev: float) -> float:
        """
        Router for PD calculation based on context configuration.
        """
        # Sum of debt
        total_debt = sum(s.balance for s in ctx.capital_structure.securities if s.security_type in ["REVOLVER", "TERM_LOAN", "MEZZANINE"])

        if ctx.credit_challenge.pd_method == "MERTON_STRUCTURAL":
            vol = ctx.credit_challenge.asset_volatility or 0.30
            rf = ctx.valuation_context.risk_free_rate
            return CreditEngine.calculate_merton_pd(ev, total_debt, vol, rf)

        elif ctx.credit_challenge.pd_method == "LOGISTIC_HYBRID":
            # Need EBITDA. Assuming current year or average.
            # Using current year derived from margins
            rev = ctx.financials.current_year_revenue
            margin = ctx.financials.historical_ebitda_margin[-1] if ctx.financials.historical_ebitda_margin else 0.20
            ebitda = rev * margin
            return CreditEngine.calculate_logistic_pd(ctx, ev, total_debt, ebitda)

        else:
            return 0.05 # Default fallback

    @staticmethod
    def calculate_lgd(ctx: WorkstreamContext) -> float:
        """
        Calculates Weighted LGD based on collateral mix.
        """
        if not ctx.collateral:
            return 0.45 # Default LGD

        c = ctx.collateral
        total_assets = c.cash_equivalents + c.accounts_receivable + c.inventory + c.ppe + c.intangibles
        if total_assets == 0:
            return 1.0

        # LGD Rates from Spec 4.3
        # Cash/Rec: 10%
        # Inventory: 50%
        # PP&E: 40-60% (Using 50% avg)
        # Intangibles: 90-100% (Using 100%)

        weighted_loss = (
            (c.cash_equivalents + c.accounts_receivable) * 0.10 +
            (c.inventory) * 0.50 +
            (c.ppe) * 0.50 +
            (c.intangibles) * 1.00
        )

        return weighted_loss / total_assets

    @staticmethod
    def calculate_expected_loss(ctx: WorkstreamContext, pd: float) -> float:
        """
        Calculates Expected Loss (EL) = EAD * PD * LGD.
        """
        # EAD = Drawn Balance + (Unused * LEQ)
        # Simplified: Sum of all debt
        total_debt = sum(s.balance for s in ctx.capital_structure.securities if s.security_type in ["REVOLVER", "TERM_LOAN", "MEZZANINE"])
        ead = total_debt # Assuming fully drawn for simplicity

        lgd = CreditEngine.calculate_lgd(ctx)

        return ead * pd * lgd
