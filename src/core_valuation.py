# src/core_valuation.py
import pandas as pd
from typing import List, Tuple
from .config import DEFAULT_ASSUMPTIONS

class ValuationEngine:
    """
    Computes enterprise value using a Discounted Cash Flow (DCF) model.
    """
    def __init__(self, ebitda_base: float, capex_percent: float, nwc_percent: float, debt_cost: float, equity_percent: float):
        if not (0 <= equity_percent <= 1):
            raise ValueError("equity_percent must be between 0 and 1.")
        if ebitda_base <= 0:
            raise ValueError("EBITDA must be greater than zero for valuation.")

        self.ebitda = float(ebitda_base)
        self.capex_pct = float(capex_percent)
        self.nwc_pct = float(nwc_percent)
        self.kd = float(debt_cost)
        self.we = float(equity_percent)
        self.wd = 1.0 - self.we
        self.t = float(DEFAULT_ASSUMPTIONS['tax_rate'])

    def calculate_wacc(self) -> float:
        """
        Calculates the Weighted Average Cost of Capital (WACC).
        """
        rf = float(DEFAULT_ASSUMPTIONS['risk_free_rate'])
        rm = float(DEFAULT_ASSUMPTIONS['market_risk_premium'])
        beta = float(DEFAULT_ASSUMPTIONS['beta'])

        ke = rf + beta * rm
        wacc = (self.we * ke) + (self.wd * self.kd * (1 - self.t))
        return wacc

    def run_dcf(self, growth_rates: List[float], terminal_growth_rate: float = None) -> Tuple[pd.DataFrame, float, float]:
        """
        Generates Free Cash Flow (FCF) projections and calculates Enterprise Value.

        Args:
            growth_rates: A list of growth rates for each projection year.

        Returns:
            A tuple containing:
                - A pandas DataFrame with the projections.
                - The calculated Enterprise Value.
                - The calculated WACC.
        """
        years = int(DEFAULT_ASSUMPTIONS['projection_years'])
        if len(growth_rates) != years:
            raise ValueError(f"Expected exactly {years} growth rates, got {len(growth_rates)}.")

        wacc = self.calculate_wacc()
        g = float(terminal_growth_rate if terminal_growth_rate is not None else DEFAULT_ASSUMPTIONS['terminal_growth_rate'])

        if wacc <= g:
            raise ValueError(f"WACC ({wacc:.4f}) must be greater than terminal growth rate ({g:.4f}) to calculate terminal value.")

        projections = []
        current_ebitda = self.ebitda

        for i, gr in enumerate(growth_rates):
            current_ebitda *= (1 + gr)
            tax = current_ebitda * self.t  # Simplified EBIT proxy
            capex = current_ebitda * self.capex_pct
            nwc_change = current_ebitda * self.nwc_pct
            fcf = current_ebitda - tax - capex - nwc_change

            df = 1 / ((1 + wacc) ** (i + 1))
            pv_fcf = fcf * df

            projections.append({
                "Year": i + 1,
                "EBITDA": current_ebitda,
                "FCF": fcf,
                "PV_FCF": pv_fcf
            })

        df_proj = pd.DataFrame(projections)

        # Terminal Value (Gordon Growth)
        terminal_fcf = projections[-1]['FCF'] * (1 + g)
        tv = terminal_fcf / (wacc - g)
        pv_tv = tv / ((1 + wacc) ** years)

        enterprise_value = df_proj['PV_FCF'].sum() + pv_tv

        return df_proj, enterprise_value, wacc
