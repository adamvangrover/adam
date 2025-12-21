# src/core_valuation.py
import pandas as pd

from .config import DEFAULT_ASSUMPTIONS


class ValuationEngine:
    def __init__(self, ebitda_base, capex_percent, nwc_percent, debt_cost, equity_percent):
        self.ebitda = ebitda_base
        self.capex_pct = capex_percent
        self.nwc_pct = nwc_percent
        self.kd = debt_cost
        self.we = equity_percent
        self.wd = 1 - equity_percent
        self.t = DEFAULT_ASSUMPTIONS['tax_rate']

    def calculate_wacc(self):
        rf = DEFAULT_ASSUMPTIONS['risk_free_rate']
        rm = DEFAULT_ASSUMPTIONS['market_risk_premium']
        beta = DEFAULT_ASSUMPTIONS['beta']

        ke = rf + beta * (rm)
        wacc = (self.we * ke) + (self.wd * self.kd * (1 - self.t))
        return wacc

    def run_dcf(self, growth_rates):
        """
        Generates Free Cash Flow (FCF) projections.
        """
        wacc = self.calculate_wacc()
        years = DEFAULT_ASSUMPTIONS['projection_years']
        g = DEFAULT_ASSUMPTIONS['terminal_growth_rate']

        projections = []
        current_ebitda = self.ebitda

        for i, gr in enumerate(growth_rates):
            current_ebitda *= (1 + gr)
            tax = current_ebitda * self.t # Simplified EBIT proxy
            capex = current_ebitda * self.capex_pct
            nwc_change = current_ebitda * self.nwc_pct
            fcf = current_ebitda - tax - capex - nwc_change

            df = 1 / ((1 + wacc) ** (i + 1))
            pv_fcf = fcf * df

            projections.append({
                "Year": i+1,
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
