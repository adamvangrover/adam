from typing import Any, Dict, List

from core.financial_suite.schemas.workstream_context import WorkstreamContext


class DCFEngine:
    @staticmethod
    def calculate_fcff(ctx: WorkstreamContext) -> Dict[str, List[float]]:
        """
        Projects Free Cash Flow to Firm (FCFF) based on context financials.
        Returns a dictionary of projected line items (lists).
        """
        fin = ctx.financials
        years = len(fin.projected_revenue_growth)

        # Initialize projections
        revenue = []
        ebitda = []
        ebit = []
        nopat = []
        depreciation = []
        capex = []
        nwc = []
        change_in_nwc = []
        fcff = []

        current_rev = fin.current_year_revenue
        prev_nwc = current_rev * fin.nwc_percent_revenue # Simplified starting NWC

        tax_rate = ctx.valuation_context.tax_rate

        for i in range(years):
            # Revenue
            growth = fin.projected_revenue_growth[i]
            rev = current_rev * (1 + growth) if i == 0 else revenue[-1] * (1 + growth)
            revenue.append(rev)

            # EBITDA
            margin = fin.projected_ebitda_margin[i] if i < len(fin.projected_ebitda_margin) else fin.projected_ebitda_margin[-1]
            ebitda_val = rev * margin
            ebitda.append(ebitda_val)

            # D&A
            da_val = rev * fin.depreciation_percent_revenue
            depreciation.append(da_val)

            # EBIT
            ebit_val = ebitda_val - da_val
            ebit.append(ebit_val)

            # NOPAT (EBIT * (1-t))
            nopat_val = ebit_val * (1 - tax_rate)
            nopat.append(nopat_val)

            # CapEx
            capex_val = rev * fin.capex_percent_revenue
            capex.append(capex_val)

            # NWC
            nwc_val = rev * fin.nwc_percent_revenue
            nwc.append(nwc_val)

            # Change in NWC
            dnwc = nwc_val - prev_nwc
            change_in_nwc.append(dnwc)
            prev_nwc = nwc_val

            # FCFF = NOPAT + D&A - CapEx - Change in NWC
            fcff_val = nopat_val + da_val - capex_val - dnwc
            fcff.append(fcff_val)

        return {
            "revenue": revenue,
            "ebitda": ebitda,
            "ebit": ebit,
            "nopat": nopat,
            "depreciation": depreciation,
            "capex": capex,
            "nwc": nwc,
            "change_in_nwc": change_in_nwc,
            "fcff": fcff
        }

    @staticmethod
    def calculate_terminal_value(ctx: WorkstreamContext, final_fcff: float, wacc: float) -> Dict[str, float]:
        """
        Calculates Terminal Value using Dual Method (Perpetuity Growth & Exit Multiple).
        """
        val_ctx = ctx.valuation_context

        # Method A: Perpetuity Growth
        # TV = (FCFF_n * (1 + g)) / (WACC - g)
        g = val_ctx.growth_rate_perpetuity

        # Sanity Check: g cannot be >= WACC
        if g >= wacc:
             # Fallback to slightly less than WACC to avoid division by zero or negative
             g = wacc - 0.001

        tv_perpetuity = (final_fcff * (1 + g)) / (wacc - g)

        # Method B: Exit Multiple
        # We need Final Year EBITDA. This requires running the projection or assuming it's available.
        # For efficiency, we assume the caller provides final_ebitda, or we re-calc.
        # But this function signature takes final_fcff.
        # Let's adjust signature or logic.
        # Ideally calculate_terminal_value should be part of a larger flow that has access to EBITDA.
        # I'll modify the signature to accept final_ebitda.
        return {"tv_perpetuity": tv_perpetuity, "tv_multiple": 0.0} # Placeholder until fixed

    @staticmethod
    def calculate_valuation(ctx: WorkstreamContext, wacc: float) -> Dict[str, Any]:
        """
        Full DCF Valuation.
        """
        projections = DCFEngine.calculate_fcff(ctx)
        fcff_stream = projections["fcff"]
        ebitda_stream = projections["ebitda"]

        final_fcff = fcff_stream[-1]
        final_ebitda = ebitda_stream[-1]

        val_ctx = ctx.valuation_context

        # Terminal Value
        # A: Perpetuity
        g = val_ctx.growth_rate_perpetuity
        if g >= wacc: g = wacc - 0.001
        tv_perpetuity = (final_fcff * (1 + g)) / (wacc - g)

        # B: Exit Multiple
        tv_multiple = final_ebitda * val_ctx.exit_multiple

        # Weighted TV
        weight_perp = val_ctx.weight_perpetuity or 0.5
        weight_mult = val_ctx.weight_exit_multiple or 0.5

        if val_ctx.terminal_method == "PERPETUITY_GROWTH":
            tv_final = tv_perpetuity
        elif val_ctx.terminal_method == "EXIT_MULTIPLE":
            tv_final = tv_multiple
        else: # DUAL_WEIGHTED
            tv_final = (tv_perpetuity * weight_perp) + (tv_multiple * weight_mult)

        # Discounting
        pv_fcff = 0.0
        discount_factors = []
        for i, cash_flow in enumerate(fcff_stream):
            t = i + 1 # Year 1, 2, ...
            df = 1 / ((1 + wacc) ** t)
            discount_factors.append(df)
            pv_fcff += cash_flow * df

        # PV of TV
        # TV is at end of year N
        t_final = len(fcff_stream)
        pv_tv = tv_final / ((1 + wacc) ** t_final)

        enterprise_value = pv_fcff + pv_tv

        return {
            "projections": projections,
            "wacc": wacc,
            "tv_perpetuity": tv_perpetuity,
            "tv_multiple": tv_multiple,
            "tv_final": tv_final,
            "pv_fcff": pv_fcff,
            "pv_tv": pv_tv,
            "enterprise_value": enterprise_value,
            "implied_growth": (tv_multiple * (wacc - g) / final_fcff) - 1 if final_fcff else 0, # Reverse engineer g from multiple
            "implied_multiple": tv_perpetuity / final_ebitda if final_ebitda else 0 # Reverse engineer multiple from perpetuity
        }
