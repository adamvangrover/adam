from core.financial_suite.engines.dcf import DCFEngine
from core.financial_suite.engines.wacc import WACCCalculator
from core.financial_suite.modules.risk.credit_model import CreditEngine
from core.financial_suite.modules.risk.regulatory import RegulatoryEngine
from core.financial_suite.schemas.workstream_context import WorkstreamContext


class IterativeSolver:
    @staticmethod
    def solve_equilibrium(ctx: WorkstreamContext, tolerance: float = 0.0001, max_iter: int = 20):
        """
        Solves the circular dependency between Valuation, Leverage, Rating, and WACC.
        Returns the updated Context (with computed WACC/Rating) and the Valuation Result.
        """
        # Working copy
        # We don't deep copy here to allow updating the passed context?
        # Or we return a new one. The method says "Returns... updated Context".
        # Let's clone first to be safe.
        solved_ctx = ctx.clone()

        # Initial State
        wacc = 0.10 # Seed

        # Debt is constant in amount (simplified for this loop, assuming fixed capital structure)
        total_debt = sum(s.balance for s in solved_ctx.capital_structure.securities if s.security_type != "COMMON" and s.security_type != "PREFERRED")

        # Iterate
        for i in range(max_iter):
            # 1. Calculate EV
            dcf_res = DCFEngine.calculate_valuation(solved_ctx, wacc)
            ev = dcf_res["enterprise_value"]

            # 2. Leverage
            equity_value = max(0, ev - total_debt)
            debt_to_equity = total_debt / equity_value if equity_value > 0 else 99.0
            ltv = total_debt / ev if ev > 0 else 1.0

            # 3. PD
            pd = CreditEngine.calculate_pd(solved_ctx, ev)

            # 4. Rating & Spread
            # Need FCCR.
            # Interest depends on Rate. Rate depends on Spread. Spread depends on Rating.
            # Inner loop? Or just update rate for next outer loop.
            # We'll update rate now based on PREVIOUS loop's rating?
            # Or assume Rate updates instantly.
            # Let's calculate Rate based on current PD/LTV and LAST loop's Interest (for FCCR).
            # Actually, Rating depends on FCCR. FCCR depends on Rate.
            # This is the "Broken Circle" (8.2).
            # We calculate Rating using *current* Interest assumptions.

            # Current Interest
            current_interest = sum(s.balance * s.interest_rate for s in solved_ctx.capital_structure.securities)
            mandatory_principal = sum(s.balance * 0.05 for s in solved_ctx.capital_structure.securities if s.security_type == "TERM_LOAN")
            ebitda = dcf_res["projections"]["ebitda"][0] # Year 1 EBITDA

            fccr = ebitda / (current_interest + mandatory_principal) if (current_interest + mandatory_principal) > 0 else 99.0

            rating, desc = RegulatoryEngine.get_rating_from_metrics(pd, fccr, ltv)

            # 5. Update Cost of Debt
            # Base Rate + Spread
            # Assume spread logic: Base Spread + Rating Adjustment
            base_sofr = solved_ctx.credit_challenge.sofr_base_rate or 0.04

            # Simple spread mapping (Mock)
            base_spread = 0.03 # 300 bps
            rating_spread_adj = 0.0
            if rating >= 6: rating_spread_adj = 0.04 # +400 bps

            new_kd_pretax = base_sofr + base_spread + rating_spread_adj
            solved_ctx.valuation_context.pre_tax_cost_of_debt = new_kd_pretax

            # Update Security Interest Rates for next loop (Consistency)
            for sec in solved_ctx.capital_structure.securities:
                if sec.sofr_spread is not None:
                     sec.interest_rate = base_sofr + sec.sofr_spread + rating_spread_adj

            # 6. Update Cost of Equity (Hamada)
            # Beta_L = Beta_U * (1 + (1-t)*D/E)
            tax_rate = solved_ctx.valuation_context.tax_rate
            beta_u = ctx.valuation_context.beta # Assume input is Unlevered/Asset Beta
            beta_l = beta_u * (1 + (1 - tax_rate) * debt_to_equity)
            solved_ctx.valuation_context.beta = beta_l # Update context for WACC calc

            # 7. Calculate New WACC
            new_wacc = WACCCalculator.calculate_wacc(solved_ctx.valuation_context, equity_value, total_debt)

            # Convergence Check
            diff = abs(new_wacc - wacc)
            if diff < tolerance:
                # Converged
                # Return final calc
                dcf_final = DCFEngine.calculate_valuation(solved_ctx, new_wacc)
                return {
                    "context": solved_ctx,
                    "valuation": dcf_final,
                    "metrics": {
                        "rating": rating,
                        "pd": pd,
                        "leverage": ltv,
                        "wacc": new_wacc,
                        "iterations": i + 1
                    }
                }

            wacc = new_wacc

        # If not converged
        return {
             "context": solved_ctx,
             "valuation": DCFEngine.calculate_valuation(solved_ctx, wacc),
             "metrics": {
                 "rating": rating, # Last state
                 "pd": pd,
                 "leverage": ltv,
                 "wacc": wacc,
                 "iterations": max_iter,
                 "converged": False
             }
        }
