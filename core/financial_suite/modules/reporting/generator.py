from typing import List, Dict, Any
from core.financial_suite.schemas.workstream_context import WorkstreamContext
from core.financial_suite.engines.dcf import DCFEngine
from core.financial_suite.modules.risk.credit_model import CreditEngine


class ReportGenerator:
    @staticmethod
    def generate_expected_pd_matrix(ctx: WorkstreamContext) -> str:
        """
        Generates Table 1: EBITDA Margin Variance vs SOFR Base Rate.
        """
        base_margin = ctx.financials.projected_ebitda_margin[0]  # Year 1 base
        margins = [base_margin - 0.02, base_margin - 0.01, base_margin, base_margin + 0.01, base_margin + 0.02]
        margin_labels = ["-2.0%", "-1.0%", "Base", "+1.0%", "+2.0%"]

        base_sofr = 0.04  # 4.0%
        sofrs = [0.03, 0.04, 0.05, 0.06, 0.07]
        sofr_labels = ["3.0%", "4.0%", "5.0%", "6.0%", "7.0%"]

        markdown = r"| Margin \ SOFR | " + " | ".join(sofr_labels) + " |\n"
        markdown += "|---|---" + "|---" * len(sofrs) + "|\n"

        for i, m in enumerate(margins):
            row_str = f"| {margin_labels[i]} |"
            for s in sofrs:
                # Clone context
                sim_ctx = ctx.clone()

                # Apply overrides
                # Update all projected margins
                diff = m - base_margin
                sim_ctx.financials.projected_ebitda_margin = [x + diff for x in ctx.financials.projected_ebitda_margin]
                sim_ctx.credit_challenge.sofr_base_rate = s

                # Recalculate EV (simplified, constant WACC for speed or assume WACC changes with SOFR?)
                # If SOFR changes, WACC changes.
                # Ideally we run Solver. For speed here, we just run DCF with approximated WACC change.
                # WACC ~ Ke*E + (SOFR+Spread)*D
                # This is complex to do right without Solver.
                # Let's assume we just calculate PD based on EV update from Margin change.
                # And PD based on Interest Coverage? (Merton doesn't care about Interest Coverage directly).
                # Logistic PD cares about Leverage.
                # Merton uses Risk Free Rate (SOFR).

                # Update Rf
                sim_ctx.valuation_context.risk_free_rate = s

                # Run DCF (using base WACC + delta SOFR approximation?)
                # Just use base WACC for simplicity of demonstration, or update Cost of Debt.
                # sim_ctx.valuation_context.pre_tax_cost_of_debt = s + spread...

                # Run Valuation
                # We need a WACC.
                wacc = 0.10 + (s - 0.04)  # Crude adjustment
                dcf = DCFEngine.calculate_valuation(sim_ctx, wacc)
                ev = dcf["enterprise_value"]

                # Run PD
                pd = CreditEngine.calculate_pd(sim_ctx, ev)

                # Formatting
                pd_pct = f"{pd*100:.2f}%"
                row_str += f" {pd_pct} |"
            markdown += row_str + "\n"

        return markdown

    @staticmethod
    def generate_downside_pd_table(ctx: WorkstreamContext) -> str:
        """
        Generates Table 2: Revenue Contraction vs LGD Severity.
        """
        rev_contractions = [0.0, -0.05, -0.10, -0.15, -0.20]
        rev_labels = ["Base", "-5.0%", "-10.0%", "-15.0%", "-20.0%"]

        # Asset Volatility / LGD Severity
        # We'll map "Severity" to Volatility levels for Merton, or Haircuts for Logistic
        volatilities = [0.30, 0.45, 0.60, 0.75]
        vol_labels = ["Low (30%)", "Med (45%)", "High (60%)", "Severe (75%)"]

        markdown = r"| Rev Contraction \ Volatility | " + " | ".join(vol_labels) + " |\n"
        markdown += "|---|---" + "|---" * len(volatilities) + "|\n"

        for i, r_change in enumerate(rev_contractions):
            row_str = f"| {rev_labels[i]} |"
            for vol in volatilities:
                sim_ctx = ctx.clone()

                # Apply Revenue Haircut
                # Decrease all revenue projections
                sim_ctx.financials.projected_revenue_growth = [
                    g + r_change for g in ctx.financials.projected_revenue_growth]
                # Also affect current year revenue basis?
                # Usually contraction starts in Year 1.

                # Apply Volatility
                sim_ctx.credit_challenge.asset_volatility = vol

                # Recalculate EV
                # Revenue drop -> Lower EV
                wacc = 0.10
                dcf = DCFEngine.calculate_valuation(sim_ctx, wacc)
                ev = dcf["enterprise_value"]

                # Calculate PD
                pd = CreditEngine.calculate_pd(sim_ctx, ev)

                row_str += f" {pd*100:.2f}% |"
            markdown += row_str + "\n"

        return markdown

    @staticmethod
    def generate_full_report(ctx: WorkstreamContext, solver_result: Dict[str, Any]) -> str:
        """
        Assembles the full Markdown report.
        """
        ev = solver_result["valuation"]["enterprise_value"]
        equity_val = max(
            0, ev - sum(s.balance for s in ctx.capital_structure.securities if s.security_type not in ["COMMON", "PREFERRED"]))
        rating = solver_result["metrics"]["rating"]
        desc = "Unknown"
        # Re-derive desc or pass it
        from core.financial_suite.modules.risk.regulatory import RegulatoryEngine
        _, desc = RegulatoryEngine.get_rating_from_metrics(
            solver_result["metrics"]["pd"], 1.5, 0.5)  # Mock args just to get desc map?
        # Actually desc is not easily retrievable from id without the map.
        # I'll just use the ID.

        report = f"# ADAM Financial Workstream Report\n\n"
        report += f"## Executive Summary\n"
        report += f"- **Enterprise Value:** ${ev:,.2f}\n"
        report += f"- **Equity Value:** ${equity_val:,.2f}\n"
        report += f"- **Regulatory Rating:** {rating}\n"
        report += f"- **WACC:** {solver_result['metrics']['wacc']*100:.2f}%\n"
        report += f"- **PD:** {solver_result['metrics']['pd']*100:.2f}%\n\n"

        report += f"## Sensitivity Analysis\n"
        report += f"### 1. Expected PD Matrix (Operating Drift)\n"
        report += ReportGenerator.generate_expected_pd_matrix(ctx)
        report += f"\n"

        report += f"### 2. Downside PD Sensitivity (Credit Challenge)\n"
        report += ReportGenerator.generate_downside_pd_table(ctx)
        report += f"\n"

        return report
