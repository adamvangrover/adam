from typing import Dict, List
from core.financial_suite.schemas.workstream_context import WorkstreamContext, Security

class WaterfallEngine:
    @staticmethod
    def calculate_exit_waterfall(ctx: WorkstreamContext, exit_enterprise_value: float) -> Dict[str, float]:
        """
        Calculates the distribution of proceeds at exit.
        """
        remaining_proceeds = exit_enterprise_value
        distribution = {}

        # Sort securities by priority (1 is highest)
        sorted_securities = sorted(ctx.capital_structure.securities, key=lambda x: x.priority)

        # 1. Debt Paydown
        for sec in sorted_securities:
            if sec.security_type in ["REVOLVER", "TERM_LOAN", "MEZZANINE"]:
                # Assume full repayment of principal + PIK if tracked (simplified here as balance)
                # In a real engine, we'd roll forward the balance with interest.
                # Here we assume 'balance' is the claim at exit for simplicity or static model.
                # Ideally, the Context would have been updated by the Solver with T_exit balances.
                claim = sec.balance
                paid = min(remaining_proceeds, claim)
                distribution[sec.name] = paid
                remaining_proceeds -= paid

        # 2. Preferred Equity
        total_common_shares = sum(s.shares for s in sorted_securities if s.security_type == "COMMON")

        for sec in sorted_securities:
            if sec.security_type == "PREFERRED":
                # Liquidation Preference
                liq_claim = (sec.investment or 0) * (sec.liq_pref_multiple or 1.0)

                # Conversion Value
                # Need to know total shares if converted.
                # Simplified: Assume we check conversion against current common pool.
                # Value if converted = (Shares / (CommonShares + PrefShares)) * (Remaining + PrefClaims?)
                # This is complex. Simplified logic per Appendix A.2 pseudo-code.

                # Assuming this is the ONLY preferred for simplicity of the "check conversion" step.
                # If multiple preferreds, it gets recursive.

                # Hypothethical Common Value if converted:
                total_shares_if_converted = total_common_shares + (sec.shares or 0)
                pct_ownership = (sec.shares or 0) / total_shares_if_converted if total_shares_if_converted > 0 else 0

                # "As-Converted" value implies we ignore the Liq Pref and just take % of Equity Value
                # Equity Value = Remaining Proceeds (because Debt is paid)
                conversion_claim = remaining_proceeds * pct_ownership

                pay_pref = 0.0
                converted = False

                if sec.is_participating:
                    # Pay Liq Pref first, then participate
                    pay_liq = min(remaining_proceeds, liq_claim)
                    distribution[sec.name] = pay_liq
                    remaining_proceeds -= pay_liq

                    # Participate
                    # Recalculate % ownership of residual
                    # (Shares / Total Shares) * Residual
                    # Note: Usually participating preferreds exclude their liq pref from the residual calc base?
                    # Or they just take their % of the *remaining*.
                    part_claim = remaining_proceeds * pct_ownership
                    distribution[sec.name] += part_claim
                    remaining_proceeds -= part_claim

                else:
                    # Non-participating: Max(LiqPref, Conversion)
                    if conversion_claim > liq_claim:
                        # Convert
                        converted = True
                        pay_pref = conversion_claim
                        # Technically, if they convert, they are "Common".
                        # We subtract this from proceeds.
                        # But wait, if they convert, they dilute Common.
                        # So we don't subtract from proceeds yet, we just note they are Common now.
                    else:
                        # Don't convert
                        pay_pref = min(remaining_proceeds, liq_claim)
                        distribution[sec.name] = pay_pref
                        remaining_proceeds -= pay_pref

                if converted:
                    # If converted, they participate in the "Common" bucket logic below.
                    # We store their 'name' as a Common holder now?
                    # Or we handle it here.
                    # Simplified: We just pay them the conversion claim and reduce proceeds.
                    # This assumes they take their slice and leave the rest for original common.
                    distribution[sec.name] = conversion_claim
                    remaining_proceeds -= conversion_claim

        # 3. Common Equity
        # Distribute remaining to Common
        common_secs = [s for s in sorted_securities if s.security_type == "COMMON"]
        total_common_shares = sum(s.shares for s in common_secs)

        for sec in common_secs:
            if total_common_shares > 0:
                share = sec.shares / total_common_shares
                paid = remaining_proceeds * share
                distribution[sec.name] = paid
            else:
                distribution[sec.name] = 0.0

        return distribution
