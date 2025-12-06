from core.financial_suite.schemas.workstream_context import ValuationContext

class WACCCalculator:
    @staticmethod
    def calculate_cost_of_equity(ctx: ValuationContext) -> float:
        """
        Calculates Cost of Equity (Ke) based on the configured method.
        """
        if ctx.wacc_method == "CAPM_STANDARD":
            # Ke = Rf + Beta * (Rm - Rf)
            return ctx.risk_free_rate + ctx.beta * (ctx.market_return - ctx.risk_free_rate)

        elif ctx.wacc_method == "CAPM_SIZE_ADJUSTED":
            # Ke = Rf + Beta * (Rm - Rf) + Size Premium
            base_capm = ctx.risk_free_rate + ctx.beta * (ctx.market_return - ctx.risk_free_rate)
            return base_capm + (ctx.size_premium or 0.0)

        elif ctx.wacc_method == "BUILD_UP":
            # Ke = Rf + Equity Risk Premium + Size Premium + Specific Risk
            # Note: In Build-Up, Beta is effectively 1.0 or ignored in favor of direct premiums
            equity_risk_premium = ctx.market_return - ctx.risk_free_rate
            return (ctx.risk_free_rate +
                    equity_risk_premium +
                    (ctx.size_premium or 0.0) +
                    (ctx.specific_risk_premium or 0.0))

        else:
            raise ValueError(f"Unknown WACC method: {ctx.wacc_method}")

    @staticmethod
    def calculate_cost_of_debt(ctx: ValuationContext, tax_rate_override: float = None) -> float:
        """
        Calculates After-Tax Cost of Debt (Kd).
        """
        tax_rate = tax_rate_override if tax_rate_override is not None else ctx.tax_rate
        return ctx.pre_tax_cost_of_debt * (1 - tax_rate)

    @staticmethod
    def calculate_wacc(ctx: ValuationContext, equity_value: float, debt_value: float) -> float:
        """
        Calculates WACC given the capital structure weights.
        """
        total_capital = equity_value + debt_value
        if total_capital == 0:
            return 0.0

        weight_equity = equity_value / total_capital
        weight_debt = debt_value / total_capital

        ke = WACCCalculator.calculate_cost_of_equity(ctx)
        kd = WACCCalculator.calculate_cost_of_debt(ctx)

        return (weight_equity * ke) + (weight_debt * kd)
