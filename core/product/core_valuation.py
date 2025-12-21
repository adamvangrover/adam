# core/product/core_valuation.py

"""
Financial Engineering Engine (Product Layer).

This module contains deterministic, audit-ready financial calculations.
It serves as the 'Iron Core' for the Neuro-Symbolic architecture.
In production, these functions bind to the Rust backend for performance.
"""

from typing import List, Dict, Optional, Union
import math


class FinancialEngineeringEngine:

    @staticmethod
    def calculate_wacc(
        market_cap: float,
        total_debt: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float
    ) -> float:
        """
        Calculates Weighted Average Cost of Capital (WACC).
        """
        total_value = market_cap + total_debt
        if total_value == 0:
            return 0.0

        weight_equity = market_cap / total_value
        weight_debt = total_debt / total_value

        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        return wacc

    @staticmethod
    def calculate_dcf(
        free_cash_flows: List[float],
        discount_rate: float,
        terminal_value: float
    ) -> float:
        """
        Calculates Discounted Cash Flow (DCF).
        """
        pv = 0.0
        for i, cash_flow in enumerate(free_cash_flows):
            period = i + 1
            pv += cash_flow / ((1 + discount_rate) ** period)

        # Discount Terminal Value
        pv_terminal = terminal_value / ((1 + discount_rate) ** len(free_cash_flows))

        return pv + pv_terminal

    @staticmethod
    def calculate_terminal_value_growth(
        final_fcf: float,
        growth_rate: float,
        discount_rate: float
    ) -> float:
        """
        Calculates Terminal Value using the Gordon Growth Model.
        """
        if discount_rate <= growth_rate:
            raise ValueError("Discount rate must be higher than growth rate for Gordon Growth Model.")

        return (final_fcf * (1 + growth_rate)) / (discount_rate - growth_rate)

    @staticmethod
    def calculate_terminal_value_multiple(
        final_metric: float,
        exit_multiple: float
    ) -> float:
        """
        Calculates Terminal Value using the Exit Multiple method.
        """
        return final_metric * exit_multiple

    @staticmethod
    def calculate_capm(
        risk_free_rate: float,
        beta: float,
        equity_risk_premium: float
    ) -> float:
        """
        Calculates Cost of Equity using CAPM.
        """
        return risk_free_rate + (beta * equity_risk_premium)

    @staticmethod
    def calculate_greeks(
        spot_price: float,
        strike_price: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = "call"
    ) -> Dict[str, float]:
        """
        Calculates Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho).
        """
        if time_to_maturity <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

        d1 = (math.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2)
              * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
        d2 = d1 - volatility * math.sqrt(time_to_maturity)

        def norm_pdf(x): return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)
        def norm_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        if option_type == "call":
            delta = norm_cdf(d1)
            rho = strike_price * time_to_maturity * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2) / 100
            theta = (- (spot_price * norm_pdf(d1) * volatility) / (2 * math.sqrt(time_to_maturity)) -
                     risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2)) / 365
        else:
            delta = norm_cdf(d1) - 1
            rho = -strike_price * time_to_maturity * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(-d2) / 100
            theta = (- (spot_price * norm_pdf(d1) * volatility) / (2 * math.sqrt(time_to_maturity)) +
                     risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm_cdf(-d2)) / 365

        gamma = norm_pdf(d1) / (spot_price * volatility * math.sqrt(time_to_maturity))
        vega = spot_price * math.sqrt(time_to_maturity) * norm_pdf(d1) / 100

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float) -> float:
        """
        Calculates Sharpe Ratio.
        """
        if not returns or len(returns) < 2:
            return 0.0

        avg_return = sum(returns) / len(returns)
        variance = sum([(x - avg_return) ** 2 for x in returns]) / (len(returns) - 1)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        return (avg_return - risk_free_rate) / std_dev

    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float, target_return: float = 0.0) -> float:
        """
        Calculates Sortino Ratio using downside deviation.
        """
        if not returns or len(returns) < 2:
            return 0.0

        avg_return = sum(returns) / len(returns)

        downside_returns = [min(0, x - target_return) for x in returns]
        downside_variance = sum([x ** 2 for x in downside_returns]) / (len(returns) - 1)
        downside_dev = math.sqrt(downside_variance)

        if downside_dev == 0:
            return 0.0

        return (avg_return - risk_free_rate) / downside_dev

    @staticmethod
    def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float:
        """
        Calculates Beta of an asset relative to the market.
        Beta = Covariance(Asset, Market) / Variance(Market)
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 0.0

        n = len(asset_returns)
        avg_asset = sum(asset_returns) / n
        avg_market = sum(market_returns) / n

        covariance = sum([(asset_returns[i] - avg_asset) * (market_returns[i] - avg_market)
                         for i in range(n)]) / (n - 1)
        market_variance = sum([(x - avg_market) ** 2 for x in market_returns]) / (n - 1)

        if market_variance == 0:
            return 0.0

        return covariance / market_variance
