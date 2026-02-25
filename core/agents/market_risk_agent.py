from __future__ import annotations
from typing import Any, Dict, Optional, List
import logging
import numpy as np
from core.agents.agent_base import AgentBase

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MarketRiskAgent(AgentBase):
    """
    Agent responsible for assessing Market Risk (Systematic Risk).
    Calculates Volatility (Standard & EWMA), Beta, Value at Risk (VaR),
    Expected Shortfall (CVaR), and performs Stress Testing.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.confidence_level = self.config.get("confidence_level", 0.95)
        self.decay_factor = self.config.get("ewma_decay", 0.94)

    async def execute(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the market risk assessment.

        Args:
            market_data: Dictionary containing:
                - 'price_history': list of floats (prices)
                - 'benchmark_history': list of floats (optional)
                - 'portfolio_value': float (optional, for monetary VaR)

        Returns:
            Dict containing risk metrics.
        """
        logger.info("Starting Market Risk Assessment...")

        prices = market_data.get("price_history", [])
        if not prices or len(prices) < 2:
             return {"error": "Insufficient price history for analysis"}

        try:
            prices = np.array(prices, dtype=float)
            # Calculate daily returns: (Price_t - Price_t-1) / Price_t-1
            returns = np.diff(prices) / prices[:-1]

            # 1. Volatility (Annualized) - Standard Deviation
            daily_vol = np.std(returns)
            annual_vol = daily_vol * np.sqrt(252)

            # 2. EWMA Volatility (More weight to recent data)
            ewma_vol = self._calculate_ewma_volatility(returns)

            # 3. Value at Risk (VaR)
            var_95_pct = self._calculate_var(returns)

            # 4. Expected Shortfall (CVaR)
            cvar_95_pct = self._calculate_cvar(returns)

            # 5. Beta (if benchmark provided)
            beta = None
            benchmark_prices = market_data.get("benchmark_history", [])

            # Simple length check - in production we'd align dates
            if benchmark_prices and len(benchmark_prices) == len(prices):
                 benchmark_prices = np.array(benchmark_prices, dtype=float)
                 benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
                 beta = self._calculate_beta(returns, benchmark_returns)

            # 6. Stress Testing
            stress_impacts = self._run_stress_test(returns)

            # 7. Additional Metrics (Sortino, Max Drawdown)
            sortino = self._calculate_sortino_ratio(returns)
            max_dd = self._calculate_max_drawdown(prices)

            # Format Monetary Values if portfolio value is provided
            portfolio_value = market_data.get("portfolio_value")
            monetary_metrics = {}
            if portfolio_value:
                monetary_metrics = {
                    "VaR_95_Amount": var_95_pct * portfolio_value,
                    "CVaR_95_Amount": cvar_95_pct * portfolio_value
                }

            result = {
                "volatility_annualized": float(annual_vol),
                "volatility_ewma_annualized": float(ewma_vol),
                "var_95_pct": float(var_95_pct),
                "cvar_95_pct": float(cvar_95_pct),
                "beta": float(beta) if beta is not None else "N/A",
                "sortino_ratio": float(sortino),
                "max_drawdown": float(max_dd),
                "stress_tests_impact_pct": stress_impacts,
                "monetary_risk": monetary_metrics,
                "risk_level": "High" if annual_vol > 0.3 else "Medium" if annual_vol > 0.15 else "Low"
            }

            logger.info(f"Market Risk Assessment Complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Error calculating market risk: {e}")
            return {"error": str(e)}

    def _calculate_ewma_volatility(self, returns: np.ndarray) -> float:
        """Calculates Exponentially Weighted Moving Average (EWMA) Volatility."""
        if len(returns) == 0:
            return 0.0

        # We can implement a simple recursive calculation or a weighted sum.
        # For efficiency on short arrays, weighted sum is fine.
        # Weights: (1-lambda) * lambda^i for i=0 to T-1 (where i=0 is most recent)

        T = len(returns)
        # Reverse returns so index 0 is most recent
        r_reversed = returns[::-1]

        weights = np.array([(1 - self.decay_factor) * (self.decay_factor ** i) for i in range(T)])

        # Normalize weights to sum to 1 (handling finite window truncation)
        weights = weights / weights.sum()

        weighted_sq_returns = np.sum(weights * (r_reversed ** 2))
        daily_ewma = np.sqrt(weighted_sq_returns)

        return float(daily_ewma * np.sqrt(252))

    def _calculate_var(self, returns: np.ndarray) -> float:
        """Calculates Value at Risk at the configured confidence level (returns positive loss %)."""
        if SCIPY_AVAILABLE:
            mean = np.mean(returns)
            std = np.std(returns)
            quantile = 1.0 - self.confidence_level
            cutoff = norm.ppf(quantile, mean, std)
            return float(-cutoff)
        else:
             # Historical simulation
             quantile = 1.0 - self.confidence_level
             return float(-np.percentile(returns, quantile * 100))

    def _calculate_cvar(self, returns: np.ndarray) -> float:
        """Calculates Conditional Value at Risk (Expected Shortfall)."""
        quantile = 1.0 - self.confidence_level
        if SCIPY_AVAILABLE:
             # Parametric CVaR under Normal assumption
             mean = np.mean(returns)
             std = np.std(returns)
             # ES = - (mu - sigma * (pdf(norm.ppf(alpha)) / alpha))
             # where alpha is the significance level (e.g., 0.05)

             z_score = norm.ppf(quantile)
             pdf_val = norm.pdf(z_score)

             # Formula for Expected Shortfall for normal distribution:
             # ES = -mean + sigma * pdf(z) / alpha
             es = -mean + std * (pdf_val / quantile)
             return float(es)
        else:
             # Historical CVaR
             cutoff = np.percentile(returns, quantile * 100)
             # Filter returns worse than cutoff
             tail_losses = returns[returns <= cutoff]
             if len(tail_losses) == 0:
                 return float(-cutoff)
             return float(-np.mean(tail_losses))

    def _calculate_beta(self, asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculates Beta relative to a benchmark."""
        if len(asset_returns) != len(benchmark_returns):
            return 0.0

        cov_matrix = np.cov(asset_returns, benchmark_returns)
        covariance = cov_matrix[0][1]
        variance = cov_matrix[1][1]

        if variance == 0:
            return 0.0
        return covariance / variance

    def _run_stress_test(self, returns: np.ndarray) -> Dict[str, float]:
        """Simulates various stress scenarios based on historical or hypothetical shocks."""
        current_vol = np.std(returns)

        # 3-Sigma Event
        sigma_3_shock = float(np.mean(returns) - 3 * current_vol)

        # 6-Sigma Event (Black Swan)
        sigma_6_shock = float(np.mean(returns) - 6 * current_vol)

        return {
            "historical_2008_crash": 0.40, # 40% loss
            "historical_covid_crash": 0.30, # 30% loss
            "hypothetical_mild_recession": 0.15,
            "statistical_3_sigma_shock": abs(sigma_3_shock),
            "statistical_6_sigma_shock": abs(sigma_6_shock),
            "correlation_breakdown_shock": 0.25 # Assumption: diversification fails
        }

    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculates Sortino Ratio (Return / Downside Deviation)."""
        mean_return = np.mean(returns)

        # Downside deviation: only consider returns below target
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            return 0.0 # No downside risk?

        downside_deviation = np.std(downside_returns)

        if downside_deviation == 0:
            return 0.0

        # Annualized Sortino (assuming daily returns)
        sortino = (mean_return - target_return) / downside_deviation
        return float(sortino * np.sqrt(252))

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculates Maximum Drawdown from peak."""
        if len(prices) < 2:
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdown for each point
        drawdowns = (prices - running_max) / running_max

        # Max drawdown is the minimum (most negative) value
        max_dd = np.min(drawdowns)

        return float(abs(max_dd))
