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
    Calculates Volatility, Beta, and Value at Risk (VaR).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.confidence_level = self.config.get("confidence_level", 0.95)

    async def execute(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the market risk assessment.

        Args:
            market_data: Dictionary containing 'price_history' (list of floats)
                         and optional 'benchmark_history' (list of floats).

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

            # Volatility (Annualized)
            daily_vol = np.std(returns)
            annual_vol = daily_vol * np.sqrt(252)

            # VaR
            var_95 = self._calculate_var(returns)

            # Beta (if benchmark provided)
            beta = None
            benchmark_prices = market_data.get("benchmark_history", [])

            # Simple length check - in production we'd align dates
            if benchmark_prices and len(benchmark_prices) == len(prices):
                 benchmark_prices = np.array(benchmark_prices, dtype=float)
                 benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
                 beta = self._calculate_beta(returns, benchmark_returns)

            result = {
                "volatility_annualized": float(annual_vol),
                "var_95": float(var_95),
                "beta": float(beta) if beta is not None else "N/A",
                "risk_level": "High" if annual_vol > 0.3 else "Medium" if annual_vol > 0.15 else "Low"
            }

            logger.info(f"Market Risk Assessment Complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Error calculating market risk: {e}")
            return {"error": str(e)}

    def _calculate_var(self, returns: np.ndarray) -> float:
        """Calculates Value at Risk at the configured confidence level."""
        if SCIPY_AVAILABLE:
            mean = np.mean(returns)
            std = np.std(returns)
            # VaR is the loss cutoff.
            # norm.ppf(0.05) gives the z-score for the bottom 5%
            # If confidence is 0.95, we look at the 0.05 quantile.
            quantile = 1.0 - self.confidence_level
            cutoff = norm.ppf(quantile, mean, std)

            # VaR is usually expressed as a positive number representing loss
            return float(-cutoff)
        else:
             # Historical simulation
             quantile = 1.0 - self.confidence_level
             return float(-np.percentile(returns, quantile * 100))

    def _calculate_beta(self, asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculates Beta relative to a benchmark."""
        if len(asset_returns) != len(benchmark_returns):
            return 0.0 # Should be handled by caller, but safety check

        # Covariance matrix: [[Var(asset), Cov(asset, bench)], [Cov(bench, asset), Var(bench)]]
        cov_matrix = np.cov(asset_returns, benchmark_returns)
        covariance = cov_matrix[0][1]
        variance = cov_matrix[1][1]

        if variance == 0:
            return 0.0
        return covariance / variance
