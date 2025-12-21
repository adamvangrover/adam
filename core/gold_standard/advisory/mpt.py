"""
Modern Portfolio Theory (MPT) Module.
Implements Mean-Variance Optimization and Risk Metrics.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
except ImportError:
    EfficientFrontier = None
    risk_models = None
    expected_returns = None

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, prices: pd.DataFrame):
        """
        Initialize with a DataFrame of historical closing prices (Date x Tickers).
        """
        self.prices = prices
        if EfficientFrontier is None:
            logger.warning("PyPortfolioOpt not installed. Optimization will fail.")

    def optimize_max_sharpe(self) -> Dict[str, float]:
        """
        Optimizes for Maximum Sharpe Ratio.
        """
        if EfficientFrontier is None: return {}

        # Calculate Expected Returns and Covariance
        mu = expected_returns.mean_historical_return(self.prices)
        S = risk_models.sample_cov(self.prices)

        # Optimize
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        return cleaned_weights

    def calculate_risk_metrics(self, weights: Dict[str, float], confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculates Historical VaR and CVaR for the portfolio.
        """
        # Convert weights to array
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])

        # Filter prices for these tickers
        df = self.prices[tickers].pct_change().dropna()

        # Calculate portfolio daily returns
        portfolio_returns = df.dot(w)

        # Historical Simulation
        var_cutoff = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        cvar = portfolio_returns[portfolio_returns <= var_cutoff].mean()

        return {
            f"VaR_{int(confidence_level*100)}": abs(var_cutoff),
            f"CVaR_{int(confidence_level*100)}": abs(cvar),
            "Annual_Vol": portfolio_returns.std() * np.sqrt(252),
            "Annual_Return": portfolio_returns.mean() * 252
        }
