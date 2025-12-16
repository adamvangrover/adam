# core/product/risk_engine/advanced_models.py
import numpy as np
from typing import Dict, Any, List

class HistoricalRiskEngine:
    """
    Risk calculations based on historical simulation.
    Uses actual historical returns to estimate VaR and ES.
    """
    def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculates Value at Risk (VaR) using historical method.
        """
        if not returns:
            return 0.0
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        # Ensure index is within bounds
        index = max(0, min(index, len(sorted_returns) - 1))
        return abs(sorted_returns[index])

    def calculate_es(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculates Expected Shortfall (CVaR).
        """
        if not returns:
            return 0.0
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))
        tail_losses = sorted_returns[:index+1]
        if not tail_losses:
            return 0.0
        return abs(sum(tail_losses) / len(tail_losses))

class JumpDiffusionEngine:
    """
    Merton Jump Diffusion Model for Asset Pricing and Risk.
    dS_t = (mu - lambda*k)S_t dt + sigma S_t dW_t + (Y-1)S_t dN_t
    """
    def simulate_paths(self, S0: float, T: float, r: float, sigma: float, lam: float, mu_j: float, sigma_j: float, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulates asset price paths using Monte Carlo Geometric Brownian Motion with Jumps.

        Args:
            S0: Initial stock price
            T: Time horizon (years)
            r: Risk-free rate
            sigma: Volatility (diffusion)
            lam: Jump intensity (expected jumps per year)
            mu_j: Mean of log jump size
            sigma_j: Std dev of log jump size
            n_steps: Number of time steps
            n_paths: Number of simulation paths
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        # Drift correction for jump mean k = E[Y-1] = exp(mu_j + 0.5*sigma_j^2) - 1
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        drift = (r - 0.5 * sigma**2 - lam * k) * dt
        vol = sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            # Brownian Motion component
            Z = np.random.normal(0, 1, n_paths)

            # Poisson Jump Process component
            # N is number of jumps in this step dt
            N = np.random.poisson(lam * dt, n_paths)

            # Jump Magnitude
            # If N jumps occur, the log-multiplier is sum of N normals ~ N(N*mu_j, N*sigma_j^2)
            # We handle the case where N=0 carefully (it results in 0 jump return)

            jump_log_return = np.zeros(n_paths)
            mask = N > 0
            if np.any(mask):
                n_jumps = N[mask]
                jump_log_return[mask] = n_jumps * mu_j + np.sqrt(n_jumps) * sigma_j * np.random.normal(0, 1, np.sum(mask))

            paths[:, t] = paths[:, t-1] * np.exp(drift + vol * Z + jump_log_return)

        return paths
