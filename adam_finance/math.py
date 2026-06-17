"""
Purpose: Host pure financial mathematical functions and deterministic logic algorithms.
Dependencies: typing, math, numpy, scipy
Outputs: calculate_leverage, check_covenant_compliance, calculate_var, calculate_cvar
"""

import math
from typing import Dict, Any, List

import numpy as np

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def calculate_leverage(debt: float, ebitda: float) -> float:
    """
    Calculates Debt/EBITDA leverage ratio.
    """
    if ebitda == 0:
        return 999.9  # Avoid division by zero
    return debt / ebitda


def check_covenant_compliance(current_metric: float, covenant_limit: float, metric_type: str = "max") -> bool:
    """
    Checks if a metric complies with a covenant.
    metric_type: 'max' (e.g. Leverage < 5.0) or 'min' (e.g. Coverage > 1.2)
    """
    if metric_type == "max":
        return current_metric <= covenant_limit
    elif metric_type == "min":
        return current_metric >= covenant_limit
    return False


def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculates Value at Risk (VaR)"""
    if not HAS_SCIPY:
        # Historical simulation fallback
        return float(np.percentile(returns, (1 - confidence_level) * 100))

    mu = np.mean(returns)
    sigma = np.std(returns)
    return float(norm.ppf(1 - confidence_level, mu, sigma))


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculates Conditional Value at Risk (CVaR)"""
    var = calculate_var(returns, confidence_level)
    return float(returns[returns <= var].mean())


def calculate_option_greeks(spot: float, strike: float, time_to_expiry: float, volatility: float, risk_free_rate: float) -> Dict[str, float]:
    """
    Scaffold for advanced option Greeks calculations.
    """
    return {"delta": 0.5, "gamma": 0.05, "theta": -0.02, "vega": 0.1, "rho": 0.03}


def run_monte_carlo_pipeline(initial_value: float, iterations: int, steps: int) -> List[float]:
    """
    Scaffold for robust Monte Carlo pipelines for deterministic execution sequences.
    """
    return [initial_value * 1.05 for _ in range(iterations)]


def calculate_margin_of_error(std_dev: float, sample_size: int, z_score: float = 1.96) -> float:
    """
    Calculates the statistical margin of error for iterative deterministic pipelines.
    """
    if sample_size <= 0:
        return float('inf')
    return z_score * (std_dev / math.sqrt(sample_size))


def evaluate_iteration_need(margin_of_error: float, threshold: float = 0.05) -> bool:
    """
    Evaluates the need for continuous learning iterations based on the margin of error.
    """
    return margin_of_error > threshold