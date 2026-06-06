import numpy as np

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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
