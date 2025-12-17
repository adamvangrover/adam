from typing import Dict, List, Any, Optional, Union
import math
import logging
import statistics

# -------------------------------------------------------------------------
# DEPENDENCY MANAGEMENT
# Progressive Enhancement: Load Scipy if available for high-precision stats
# -------------------------------------------------------------------------
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logger = logging.getLogger("RiskEngine_Unified")


class RiskEngine:
    """
    RiskEngine v3.0 (Unified)
    
    A comprehensive risk management engine capable of:
    1. Parametric VaR (Variance-Covariance & Simplified).
    2. Historical Simulation VaR & Expected Shortfall (CVaR).
    3. Option Greeks (Analytical Black-Scholes with Scipy/Math fallbacks).
    
    "Risk varies inversely with knowledge."
    """

    def __init__(self):
        self.scipy_enabled = SCIPY_AVAILABLE
        if not self.scipy_enabled:
            logger.warning("Scipy not found. Running in Analytical Math (Stand-alone) mode.")

    # -------------------------------------------------------------------------
    # 1. VALUE AT RISK (VaR) & CVaR
    # -------------------------------------------------------------------------

    def calculate_parametric_var(self, 
                                 portfolio: List[Dict[str, Any]], 
                                 confidence_level: float = 0.95, 
                                 correlation_matrix: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Calculates Parametric Value at Risk using Variance-Covariance Method.
        
        Logic:
        $$ \sigma_p^2 = w^T \cdot \Sigma \cdot w $$
        
        If correlation_matrix is None, it defaults to a 'Simplified' mode (Sum of VaRs),
        which assumes Correlation = 1.0 (The most conservative 'worst-case' correlation).
        """
        total_value = sum(p.get("market_value", 0.0) for p in portfolio)
        if total_value == 0:
            return {"VaR": 0.0, "TotalValue": 0.0}

        # Z-Score Selection (1-tailed)
        # 95% -> 1.645, 99% -> 2.326
        z_score = 1.645 if confidence_level <= 0.95 else 2.326
        time_scaling = math.sqrt(1/252)  # Annualized -> Daily

        # Extract vectors
        weights = [p.get("market_value", 0)/total_value for p in portfolio]
        vols = [p.get("volatility", 0.2) for p in portfolio]
        n = len(portfolio)

        # --- BRANCH: COVARIANCE METHOD (If matrix provided) ---
        if correlation_matrix and len(correlation_matrix) == n:
            portfolio_variance = 0.0
            for i in range(n):
                for j in range(n):
                    # Cov_ij = vol_i * vol_j * rho_ij
                    cov_ij = vols[i] * vols[j] * correlation_matrix[i][j]
                    portfolio_variance += (weights[i] * weights[j] * cov_ij)
            
            portfolio_std = math.sqrt(portfolio_variance)
            method_tag = "Variance-Covariance (Matrix)"
            
        # --- BRANCH: SIMPLIFIED / HEURISTIC (Fallback) ---
        else:
            # Fallback 1: Weighted sum of vols (assumes correlation = 1.0)
            # This effectively matches the logic of the simple engine 
            # but creates a comparable 'portfolio_std' metric.
            portfolio_std = sum(w * v for w, v in zip(weights, vols))
            method_tag = "Parametric (Sum of Parts / Corr=1.0)"

        daily_var = total_value * portfolio_std * z_score * time_scaling
        annual_var = total_value * portfolio_std * z_score

        return {
            "VaR_Daily": round(daily_var, 2),
            "VaR_Annual": round(annual_var, 2),
            "TotalValue": total_value,
            "Implied_Portfolio_Vol": round(portfolio_std, 4),
            "Confidence": confidence_level,
            "Method": method_tag
        }

    def calculate_historical_var(self, 
                                 portfolio_returns: List[float], 
                                 portfolio_value: float, 
                                 confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculates VaR and CVaR (Expected Shortfall) using Historical Simulation.
        Captures 'fat tails' that parametric normal distributions miss.
        """
        if not portfolio_returns:
            return {"Hist_VaR": 0.0, "CVaR": 0.0, "Note": "No historical data"}

        # Sort returns from worst (negative) to best
        sorted_returns = sorted(portfolio_returns)
        
        # Calculate index for cutoff
        # e.g., for 95%, we find the 5th percentile worst return
        cutoff_index = int((1 - confidence_level) * len(sorted_returns))
        cutoff_index = max(0, min(cutoff_index, len(sorted_returns) - 1))

        var_return = sorted_returns[cutoff_index]
        hist_var = abs(var_return * portfolio_value)

        # Calculate CVaR (Average of losses exceeding VaR)
        tail_losses = sorted_returns[:cutoff_index]
        if tail_losses:
            avg_tail_loss = sum(tail_losses) / len(tail_losses)
            cvar = abs(avg_tail_loss * portfolio_value)
        else:
            cvar = hist_var

        return {
            "Hist_VaR": round(hist_var, 2),
            "CVaR": round(cvar, 2),
            "Method": "Historical_Simulation",
            "DataPoints": len(portfolio_returns)
        }

    # -------------------------------------------------------------------------
    # 2. DERIVATIVES PRICING (GREEKS)
    # -------------------------------------------------------------------------

    def calculate_greeks(self, position: Dict[str, Any]) -> Dict[str, float]:
        """
        Routes calculation to the most precise engine available:
        1. Scipy Engine (Exact stats)
        2. Analytical Math Engine (Approximation via Erf)
        """
        S = float(position.get("spot", 100))
        K = float(position.get("strike", 100))
        T = float(position.get("time_to_expiry", 1.0))
        r = float(position.get("rate", 0.05))
        sigma = float(position.get("volatility", 0.2))
        opt_type = position.get("type", "call").lower()

        # Edge Case Safety
        if T <= 0:
            return self._get_expired_greeks(S, K, opt_type)
        if sigma <= 0.001:
            sigma = 0.001

        # Router
        if self.scipy_enabled:
            return self._greeks_scipy(S, K, T, r, sigma, opt_type)
        else:
            return self._greeks_analytical_math(S, K, T, r, sigma, opt_type)

    def _greeks_scipy(self, S, K, T, r, sigma, opt_type) -> Dict[str, float]:
        """High-precision Scipy implementation."""
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        if opt_type == "call":
            delta = cdf_d1
            theta = (- (S * sigma * pdf_d1) / (2 * sqrt_T) - r * K * math.exp(-r * T) * cdf_d2)
            rho = K * T * math.exp(-r * T) * cdf_d2
        else: # Put
            delta = cdf_d1 - 1
            theta = (- (S * sigma * pdf_d1) / (2 * sqrt_T) + r * K * math.exp(-r * T) * norm.cdf(-d2))
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * sqrt_T * pdf_d1 * 0.01

        return {
            "delta": round(delta, 4), "gamma": round(gamma, 4),
            "theta": round(theta / 365, 4), "vega": round(vega, 4),
            "rho": round(rho * 0.01, 4), "method": "BSM_Scipy"
        }

    def _greeks_analytical_math(self, S, K, T, r, sigma, opt_type) -> Dict[str, float]:
        """
        Dependency-free analytical implementation using Error Function (erf).
        Used when Scipy is unavailable.
        """
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Custom PDF/CDF
        pdf_d1 = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1**2)
        cdf_d1 = self._norm_cdf(d1)
        cdf_d2 = self._norm_cdf(d2)

        if opt_type == "call":
            delta = cdf_d1
            # theta calculation (annual)
            theta_part1 = -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            theta_part2 = -r * K * math.exp(-r * T) * cdf_d2
            theta = theta_part1 + theta_part2
            rho = K * T * math.exp(-r * T) * cdf_d2
        else: # Put
            delta = cdf_d1 - 1.0
            cdf_neg_d2 = self._norm_cdf(-d2)
            theta_part1 = -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            theta_part2 = r * K * math.exp(-r * T) * cdf_neg_d2
            theta = theta_part1 + theta_part2
            rho = -K * T * math.exp(-r * T) * cdf_neg_d2

        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * pdf_d1 * sqrt_T * 0.01

        return {
            "delta": round(delta, 4), "gamma": round(gamma, 4),
            "theta": round(theta / 365.0, 4), "vega": round(vega, 4),
            "rho": round(rho * 0.01, 4), "method": "BSM_Analytical_Math"
        }

    def _norm_cdf(self, x: float) -> float:
        """Standard Normal CDF approximation using math.erf"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _get_expired_greeks(self, S, K, opt_type):
        """Helper for expired options"""
        is_itm = (S > K) if opt_type == "call" else (S < K)
        return {
            "delta": 1.0 if is_itm else 0.0,
            "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0,
            "method": "Expired"
        }
