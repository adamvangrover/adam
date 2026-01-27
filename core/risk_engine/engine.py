from typing import Dict, List, Any, Optional, Union
import math
import logging
import statistics

# -------------------------------------------------------------------------
# DEPENDENCY MANAGEMENT
# Progressive Enhancement: Load Scipy/Numpy if available
# -------------------------------------------------------------------------
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

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
        self.numpy_enabled = NUMPY_AVAILABLE

        if not self.scipy_enabled:
            logger.warning("Scipy not found. Running in Analytical Math (Stand-alone) mode.")

        if self.numpy_enabled:
            logger.info("Numpy detected. Matrix-based Risk Attribution and Monte Carlo enabled.")
        else:
            logger.warning("Numpy not found. Falling back to scalar approximation.")

    # -------------------------------------------------------------------------
    # 1. VALUE AT RISK (VaR) & CVaR
    # -------------------------------------------------------------------------

    def calculate_portfolio_risk(self, **kwargs) -> Dict[str, Any]:
        """
        Alias for calculate_parametric_var to support MCP Registry interface.
        """
        return self.calculate_parametric_var(
            portfolio=kwargs.get("portfolio", []),
            confidence_level=kwargs.get("confidence_level", 0.95),
            correlation_matrix=kwargs.get("correlation_matrix", None)
        )

    def calculate_parametric_var(self,
                                 portfolio: List[Dict[str, Any]],
                                 confidence_level: float = 0.95,
                                 correlation_matrix: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        r"""
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
        z_score = self._get_z_score(confidence_level)
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

    def calculate_component_var(self,
                                portfolio: List[Dict[str, Any]],
                                confidence_level: float = 0.95,
                                correlation_matrix: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Decomposes Portfolio VaR into individual asset contributions (Risk Attribution).

        Math:
            Marginal VaR (MVaR) = d(VaR)/d(weight)
            Component VaR (CVaR) = Weight * MVaR

            Euler's Theorem: Sum(Component VaR) = Portfolio VaR (for homogeneous functions)

        Requires:
            Numpy for matrix operations.
            If Numpy is missing, falls back to a standalone approximation (Correlation = 1.0).
        """
        if not portfolio:
            return {"error": "Empty portfolio"}

        # Z-Score
        z_score = self._get_z_score(confidence_level)
        time_scaling = math.sqrt(1/252) # Annualized -> Daily

        # Extract basic vectors
        total_value = sum(p.get("market_value", 0.0) for p in portfolio)
        weights = [p.get("market_value", 0.0) / total_value for p in portfolio]
        vols = [p.get("volatility", 0.2) for p in portfolio]
        asset_ids = [p.get("id", f"Asset_{i}") for i, p in enumerate(portfolio)]

        # --- NUMPY PATH (MATRIX ALGEBRA) ---
        if self.numpy_enabled and correlation_matrix:
            try:
                w = np.array(weights)
                sigma = np.array(vols)
                corr = np.array(correlation_matrix)

                # Covariance Matrix: Sigma_ij = rho_ij * vol_i * vol_j
                # Efficient calculation: D * C * D where D is diagonal of vols
                D = np.diag(sigma)
                cov_matrix = D @ corr @ D

                # Portfolio Variance = w.T * Cov * w
                port_var = w.T @ cov_matrix @ w
                port_vol = np.sqrt(port_var)

                if port_vol == 0:
                    return {id: 0.0 for id in asset_ids}

                # Marginal VaR Vector = (Cov * w) * (Z / port_vol)
                # Explanation: The derivative of sigma_p w.r.t w is (Cov * w) / sigma_p
                marginal_vals = (cov_matrix @ w) / port_vol
                marginal_var = marginal_vals * z_score * time_scaling

                # Component VaR = Marginal VaR * Weight * Portfolio Value
                # (We scale by value to get dollar terms)
                component_var_dollars = marginal_var * w * total_value

                return {
                    "contributions": {
                        aid: round(float(cv), 2) for aid, cv in zip(asset_ids, component_var_dollars)
                    },
                    "method": "Matrix_Attribution"
                }
            except Exception as e:
                logger.error(f"Numpy matrix calculation failed: {e}. Falling back.")

        # --- FALLBACK (SCALAR APPROXIMATION) ---
        # Assumes Correlation = 1.0 (Worst Case)
        # In this case, Component VaR is just the standalone VaR of the position.
        results = {}
        for i, p in enumerate(portfolio):
            val = p.get("market_value", 0.0)
            vol = p.get("volatility", 0.2)
            # VaR = Value * Vol * Z * time_scaling
            c_var = val * vol * z_score * time_scaling
            results[asset_ids[i]] = round(c_var, 2)

        return {
            "contributions": results,
            "method": "Scalar_Approximation (Corr=1.0)",
            "note": "Install Numpy and provide correlation matrix for true diversification benefits."
        }

    def simulate_monte_carlo_var(self,
                                 portfolio: List[Dict[str, Any]],
                                 simulations: int = 10000,
                                 confidence_level: float = 0.95,
                                 correlation_matrix: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Executes a Multi-Asset Monte Carlo Simulation.

        Technique:
            Uses Cholesky Decomposition to generate correlated random paths.
            Simulates portfolio P&L distribution.

        Args:
            simulations: Number of random walks to generate (default 10,000).

        Returns:
            Dict with VaR, CVaR, and simulation metadata.
        """
        if not self.numpy_enabled:
             return {
                 "error": "Monte Carlo requires Numpy",
                 "VaR": 0.0,
                 "Note": "Please install numpy to unlock this feature."
             }

        if not portfolio:
             return {"error": "Empty portfolio"}

        # Setup
        n_assets = len(portfolio)
        asset_values = np.array([p.get("market_value", 0.0) for p in portfolio])
        vols = np.array([p.get("volatility", 0.2) for p in portfolio])

        # Covariance setup
        if correlation_matrix and len(correlation_matrix) == n_assets:
            corr = np.array(correlation_matrix)
        else:
            # Default to identity if missing (Independent assets)
            corr = np.eye(n_assets)

        # Construct Covariance Matrix
        D = np.diag(vols)
        cov_matrix = D @ corr @ D

        try:
            # Cholesky Decomposition: L * L.T = Sigma
            # Used to correlate random variables: X_corr = L * X_uncorr
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback if matrix is not positive semi-definite (e.g. bad data)
            logger.warning("Matrix not positive definite. Using Eigendecomposition fallback.")
            # Approximation using eigenvalues (reconstruct valid matrix)
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals = np.maximum(eigvals, 0) # Clamp negative eigenvalues
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        # Simulation
        # Generate random shocks: Z ~ N(0, 1)
        # Shape: (simulations, n_assets)
        Z_uncorr = np.random.standard_normal((simulations, n_assets))

        # Correlate shocks
        # (sims, n) @ (n, n).T -> (sims, n)
        Z_corr = Z_uncorr @ L.T

        # Calculate P&L for one day (dt = 1/252)
        # P&L = Value * Return
        # Return = Vol * sqrt(dt) * Z
        # We ignore drift (mu) for VaR as it's negligible daily and conservative to exclude.
        dt_sqrt = np.sqrt(1/252)

        # Asset P&L matrix
        # asset_values is (n,), Z_corr is (sims, n) -> need broadcasting
        # We need simulated returns in percentage terms first? No, Z_corr accounts for vol because L includes vol.
        # L comes from Covariance (vol^2 units). So L has units of vol.
        # So Z_corr has units of annual volatility.
        # Daily Return = Z_corr * dt_sqrt

        simulated_returns = Z_corr * dt_sqrt
        simulated_pnl = simulated_returns * asset_values # Broadcasting

        # Portfolio P&L per simulation (sum across assets)
        portfolio_pnl = np.sum(simulated_pnl, axis=1)

        # Calculate Risk Metrics from Distribution
        # VaR is the Xth percentile loss.
        # Since we want positive VaR for a loss, we look at the bottom tail.

        # Sort P&L from worst to best
        sorted_pnl = np.sort(portfolio_pnl)

        # Percentile index
        alpha = 1.0 - confidence_level # e.g. 0.05
        cutoff_idx = int(alpha * simulations)

        var_val = -sorted_pnl[cutoff_idx] # Negate to express as positive risk number

        # Expected Shortfall (CVaR) - Mean of losses worse than VaR
        tail_losses = sorted_pnl[:cutoff_idx]
        if len(tail_losses) > 0:
            cvar_val = -np.mean(tail_losses)
        else:
            cvar_val = var_val

        return {
            "VaR_MonteCarlo": round(float(var_val), 2),
            "CVaR_MonteCarlo": round(float(cvar_val), 2),
            "Simulations": simulations,
            "Confidence": confidence_level,
            "Method": "Monte_Carlo_Cholesky"
        }

    def execute_stress_test(self,
                            portfolio: List[Dict[str, Any]],
                            shocks: Dict[str, float]) -> Dict[str, Any]:
        """
        Performs deterministic stress testing on the portfolio.

        Args:
            portfolio: List of asset dictionaries.
            shocks: Dictionary mapping asset IDs to percentage return shocks (e.g., {"BTC": -0.20}).
                    Special keys:
                    - "ALL": Applies shock to all assets.

        Returns:
            Dict containing 'PnL', 'New_Total_Value', and 'Asset_Impacts'.
        """
        if not portfolio:
             return {"error": "Empty portfolio"}

        initial_value = sum(p.get("market_value", 0.0) for p in portfolio)

        shock_pnl = 0.0
        asset_impacts = {}

        global_shock = shocks.get("ALL", 0.0)

        for p in portfolio:
            asset_id = p.get("id", "Unknown")
            market_val = p.get("market_value", 0.0)

            # Determine specific shock
            # If asset specific shock exists, use it. Else use global shock.
            asset_shock = shocks.get(asset_id, global_shock)

            # Calculate P&L Impact
            # PnL = Value * Shock
            delta = market_val * asset_shock

            shock_pnl += delta
            asset_impacts[asset_id] = {
                "original_value": market_val,
                "shock_pct": asset_shock,
                "pnl": round(delta, 2)
            }

        return {
            "Scenario_PnL": round(shock_pnl, 2),
            "Scenario_Value": round(initial_value + shock_pnl, 2),
            "Initial_Value": round(initial_value, 2),
            "Asset_Impacts": asset_impacts,
            "Shocks_Applied": shocks
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
        else:  # Put
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
        else:  # Put
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

    def _get_z_score(self, confidence_level: float) -> float:
        """
        Returns the Z-score (1-tailed) for a given confidence level.
        Uses Scipy if available for precision.
        """
        if self.scipy_enabled:
            return norm.ppf(confidence_level)

        # Fallback Lookup
        if confidence_level <= 0.90: return 1.282
        if confidence_level <= 0.95: return 1.645
        if confidence_level <= 0.975: return 1.960
        if confidence_level <= 0.99: return 2.326
        return 3.090 # Default/Cap for >99%
