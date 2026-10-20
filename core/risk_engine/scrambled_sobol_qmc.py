"""
Scrambled Sobol Quasi-Monte Carlo (QMC) Stress Simulation Engine.

This module implements low-discrepancy Scrambled Sobol sequence generation (Owen's scrambling),
multivariate inverse transform Gaussian mapping via Cholesky decomposition, standard error
estimation across randomized scramblings, and multi-factor macro/credit risk stress testing.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try importing scipy for high-performance QMC and norm.ppf
try:
    import scipy.stats as stats
    import scipy.stats.qmc as qmc

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


def _erfinv_numpy(x: np.ndarray) -> np.ndarray:
    """Fallback inverse error function approximation if scipy is not installed."""
    # Winitzki approximation for erfinv
    a = 0.147
    log_term = np.log(1.0 - x**2)
    term1 = 2.0 / (np.pi * a) + log_term / 2.0
    inner = term1**2 - log_term / a
    sign = np.sign(x)
    return sign * np.sqrt(np.sqrt(inner) - term1)


def norm_ppf(u: np.ndarray) -> np.ndarray:
    """
    Inverse cumulative distribution function (quantile function) for Standard Normal N(0,1).
    Clips inputs to (1e-12, 1 - 1e-12) to prevent numerical infinity.
    """
    u_clipped = np.clip(u, 1e-12, 1.0 - 1e-12)
    if HAS_SCIPY:
        return stats.norm.ppf(u_clipped)
    else:  # pragma: no cover
        # norm.ppf(u) = sqrt(2) * erfinv(2*u - 1)
        return math.sqrt(2.0) * _erfinv_numpy(2.0 * u_clipped - 1.0)


class ScrambledSobolQMCEngine:
    """
    Scrambled Sobol Quasi-Monte Carlo Engine for Quantitative Stress Simulation.

    Achieves O(N^{-3/2 + epsilon}) convergence rate for smooth integrands compared to O(N^{-1/2})
    in standard pseudo-random Monte Carlo. Uses Owen's scrambling or randomized digital shifts
    to eliminate high-dimensional alignment artifacts and provide unbiased standard error estimates.
    """

    def __init__(self, default_scramble: bool = True, seed: Optional[int] = None) -> None:
        self.default_scramble = default_scramble
        self.seed = seed
        logger.info(
            f"Initialized ScrambledSobolQMCEngine (SciPy available: {HAS_SCIPY}, default_scramble: {default_scramble})"
        )

    def generate_uniform_samples(
        self,
        n_samples: int,
        dimension: int,
        scramble: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates low-discrepancy points in [0, 1)^dimension.

        Args:
            n_samples: Number of samples to draw (preferably power of 2 for optimal Sobol balance).
            dimension: Dimensionality of factor space.
            scramble: Whether to apply Owen scrambling/randomized shift.
            seed: Optional seed for scrambling reproducibility.

        Returns:
            np.ndarray of shape (n_samples, dimension) in [0, 1).
        """
        if dimension < 1:
            raise ValueError(f"Dimension must be >= 1, got {dimension}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        should_scramble = self.default_scramble if scramble is None else scramble
        effective_seed = self.seed if seed is None else seed

        if HAS_SCIPY:
            # SciPy's Sobol requires n_samples to be a power of 2 for full balance property,
            # but random(n) accepts any integer n.
            sampler = qmc.Sobol(d=dimension, scramble=should_scramble, seed=effective_seed)
            points = sampler.random(n=n_samples)
            return points
        else:  # pragma: no cover
            # Fallback to pseudo-random uniform if SciPy missing
            rng = np.random.default_rng(effective_seed)
            return rng.random((n_samples, dimension))

    def generate_gaussian_samples(
        self,
        n_samples: int,
        mean: Optional[np.ndarray] = None,
        cov: Optional[np.ndarray] = None,
        dimension: Optional[int] = None,
        scramble: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates low-discrepancy multivariate normal Gaussian draws N(mean, cov).

        Args:
            n_samples: Number of samples.
            mean: Mean vector of length `d`. Defaults to zeros(d).
            cov: Covariance matrix (d, d). Defaults to Identity matrix.
            dimension: Dimension `d` if mean/cov not specified.
            scramble: Whether to scramble Sobol sequence.
            seed: Seed for scrambling.

        Returns:
            np.ndarray of shape (n_samples, d) with target mean and covariance.
        """
        if mean is not None:
            mean = np.asarray(mean, dtype=np.float64)
            d = mean.shape[0]
        elif cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            d = cov.shape[0]
        elif dimension is not None:
            d = dimension
            mean = np.zeros(d, dtype=np.float64)
            cov = np.eye(d, dtype=np.float64)
        else:
            raise ValueError("Must specify at least one of mean, cov, or dimension.")

        if mean is None:
            mean = np.zeros(d, dtype=np.float64)
        if cov is None:
            cov = np.eye(d, dtype=np.float64)

        # 1. Sample low-discrepancy uniforms U in [0, 1)^d
        u = self.generate_uniform_samples(n_samples=n_samples, dimension=d, scramble=scramble, seed=seed)

        # 2. Map via Inverse Normal CDF: Z = Phi^{-1}(U) ~ N(0, I)
        z = norm_ppf(u)

        # 3. Apply Cholesky factor L (or SVD fallback) to enforce correlation structure: X = mean + Z * L^T
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:  # pragma: no cover
            # Fallback for near-PD or semi-PD matrices using SVD/Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        x = mean + z @ L.T
        return x

    def simulate_merton_credit_stress(
        self,
        asset_value: float,
        debt_face_value: float,
        volatility: float,
        risk_free_rate: float,
        time_horizon: float,
        n_simulations: int = 1024,
        scramble: bool = True,
        n_scramblings: int = 10,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Simulates Merton Structural Credit Risk Model using Scrambled Sobol QMC
        and estimates Standard Errors across randomized scramblings.

        Args:
            asset_value: V_0 (Current Asset Value).
            debt_face_value: D (Debt Face Value due at T).
            volatility: sigma (Asset Volatility).
            risk_free_rate: r (Risk-free interest rate).
            time_horizon: T (Maturity in years).
            n_simulations: N (Number of QMC simulation paths per scrambling).
            scramble: Whether to use Scrambled Sobol vs Unscrambled Sobol.
            n_scramblings: M (Number of randomized scramblings for SE estimation).
            seed: Random seed base.

        Returns:
            Dict containing PD estimate, expected terminal asset value, QMC standard error,
            VaR_95, VaR_99, Expected Shortfall (CVaR_95), and efficiency metrics.
        """
        if asset_value <= 0 or debt_face_value <= 0 or volatility <= 0 or time_horizon <= 0:
            raise ValueError("Asset value, debt face value, volatility, and time horizon must be positive.")

        dt = time_horizon
        drift = (risk_free_rate - 0.5 * volatility**2) * dt
        diffusion_scale = volatility * math.sqrt(dt)

        pd_estimates = []
        expected_v_t_estimates = []
        all_v_t = []

        base_seed = self.seed if seed is None else seed

        for m in range(n_scramblings):
            sub_seed = None if base_seed is None else (base_seed + m * 1000 + 1)
            # Generate 1D Gaussian shocks using Scrambled Sobol
            z = self.generate_gaussian_samples(
                n_samples=n_simulations,
                dimension=1,
                scramble=scramble,
                seed=sub_seed,
            ).flatten()

            # Merton GBM solution: V_T = V_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
            v_t = asset_value * np.exp(drift + diffusion_scale * z)
            defaults = (v_t < debt_face_value).astype(np.float64)

            pd_estimates.append(np.mean(defaults))
            expected_v_t_estimates.append(np.mean(v_t))
            all_v_t.append(v_t)

        pd_mean = float(np.mean(pd_estimates))
        expected_v_t_mean = float(np.mean(expected_v_t_estimates))

        # Standard Error across M scramblings: SE = std(pd_estimates) / sqrt(M)
        if n_scramblings > 1:
            qmc_se_pd = float(np.std(pd_estimates, ddof=1) / math.sqrt(n_scramblings))
            qmc_se_vt = float(np.std(expected_v_t_estimates, ddof=1) / math.sqrt(n_scramblings))
        else:
            qmc_se_pd = float(math.sqrt(max(0.0, pd_mean * (1.0 - pd_mean)) / n_simulations))
            qmc_se_vt = float(np.std(all_v_t[0]) / math.sqrt(n_simulations))

        # Combine all terminal asset values for risk metrics calculation
        combined_v_t = np.concatenate(all_v_t)
        losses = np.maximum(0.0, asset_value - combined_v_t)

        var_95 = float(np.percentile(losses, 95))
        var_99 = float(np.percentile(losses, 99))

        tail_losses_95 = losses[losses >= var_95]
        cvar_95 = float(np.mean(tail_losses_95)) if len(tail_losses_95) > 0 else var_95

        # Theoretical error comparison
        total_paths = n_simulations * n_scramblings
        standard_mc_se = math.sqrt(max(0.0, pd_mean * (1.0 - pd_mean)) / total_paths)

        return {
            "probability_of_default": pd_mean,
            "expected_asset_value": expected_v_t_mean,
            "qmc_standard_error_pd": qmc_se_pd,
            "standard_mc_equivalent_se": standard_mc_se,
            "variance_reduction_factor": float((standard_mc_se / max(1e-12, qmc_se_pd)) ** 2),
            "VaR_95": var_95,
            "VaR_99": var_99,
            "Expected_Shortfall_95": cvar_95,
            "simulation_parameters": {
                "n_simulations_per_batch": n_simulations,
                "n_scramblings": n_scramblings,
                "total_paths": total_paths,
                "scramble_enabled": scramble,
            },
        }

    def simulate_multi_factor_stress(
        self,
        portfolio_exposure: np.ndarray,
        asset_volatilities: np.ndarray,
        correlation_matrix: np.ndarray,
        stress_shocks: Optional[np.ndarray] = None,
        n_simulations: int = 2048,
        scramble: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes multi-factor joint risk factor stress testing using Scrambled Sobol QMC.

        Args:
            portfolio_exposure: Vector of monetary exposures (N_assets,).
            asset_volatilities: Vector of asset volatilities (N_assets,).
            correlation_matrix: Correlation matrix (N_assets, N_assets).
            stress_shocks: Vector of macroeconomic deterministic mean shifts (N_assets,).
            n_simulations: Number of QMC paths.
            scramble: Whether to apply Owen scrambling.
            seed: Seed for scrambling.

        Returns:
            Dict containing portfolio mean loss, VaR_95, VaR_99, Expected Shortfall, and asset contributions.
        """
        portfolio_exposure = np.asarray(portfolio_exposure, dtype=np.float64)
        asset_volatilities = np.asarray(asset_volatilities, dtype=np.float64)
        correlation_matrix = np.asarray(correlation_matrix, dtype=np.float64)
        d = portfolio_exposure.shape[0]

        if stress_shocks is None:
            stress_shocks = np.zeros(d, dtype=np.float64)
        else:
            stress_shocks = np.asarray(stress_shocks, dtype=np.float64)

        # Construct Covariance Matrix from volatilities and correlation matrix
        D = np.diag(asset_volatilities)
        cov_matrix = D @ correlation_matrix @ D

        # Generate low-discrepancy factor draws
        simulated_returns = self.generate_gaussian_samples(
            n_samples=n_simulations,
            mean=stress_shocks,
            cov=cov_matrix,
            scramble=scramble,
            seed=seed,
        )

        # Calculate Asset Value Changes and Portfolio Losses
        asset_losses = -simulated_returns * portfolio_exposure  # shape (n_simulations, d)
        portfolio_losses = np.sum(asset_losses, axis=1)  # shape (n_simulations,)

        mean_loss = float(np.mean(portfolio_losses))
        var_95 = float(np.percentile(portfolio_losses, 95))
        var_99 = float(np.percentile(portfolio_losses, 99))

        tail_mask_95 = portfolio_losses >= var_95
        cvar_95 = float(np.mean(portfolio_losses[tail_mask_95])) if np.any(tail_mask_95) else var_95

        # Component Risk Contribution (Tail Expectation)
        subgroup_tail_losses = asset_losses[tail_mask_95]
        risk_contributions = np.mean(subgroup_tail_losses, axis=0) if np.any(tail_mask_95) else np.zeros(d)

        return {
            "portfolio_mean_loss": mean_loss,
            "portfolio_VaR_95": var_95,
            "portfolio_VaR_99": var_99,
            "portfolio_CVaR_95": cvar_95,
            "component_risk_contributions": risk_contributions.tolist(),
            "n_simulations": n_simulations,
            "dimension": d,
            "scramble": scramble,
        }

    def compare_mc_vs_qmc(
        self,
        n_simulations: int = 1024,
        dimension: int = 4,
        n_runs: int = 20,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Executes empirical comparison between Standard Pseudo-Random MC, Unscrambled Sobol QMC,
        and Scrambled Sobol QMC to demonstrate variance reduction and convergence efficiency.

        Args:
            n_simulations: Paths per estimation run.
            dimension: Dimension of integration space.
            n_runs: Number of repeated estimation runs to calculate empirical variance.
            seed: Base seed.

        Returns:
            Dict containing empirical variances, standard errors, and variance reduction ratios.
        """
        mc_estimates = []
        unscrambled_qmc_estimates = []
        scrambled_qmc_estimates = []

        rng = np.random.default_rng(seed)

        # Test Integrand: Smooth Gaussian Integral f(x) = prod(x_i^2)
        for r in range(n_runs):
            # 1. Standard Pseudo-Random MC
            z_mc = rng.normal(size=(n_simulations, dimension))
            val_mc = np.mean(np.prod(z_mc**2, axis=1))
            mc_estimates.append(val_mc)

            # 2. Unscrambled Sobol QMC
            z_unscrambled = self.generate_gaussian_samples(
                n_samples=n_simulations,
                dimension=dimension,
                scramble=False,
                seed=(seed + r if seed else None),
            )
            val_unscrambled = np.mean(np.prod(z_unscrambled**2, axis=1))
            unscrambled_qmc_estimates.append(val_unscrambled)

            # 3. Scrambled Sobol QMC
            z_scrambled = self.generate_gaussian_samples(
                n_samples=n_simulations,
                dimension=dimension,
                scramble=True,
                seed=(seed + r * 100 if seed else None),
            )
            val_scrambled = np.mean(np.prod(z_scrambled**2, axis=1))
            scrambled_qmc_estimates.append(val_scrambled)

        var_mc = float(np.var(mc_estimates, ddof=1))
        var_unscrambled = float(np.var(unscrambled_qmc_estimates, ddof=1))
        var_scrambled = float(np.var(scrambled_qmc_estimates, ddof=1))

        # True analytical expectation for prod(Z_i^2) where Z_i ~ N(0,1) is 1^dimension = 1.0
        true_val = 1.0

        return {
            "empirical_mean_MC": float(np.mean(mc_estimates)),
            "empirical_mean_unscrambled_QMC": float(np.mean(unscrambled_qmc_estimates)),
            "empirical_mean_scrambled_QMC": float(np.mean(scrambled_qmc_estimates)),
            "empirical_variance_MC": var_mc,
            "empirical_variance_unscrambled_QMC": var_unscrambled,
            "empirical_variance_scrambled_QMC": var_scrambled,
            "variance_reduction_vs_MC": float(var_mc / max(1e-12, var_scrambled)),
            "rmse_MC": float(math.sqrt(np.mean((np.array(mc_estimates) - true_val) ** 2))),
            "rmse_scrambled_QMC": float(math.sqrt(np.mean((np.array(scrambled_qmc_estimates) - true_val) ** 2))),
            "n_simulations": n_simulations,
            "dimension": dimension,
            "n_runs": n_runs,
        }
