"""
Unit tests for Scrambled Sobol Quasi-Monte Carlo (QMC) Engine.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.risk_engine.scrambled_sobol_qmc import ScrambledSobolQMCEngine, norm_ppf


def test_norm_ppf_bounds():
    u = np.array([0.0, 0.5, 1.0])
    z = norm_ppf(u)
    assert np.isclose(z[1], 0.0, atol=1e-5)
    assert np.isfinite(z[0])
    assert np.isfinite(z[2])


def test_uniform_samples_shape_and_bounds():
    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=123)
    n_samples = 64
    dim = 5
    samples = engine.generate_uniform_samples(n_samples=n_samples, dimension=dim)

    assert samples.shape == (n_samples, dim)
    assert np.all(samples >= 0.0)
    assert np.all(samples < 1.0)


def test_gaussian_samples_multivariate():
    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)
    n_samples = 1024
    mean = np.array([1.0, -0.5])
    cov = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ])

    samples = engine.generate_gaussian_samples(
        n_samples=n_samples,
        mean=mean,
        cov=cov,
        seed=42
    )

    assert samples.shape == (n_samples, 2)
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples, rowvar=False)

    assert np.allclose(sample_mean, mean, atol=0.1)
    assert np.allclose(sample_cov, cov, atol=0.2)


def test_merton_credit_stress_simulation():
    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)
    res = engine.simulate_merton_credit_stress(
        asset_value=100.0,
        debt_face_value=80.0,
        volatility=0.30,
        risk_free_rate=0.05,
        time_horizon=1.0,
        n_simulations=512,
        scramble=True,
        n_scramblings=5,
        seed=42,
    )

    assert "probability_of_default" in res
    assert 0.0 <= res["probability_of_default"] <= 1.0
    assert "expected_asset_value" in res
    assert res["expected_asset_value"] > 0
    assert "qmc_standard_error_pd" in res
    assert res["qmc_standard_error_pd"] >= 0.0
    assert "VaR_95" in res
    assert "Expected_Shortfall_95" in res
    assert res["Expected_Shortfall_95"] >= res["VaR_95"]


def test_multi_factor_stress_simulation():
    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)
    exposures = np.array([10.0, 20.0])
    vols = np.array([0.2, 0.3])
    corr = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    shocks = np.array([-0.05, -0.10])

    res = engine.simulate_multi_factor_stress(
        portfolio_exposure=exposures,
        asset_volatilities=vols,
        correlation_matrix=corr,
        stress_shocks=shocks,
        n_simulations=512,
        scramble=True,
        seed=42,
    )

    assert "portfolio_mean_loss" in res
    assert "portfolio_VaR_95" in res
    assert "portfolio_CVaR_95" in res
    assert res["portfolio_CVaR_95"] >= res["portfolio_VaR_95"]
    assert len(res["component_risk_contributions"]) == 2


def test_compare_mc_vs_qmc():
    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)
    res = engine.compare_mc_vs_qmc(
        n_simulations=512,
        dimension=3,
        n_runs=10,
        seed=42,
    )

    assert "empirical_variance_MC" in res
    assert "empirical_variance_scrambled_QMC" in res
    assert "variance_reduction_vs_MC" in res
    assert res["variance_reduction_vs_MC"] > 0.5  # QMC variance is lower or comparable


def test_invalid_inputs():
    engine = ScrambledSobolQMCEngine()

    with pytest.raises(ValueError, match="Dimension must be >= 1"):
        engine.generate_uniform_samples(n_samples=100, dimension=0)

    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        engine.generate_uniform_samples(n_samples=0, dimension=2)

    with pytest.raises(ValueError, match="Asset value"):
        engine.simulate_merton_credit_stress(
            asset_value=-10.0,
            debt_face_value=80.0,
            volatility=0.2,
            risk_free_rate=0.05,
            time_horizon=1.0,
        )
