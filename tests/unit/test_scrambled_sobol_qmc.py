"""
Unit tests for Scrambled Sobol Quasi-Monte Carlo (QMC) Engine.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import subprocess
import sys
from pathlib import Path

from core.risk_engine.scrambled_sobol_qmc import (
    ScrambledSobolQMCEngine,
    norm_ppf,
    student_t_ppf,
)


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


def test_student_t_ppf_and_samples():
    u = np.array([0.01, 0.5, 0.99])
    t_val = student_t_ppf(u, df=5.0)
    assert np.isclose(t_val[1], 0.0, atol=1e-4)
    assert t_val[0] < 0.0 < t_val[2]

    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)
    n_samples = 1024
    mean = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.3], [0.3, 1.5]])

    samples = engine.generate_student_t_samples(
        n_samples=n_samples,
        df=5.0,
        mean=mean,
        cov=cov,
        seed=42,
    )

    assert samples.shape == (n_samples, 2)
    sample_mean = np.mean(samples, axis=0)
    assert np.allclose(sample_mean, mean, atol=0.2)


def test_compute_sobol_sensitivity_indices():
    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)

    # Linear additive model Y = 5*X1 + X2 + 0*X3 where X_i ~ Uniform(0,1)
    def linear_model(x):
        return 5.0 * x[:, 0] + 1.0 * x[:, 1] + 0.0 * x[:, 2]

    res = engine.compute_sobol_sensitivity_indices(
        model_func=linear_model,
        dimension=3,
        n_samples=512,
        param_names=["Alpha", "Beta", "Gamma"],
        seed=42,
    )

    assert "first_order_indices" in res
    assert "total_order_indices" in res
    assert res["parameter_rankings"][0] == "Alpha"  # Alpha should dominate
    assert res["first_order_indices"]["Alpha"] > res["first_order_indices"]["Beta"]
    assert res["first_order_indices"]["Beta"] >= res["first_order_indices"]["Gamma"]

    for name in ["Alpha", "Beta", "Gamma"]:
        s_i = res["first_order_indices"][name]
        st_i = res["total_order_indices"][name]
        assert 0.0 <= s_i <= st_i <= 1.0


def test_compute_sobol_sensitivity_edge_cases():
    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)

    # Zero-variance model
    def zero_model(x):
        return np.zeros(x.shape[0])

    res = engine.compute_sobol_sensitivity_indices(
        model_func=zero_model,
        dimension=2,
        n_samples=64,
        seed=42,
    )
    assert res["model_variance"] == 0.0
    assert res["first_order_indices"]["X_1"] == 0.0

    with pytest.raises(ValueError, match="Dimension must be >= 1"):
        engine.compute_sobol_sensitivity_indices(model_func=zero_model, dimension=0)

    with pytest.raises(ValueError, match="n_samples must be >= 16"):
        engine.compute_sobol_sensitivity_indices(model_func=zero_model, dimension=2, n_samples=8)

    with pytest.raises(ValueError, match="param_names length"):
        engine.compute_sobol_sensitivity_indices(model_func=zero_model, dimension=2, n_samples=32, param_names=["A"])


def test_simulate_covenant_headroom_stress():
    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)
    res = engine.simulate_covenant_headroom_stress(
        consolidated_ebitda=20.0,
        total_funded_debt=80.0,
        annual_interest_expense=6.0,
        debt_service_due=12.0,
        annual_capex=2.0,
        annual_tax=1.0,
        max_leverage_covenant=4.5,
        min_interest_coverage_covenant=3.0,
        min_dscr_covenant=1.25,
        ebitda_volatility=0.15,
        n_simulations=512,
        seed=42,
    )

    assert res["simulation_status"] == "DETERMINISTIC_QMC_CONVERGED"
    assert "breach_probabilities" in res
    bp = res["breach_probabilities"]
    assert 0.0 <= bp["joint_covenant_breach_probability"] <= 1.0
    assert bp["joint_covenant_breach_probability"] >= bp["leverage_breach_probability"]
    assert "mean_headroom" in res
    assert "tail_risk" in res


def test_simulate_covenant_headroom_stress_invalid_inputs():
    engine = ScrambledSobolQMCEngine()

    with pytest.raises(ValueError, match="Consolidated EBITDA"):
        engine.simulate_covenant_headroom_stress(
            consolidated_ebitda=-5.0,
            total_funded_debt=80.0,
            annual_interest_expense=6.0,
            debt_service_due=12.0,
        )

    with pytest.raises(ValueError, match="Covenant thresholds"):
        engine.simulate_covenant_headroom_stress(
            consolidated_ebitda=20.0,
            total_funded_debt=80.0,
            annual_interest_expense=6.0,
            debt_service_due=12.0,
            max_leverage_covenant=0.0,
        )


def test_simulate_multi_horizon_credit_stress():
    engine = ScrambledSobolQMCEngine(default_scramble=True, seed=42)
    horizons = [1.0, 3.0, 5.0]
    res = engine.simulate_multi_horizon_credit_stress(
        asset_value=100.0,
        debt_face_value=85.0,
        volatility=0.30,
        risk_free_rate=0.04,
        horizons=horizons,
        n_simulations=512,
        seed=42,
    )

    assert res["horizons"] == horizons
    cum_pd = res["cumulative_pd_term_structure"]
    assert cum_pd["1Y"] <= cum_pd["3Y"] <= cum_pd["5Y"]
    assert len(res["marginal_pd_term_structure"]) == 3
    assert len(res["expected_asset_value_term_structure"]) == 3


def test_simulate_multi_horizon_invalid_inputs():
    engine = ScrambledSobolQMCEngine()

    with pytest.raises(ValueError, match="strictly increasing"):
        engine.simulate_multi_horizon_credit_stress(
            asset_value=100.0,
            debt_face_value=85.0,
            volatility=0.30,
            risk_free_rate=0.04,
            horizons=[3.0, 1.0],
        )

    with pytest.raises(ValueError, match="At least one horizon"):
        engine.simulate_multi_horizon_credit_stress(
            asset_value=100.0,
            debt_face_value=85.0,
            volatility=0.30,
            risk_free_rate=0.04,
            horizons=[],
        )


def test_cli_runner(tmp_path: Path):
    out_json = tmp_path / "qmc_test_out.json"
    out_md = tmp_path / "qmc_test_report.md"

    cmd = [
        sys.executable,
        "scripts/run_scrambled_sobol_qmc.py",
        "--simulations",
        "256",
        "--mode",
        "all",
        "--json",
        str(out_json),
        "--markdown",
        str(out_md),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI runner failed: {result.stderr}"
    assert out_json.exists()
    assert out_md.exists()
    assert "SCRAMBLED SOBOL QUASI-MONTE CARLO" in result.stdout
