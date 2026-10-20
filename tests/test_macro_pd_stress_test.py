import pytest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_macro_pd_stress_test import (
    classify_macro_regime,
    calculate_merton_pd,
    compute_cumulative_pd_horizon,
    generate_parameter_sensitization_matrix,
    execute_red_team_challenge,
    generate_audit_dossier,
    run_analysis,
    DEFAULT_TARGET,
    LIVE_MACRO_DATA,
    norm_cdf,
    norm_ppf
)

def test_norm_cdf_and_ppf():
    assert math.isclose(norm_cdf(0.0), 0.5, abs_tol=1e-5)
    assert math.isclose(norm_ppf(0.5), 0.0, abs_tol=1e-5)
    assert norm_cdf(1.96) > 0.97
    assert norm_ppf(0.975) > 1.95

def test_classify_macro_regime():
    # Soft landing case
    regime = classify_macro_regime(LIVE_MACRO_DATA)
    assert regime == "Slowdown / Disinflationary Soft Landing"

    # Liquidity Crunch case
    stress_macro = dict(LIVE_MACRO_DATA)
    stress_macro["vix_index"] = {"value": 35.0, "unit": "pts", "source": "test"}
    assert classify_macro_regime(stress_macro) == "Liquidity Crunch / Market Disruption"

def test_calculate_merton_pd():
    dd, sigma_a, pd_cum = calculate_merton_pd(
        equity_val=3960.0,
        equity_vol=0.195,
        debt_val=108.5,
        rf_rate=0.045,
        horizon=1.0
    )
    assert dd > 10.0 # High distance to default for mega-cap tech
    assert 0.0 <= pd_cum < 0.01
    assert sigma_a < 0.195

def test_compute_cumulative_pd_horizon():
    pd_1y = 0.0025 # 25 bps
    pd_3y = compute_cumulative_pd_horizon(pd_1y, years=3.0)
    assert pd_3y > pd_1y
    assert math.isclose(pd_3y, 1 - math.exp(-(-math.log(1 - pd_1y))*3), rel_tol=1e-5)

def test_generate_parameter_sensitization_matrix():
    matrix = generate_parameter_sensitization_matrix(DEFAULT_TARGET, LIVE_MACRO_DATA)
    assert len(matrix) == 4

    scenarios = [m["scenario"] for m in matrix]
    assert "Baseline (Historical Anchor)" in scenarios[0]
    assert "Severe Liquidity Shock" in scenarios[3]

    # Monotonicity check: Severe liquidity shock should have higher 1Y PD than Baseline
    pd_base = matrix[0]["pd_1y_bps"]
    pd_severe = matrix[3]["pd_1y_bps"]
    assert pd_severe > pd_base

def test_execute_red_team_challenge():
    matrix = generate_parameter_sensitization_matrix(DEFAULT_TARGET, LIVE_MACRO_DATA)
    red_team = execute_red_team_challenge(matrix)

    assert len(red_team["vulnerabilities"]) == 4
    assert red_team["total_red_team_surcharge_bps"] == 29.0
    assert "bps" in red_team["challenged_pd_range_live_bps"]

def test_generate_audit_dossier():
    target = DEFAULT_TARGET
    macro = LIVE_MACRO_DATA
    regime = classify_macro_regime(macro)
    matrix = generate_parameter_sensitization_matrix(target, macro)
    red_team = execute_red_team_challenge(matrix)

    audit = generate_audit_dossier(target, macro, regime, matrix, red_team)
    assert "dossier_hash" in audit
    assert len(audit["dossier_hash"]) == 64 # SHA-256 length
    assert len(audit["lineage"]) == 8

def test_run_analysis_end_to_end():
    report_md, audit = run_analysis(DEFAULT_TARGET)
    assert "# Institutional Credit Risk Audit Report" in report_md
    assert "Step 1: Live Macro & Market Pulse Check" in report_md
    assert "Step 2: Entity PD Parameter Sensitization Matrix" in report_md
    assert "Step 3: Adversarial Machine Challenge (Red Team)" in report_md
    assert "Step 4: Audit Trail & Expert Review Log" in report_md
    assert audit["dossier_hash"] in report_md
