"""
Unit test suite for Neuro-Symbolic Risk Adapter & Multiplicative Assurance Gate
Module: tests/test_risk_adapter_gate.py
Compliance: Federal Reserve SR 11-7, OCC 2026-13, Basel CRE 31 (A-IRB)
"""

from __future__ import annotations

import pytest

from adam_finance.risk_adapter_gate import (
    CreditSymbolicMetrics,
    EpistemicStatus,
    RoutingAction,
    TimesFMRiskSurface,
    evaluate_neuro_symbolic_decision,
)


@pytest.fixture
def cloudwave_market_surface() -> TimesFMRiskSurface:
    return TimesFMRiskSurface(
        asset_id="CLOUDWAVE_1400M_TLB",
        horizon_days=60,
        adapter_param_count=2460000,
        quantiles={
            0.01: -0.195,
            0.05: -0.118,
            0.10: -0.065,
            0.50: 0.004,
            0.90: 0.042,
            0.95: 0.081,
            0.99: 0.125,
        },
        trailing_252d_exceedances_99=4,
        epistemic_tag=EpistemicStatus.MODEL_EST,
    )


@pytest.fixture
def cloudwave_credit_metrics() -> CreditSymbolicMetrics:
    return CreditSymbolicMetrics(
        obligor_id="CLOUDWAVE_SYSTEMS_LBO",
        fccr_reported=1.28,
        fccr_true_covenant=1.12,
        gross_leverage=8.38,
        dscr_stressed_p05=0.91,
        pd_pit=0.084,
        pd_ttc=0.038,
        cohort_75th_pct_pd_floor=0.092,
        brier_score_oot=0.041,
        lmt_uncapped_leakage_pct=0.245,
        accounting_identity_residual_usd_m=0.000,
        prov_o_root_sources={
            "urn:sec:edgar:0001844921-26-000412:10-Q:Note14",
            "urn:syndicate:loantape:cloudwave_tlb:q3_2026",
            "urn:market:timesfm:tail_risk_surface:tmt_software_b",
            "urn:legal:credit_agreement:cloudwave_2024_amended_inc",
        },
    )


def test_timesfm_monotonic_quantiles_repair() -> None:
    # Intentionally crossing quantiles
    surface = TimesFMRiskSurface(
        asset_id="TEST_ASSET",
        horizon_days=30,
        adapter_param_count=2460000,
        quantiles={
            0.01: -0.10,
            0.05: -0.15,  # Crossing violation: q_05 < q_01
            0.50: 0.02,
            0.99: 0.08,
        },
        trailing_252d_exceedances_99=2,
    )

    repaired = surface.enforce_monotonic_quantiles()
    assert repaired[0.01] == -0.10
    assert repaired[0.05] == -0.10  # Repaired to maintain monotonicity
    assert repaired[0.50] == 0.02
    assert repaired[0.99] == 0.08


def test_timesfm_ae_stress_multiplier(cloudwave_market_surface: TimesFMRiskSurface) -> None:
    surface = cloudwave_market_surface
    # Expected = 252 * 0.01 = 2.52; 4 / 2.52 = 1.5873...
    assert round(surface.actual_over_expected_99, 2) == 1.59
    # ae > 1.35 triggers stress multiplier equal to ae
    calibrated_var = surface.calibrated_tail_var_99()
    assert round(calibrated_var, 4) == 0.3095


def test_basel_cre31_hybrid_pd_floor(cloudwave_credit_metrics: CreditSymbolicMetrics) -> None:
    metrics = cloudwave_credit_metrics
    # Leverage = 8.38 > 6.0 and FCCR = 1.12 < 1.15 triggers 2LoD cohort 75th pct floor (0.092)
    pd_hybrid = metrics.hybrid_regulatory_pd(pit_weight=0.35)
    assert pd_hybrid == 0.092


def test_cloudwave_end_to_end_2lod_stress_test(
    cloudwave_credit_metrics: CreditSymbolicMetrics,
    cloudwave_market_surface: TimesFMRiskSurface,
) -> None:
    decision = evaluate_neuro_symbolic_decision(
        credit=cloudwave_credit_metrics,
        market=cloudwave_market_surface,
        s_grounding=4.9,
        s_downside=4.8,
        s_reusability=4.85,
        materiality_tier=2,
        human_approved=False,
    )

    assert decision["obligor_id"] == "CLOUDWAVE_SYSTEMS_LBO"
    assert decision["layer_0_multiplier"] == 1
    assert decision["final_assured_score"] == 4.853
    assert decision["hybrid_regulatory_pd"] == 0.092
    assert decision["calibrated_timesfm_var_99"] == 0.3095
    assert decision["timesfm_ae_ratio_99"] == 1.59
    assert decision["hard_tripwire_breached"] is True
    assert decision["routing_action"] == RoutingAction.HUMAN_ESCALATE.value


def test_layer_0_gating_failures(
    cloudwave_credit_metrics: CreditSymbolicMetrics,
    cloudwave_market_surface: TimesFMRiskSurface,
) -> None:
    # Insufficient provenance sources (< 3)
    unprovenanced_credit = CreditSymbolicMetrics(
        obligor_id="TEST_UNPROVENANCED",
        fccr_reported=1.40,
        fccr_true_covenant=1.35,
        gross_leverage=4.0,
        dscr_stressed_p05=1.20,
        pd_pit=0.01,
        pd_ttc=0.01,
        cohort_75th_pct_pd_floor=0.02,
        brier_score_oot=0.03,
        lmt_uncapped_leakage_pct=0.05,
        accounting_identity_residual_usd_m=0.00,
        prov_o_root_sources={"urn:root:only_one"},
    )

    res = evaluate_neuro_symbolic_decision(
        credit=unprovenanced_credit,
        market=cloudwave_market_surface,
        s_grounding=4.5,
        s_downside=4.5,
        s_reusability=4.5,
        materiality_tier=1,
    )
    assert res["layer_0_multiplier"] == 0
    assert res["final_assured_score"] == 0.0
    assert res["routing_action"] == RoutingAction.BLOCK_AND_LOG.value

    # Failed Brier score calibration triggers C5 degraded fallback
    uncalibrated_credit = CreditSymbolicMetrics(
        obligor_id="TEST_UNCALIBRATED",
        fccr_reported=1.40,
        fccr_true_covenant=1.35,
        gross_leverage=4.0,
        dscr_stressed_p05=1.20,
        pd_pit=0.01,
        pd_ttc=0.01,
        cohort_75th_pct_pd_floor=0.02,
        brier_score_oot=0.085,  # > 0.065
        lmt_uncapped_leakage_pct=0.05,
        accounting_identity_residual_usd_m=0.00,
        prov_o_root_sources={"urn:r1", "urn:r2", "urn:r3"},
    )
    res_degraded = evaluate_neuro_symbolic_decision(
        credit=uncalibrated_credit,
        market=cloudwave_market_surface,
        s_grounding=4.5,
        s_downside=4.5,
        s_reusability=4.5,
        materiality_tier=1,
    )
    assert res_degraded["layer_0_multiplier"] == 0
    assert res_degraded["routing_action"] == RoutingAction.C5_DEGRADED_FALLBACK.value
