import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.risk_benchmark_replication import (
    CapitalStructureInput,
    MacroClearingInput,
    CreditStressEngine,
    RegulatoryFilingEvaluator,
    generate_ledger_payload,
)

@pytest.fixture
def oct_02_corp():
    return CapitalStructureInput(
        entity_name="Benchmark Levered Telecom Corp",
        ticker="BLTC",
        total_debt=2500.00,
        senior_floating_pct=0.60,
        senior_floating_spread_bps=450.0,
        senior_fixed_pct=0.40,
        senior_fixed_coupon_pct=5.00,
        base_sofr_pct=0.0388,
        ebitda=280.00,
        maintenance_capex=90.00,
        mandatory_amort_pct=0.01,
        liquidation_ebitda_multiple=5.0,
        hazard_rates=[0.145, 0.220, 0.315],
    )

@pytest.fixture
def oct_02_macro():
    return MacroClearingInput(
        spx_level=7721.58,
        us_10y_yield_pct=5.276,
        sofr_cash_pct=3.88,
        brent_crude_usd=102.25,
        vix_level=15.31,
        cdx_hy_spread_bps=395.0,
    )

def test_compute_amortization_and_coverage(oct_02_corp):
    engine = CreditStressEngine(oct_02_corp)
    res = engine.compute_amortization_and_coverage()

    assert res["all_in_floating_rate_pct"] == pytest.approx(8.38)
    assert res["gross_leverage_ratio"] == pytest.approx(8.93)
    assert res["year_1"]["cash_fcf_dscr"] == pytest.approx(0.996)
    assert res["year_1"]["net_fcf_post_debt_service"] == pytest.approx(-0.70)
    assert res["year_2"]["cash_fcf_dscr"] == pytest.approx(1.003)
    assert res["year_2"]["net_fcf_post_debt_service"] == pytest.approx(0.56)

def test_compute_valuation_waterfall_and_lgd(oct_02_corp):
    engine = CreditStressEngine(oct_02_corp)
    res = engine.compute_valuation_waterfall_and_lgd()

    assert res["distressed_ev"] == pytest.approx(1400.00)
    assert res["waterfall"]["senior_secured"]["recovery"] == pytest.approx(1400.00)
    assert res["waterfall"]["senior_secured"]["recovery_rate_pct"] == pytest.approx(93.33)
    assert res["waterfall"]["senior_secured"]["lgd_pct"] == pytest.approx(6.67)
    assert res["waterfall"]["senior_unsecured"]["recovery"] == pytest.approx(0.00)
    assert res["waterfall"]["senior_unsecured"]["lgd_pct"] == pytest.approx(100.00)

def test_compute_cumulative_pd(oct_02_corp):
    engine = CreditStressEngine(oct_02_corp)
    res = engine.compute_cumulative_pd()

    assert res["cumulative_3y_survival_prob_pct"] == pytest.approx(45.68)
    assert res["cumulative_3y_pd_pct"] == pytest.approx(54.32)

def test_oct_02_2026_ledger_payload(oct_02_corp, oct_02_macro):
    engine = CreditStressEngine(oct_02_corp)
    payload = generate_ledger_payload(engine, oct_02_macro)

    assert payload["ledger_id"] == "MM-V30-20261002-SYSCORE"
    assert payload["system_status"] == "CRITICAL"
    assert payload["clearing_metrics"]["spx_close"] == 7721.58
    assert payload["clearing_metrics"]["us10y_yield"] == 5.276
    assert payload["clearing_metrics"]["sofr_rate"] == 3.88
    assert payload["clearing_metrics"]["brent_usd"] == 102.25

    # Trailing 5D checks
    assert len(payload["trailing_5d_series"]["spx"]) == 5
    assert payload["trailing_5d_series"]["spx"][-1] == 7721.58
    assert payload["trailing_5d_series"]["us10y_yield"][-1] == 5.276

    # YoY Comparative metrics checks
    assert payload["yoy_comparative_metrics"]["us10y_yield_yoy_delta_bps"] == 115.6
    assert payload["yoy_comparative_metrics"]["brent_yoy_pct_change"] == 30.3

    assert payload["quantitative_credit_stress"]["cumulative_3y_pd_pct"] == pytest.approx(54.32)
    assert payload["headline_arena_forecasts"]["us10y_target"] == 5.35

def test_regulatory_filing_evaluator():
    telemetry = {
        "cash_runway_months": 8,
        "going_concern_flag": True,
        "leverage_covenant_headroom": 0.30,
        "revolver_draw_pct": 75.0,
        "primary_spread_over_sofr_bps": 475,
        "lmt_priming_clause_detected": True,
    }
    triggers = RegulatoryFilingEvaluator.evaluate_triggers(telemetry)
    assert len(triggers) == 4
    forms = [t["form"] for t in triggers]
    assert "10-K" in forms
    assert "10-Q" in forms
    assert "424B5 / Indenture" in forms
    assert "8-K (Item 1.01)" in forms
