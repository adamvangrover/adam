import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.risk_benchmark_replication import CapitalStructureInput, CreditStressEngine

@pytest.fixture
def default_corp():
    return CapitalStructureInput(
        entity_name="Benchmark Levered Telecom Corp",
        ticker="BLTC",
        total_debt=2500.00,
        senior_floating_pct=0.60,
        senior_floating_spread_bps=450.0,
        senior_fixed_pct=0.40,
        senior_fixed_coupon_pct=5.00,
        base_sofr_pct=0.0389,
        ebitda=280.00,
        maintenance_capex=90.00,
        mandatory_amort_pct=0.01,
        liquidation_ebitda_multiple=5.0,
        hazard_rates=[0.145, 0.220, 0.315],
    )

def test_compute_amortization_and_coverage(default_corp):
    engine = CreditStressEngine(default_corp)
    res = engine.compute_amortization_and_coverage()

    assert res["year_1"]["cash_fcf_dscr"] == pytest.approx(0.996)
    assert res["year_2"]["cash_fcf_dscr"] == pytest.approx(1.002)

def test_compute_valuation_waterfall_and_lgd(default_corp):
    engine = CreditStressEngine(default_corp)
    res = engine.compute_valuation_waterfall_and_lgd()

    assert res["waterfall"]["senior_secured"]["lgd_pct"] == pytest.approx(6.67)

def test_compute_cumulative_pd(default_corp):
    engine = CreditStressEngine(default_corp)
    res = engine.compute_cumulative_pd()

    assert res["cumulative_3y_pd_pct"] == pytest.approx(54.32)
