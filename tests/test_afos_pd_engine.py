import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.afos_pd_engine import calculate_pd

def test_calculate_pd_default():
    res = calculate_pd(
        name="Carvana Co.",
        ticker_or_lei="CVNA",
        E=18200.0,
        sigma_E=0.68,
        s=420.0,
        t_recency=0,
        Total_Debt=5400.0,
        STD=600.0,
        EBITDA=420.0,
        Interest=380.0,
        FCF=None,
        Total_Assets=7800.0,
        Cash=550.0,
        active_risk_flags=["debt_restructuring_advisors"],
        flags_sum=1.10
    )

    assert res["entity_identification"]["name"] == "Carvana Co."
    assert res["dual_world_arbitration"]["execution_state"] == "CONTESTED"
    assert res["model_pillars"]["structural_merton_pd"] == 0.0009
    assert res["model_pillars"]["market_spread_pd"] == 0.0676
    assert "Free Cash Flow (FCF) - Estimated as EBITDA minus Interest Expense" in res["telemetry_audit"]["missing_telemetry_inputs"]
