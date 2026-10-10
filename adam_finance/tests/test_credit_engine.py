"""
Pytest unit test suite for adam_finance/credit_engine.py
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from adam_finance.credit_engine import (
    MultiHorizonCreditEngine,
    TTCAnchorEngine,
    PITLTMEngine,
    ForwardShadowEngine,
    EquityStructuralEngine,
    TemporalEnvelope,
    ModelProposal,
)


def mock_envelope():
    return {
        "t_effective": datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc),
        "t_knowledge": datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc),
        "t_decision": datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc),
        "t_execution": datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc),
    }


def standard_proposals(ttc_bps=150, ltm_bps=185, fwd_bps=210, eq_bps=240):
    return [
        {"model_name": "TTC", "pd_1y_bps": ttc_bps, "confidence_permille": 850},
        {"model_name": "LTM", "pd_1y_bps": ltm_bps, "confidence_permille": 750},
        {"model_name": "FWD", "pd_1y_bps": fwd_bps, "confidence_permille": 700},
        {"model_name": "EQUITY", "pd_1y_bps": eq_bps, "confidence_permille": 900},
    ]


def test_ttc_anchor_calculation():
    res = TTCAnchorEngine.calculate_ttc_pd(fundamental_score=-2.5, industry_score=-1.0)
    assert 1 <= res["pd_1y_bps"] <= 9999
    assert res["pd_1y_bps"] <= res["pd_3y_bps"] <= res["pd_5y_bps"]


def test_merton_equity_structural():
    pd_bps = EquityStructuralEngine.calculate_merton_pd(
        equity_val=100.0,
        debt_val=80.0,
        equity_vol=0.25,
        r=0.04,
        t=1.0,
    )
    assert isinstance(pd_bps, int)
    assert 1 <= pd_bps <= 9999

    with pytest.raises(ValueError):
        EquityStructuralEngine.calculate_merton_pd(
            equity_val=-100.0,
            debt_val=80.0,
            equity_vol=0.25,
        )


def test_judge_adjudication_regimes():
    engine = MultiHorizonCreditEngine(tolerance_bps=35)
    env = mock_envelope()
    proposals = standard_proposals(ttc_bps=100, ltm_bps=120, fwd_bps=130, eq_bps=140)

    for regime in ["EXPANSION", "NORMAL", "SLOWDOWN", "RECESSION", "CREDIT_STRESS", "CRISIS"]:
        res = engine.adjudicate(env, proposals, regime, macro_stress_factor=1.0)
        assert res["circuit_breaker_status"] == "OPERATIONAL"
        assert res["physical_pd_1y_bps"] > 0
        assert res["par_cds_spread_bps"] > 0


def test_cds_pricing_bootstrapping():
    engine = MultiHorizonCreditEngine()
    env = mock_envelope()
    res = engine.adjudicate(env, standard_proposals(), "NORMAL", macro_stress_factor=1.0, cds_recovery_rate=0.40)
    assert res["par_cds_spread_bps"] > 0
    assert res["risk_neutral_hazard_bps"] > 0


def test_ev_feedback_loop():
    engine = MultiHorizonCreditEngine()
    fcf = [10.0, 12.0, 15.0, 18.0, 20.0]
    res = engine.evaluate_enterprise_feedback(
        fcf_projections=fcf,
        net_debt=30.0,
        market_equity_val=100.0,
        par_cds_spread_bps=250,
    )
    assert "enterprise_value" in res
    assert "implied_equity_value" in res
    assert "trigger_volatility_feedback" in res


def test_fail_closed_non_finite():
    engine = MultiHorizonCreditEngine()
    bad_proposals = standard_proposals()
    bad_proposals[0]["pd_1y_bps"] = float("inf")
    res = engine.adjudicate(mock_envelope(), bad_proposals, "NORMAL", 1.0)
    assert res["circuit_breaker_status"] == "TRIPPED"
    assert res["error_reason"] == "NON_FINITE_NUMERIC_INPUT_DETECTED"
