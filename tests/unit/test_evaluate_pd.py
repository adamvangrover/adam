import pytest
from scripts.evaluate_pd import DeterministicRiskJudge

def test_evaluate_divergence_critical():
    judge = DeterministicRiskJudge()
    result = judge.evaluate_divergence(
        baseline_yield_t2=5.18,
        challenger_yield_t2=4.72,
        sim_hy_drawdown=-7.50,
        bsl_spread_widening_bps=135.0
    )
    assert result["adjudication_verdict"]["divergence_severity"] == "CRITICAL"
    assert result["adjudication_verdict"]["circuit_breaker_status"] == "TRIPPED_TIER_2_SOFT_STOP"
    assert result["runtime_telemetry"]["absolute_yield_delta_bps"] == pytest.approx(46.0)

def test_evaluate_divergence_normal():
    judge = DeterministicRiskJudge()
    result = judge.evaluate_divergence(
        baseline_yield_t2=5.18,
        challenger_yield_t2=5.15,
        sim_hy_drawdown=-1.0,
        bsl_spread_widening_bps=10.0
    )
    assert result["adjudication_verdict"]["divergence_severity"] == "NORMAL"
    assert result["adjudication_verdict"]["circuit_breaker_status"] == "OPERATIONAL"
    assert result["runtime_telemetry"]["absolute_yield_delta_bps"] == pytest.approx(3.0)
