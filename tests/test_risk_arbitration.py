import pytest
from afos.risk.arbitration import ArbitrationEngine


@pytest.fixture
def engine():
    return ArbitrationEngine(divergence_threshold=0.05, penalty_multiplier=1.5)


def test_parity_condition(engine):
    metrics = engine.compute_metrics("OBL-001", 0.03, 0.03)
    assert metrics.bidirectional_divergence == 0.0
    assert metrics.downside_spread == 0.0
    assert not metrics.requires_arbitration
    assert metrics.capital_buffer_penalty == 0.0


def test_beta_greater_than_alpha(engine):
    metrics = engine.compute_metrics("OBL-002", 0.02, 0.09)
    assert metrics.bidirectional_divergence == 0.07
    assert metrics.downside_spread == 0.07
    assert metrics.requires_arbitration is True
    assert metrics.capital_buffer_penalty == round(0.07 * 1.5, 6)


def test_alpha_greater_than_beta(engine):
    # Tests that downside spread zeroes out when Alpha > Beta, but bidirectional flags discrepancy
    metrics = engine.compute_metrics("OBL-003", 0.10, 0.02)
    assert metrics.bidirectional_divergence == 0.08
    assert metrics.downside_spread == 0.0
    assert metrics.requires_arbitration is True
    assert metrics.capital_buffer_penalty == 0.0


def test_invalid_pd_bounds(engine):
    with pytest.raises(ValueError):
        engine.compute_metrics("OBL-ERR", -0.01, 0.05)
    with pytest.raises(ValueError):
        engine.compute_metrics("OBL-ERR", 0.05, 1.05)