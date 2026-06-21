import pytest


def test_expected_loss_determinism():
    pd = 0.05
    lgd = 0.4
    ead = 1000000
    expected_loss = pd * lgd * ead
    assert expected_loss == pytest.approx(20000.0)
