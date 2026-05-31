import pytest
from adam_finance.math import calculate_leverage, check_covenant_compliance

def test_calculate_leverage():
    # Normal case
    assert calculate_leverage(500.0, 100.0) == 5.0
    # Zero EBITDA case
    assert calculate_leverage(500.0, 0.0) == 999.9

def test_check_covenant_compliance():
    # Max covenant pass
    assert check_covenant_compliance(4.0, 5.0, "max") is True
    # Max covenant fail
    assert check_covenant_compliance(6.0, 5.0, "max") is False
    # Min covenant pass
    assert check_covenant_compliance(1.5, 1.2, "min") is True
    # Min covenant fail
    assert check_covenant_compliance(1.0, 1.2, "min") is False
    # Unknown type
    assert check_covenant_compliance(1.0, 1.2, "unknown") is False
