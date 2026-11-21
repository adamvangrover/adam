import pytest
from scripts.adversarial_pd_challenger import calculate_forward_looking_pd

def test_calculate_forward_looking_pd_default():
    result = calculate_forward_looking_pd(
        name="Test",
        ticker="TST",
        current_debt=5000.0,
        short_term_debt=1000.0,
        projected_fcf=200.0,
        implied_volatility=0.40,
        forward_interest_rate=0.05,
        equity_value=3000.0,
        macro_stress_factor=0.85
    )

    assert result["entity"]["name"] == "Test"
    assert result["entity"]["ticker"] == "TST"
    assert "adversarial_pd_challenger" in result
    assert 0.0001 <= result["adversarial_pd_challenger"] <= 0.9999

    # Check intermediate calculations
    assert result["inputs"]["v_firm_stressed"] == (3000.0 + 5000.0) * 0.85
    assert "structural_pd" in result["components"]
    assert "liquidity_pd" in result["components"]

def test_calculate_forward_looking_pd_zero_stress():
    result = calculate_forward_looking_pd(
        name="Test2",
        ticker="TST2",
        current_debt=5000.0,
        short_term_debt=1000.0,
        projected_fcf=200.0,
        implied_volatility=0.40,
        forward_interest_rate=0.05,
        equity_value=3000.0,
        macro_stress_factor=0.0
    )

    assert result["adversarial_pd_challenger"] == 0.9999

def test_calculate_forward_looking_pd_negative_refi():
    result = calculate_forward_looking_pd(
        name="Test3",
        ticker="TST3",
        current_debt=5000.0,
        short_term_debt=100.0,  # low short term debt
        projected_fcf=200.0,    # higher fcf than short term debt
        implied_volatility=0.40,
        forward_interest_rate=0.05,
        equity_value=3000.0,
        macro_stress_factor=0.85
    )

    assert result["components"]["liquidity_pd"] == 0.0
