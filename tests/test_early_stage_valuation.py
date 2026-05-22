import pytest
from pydantic import ValidationError
from src.early_stage_valuation import EarlyStageModel

def test_early_stage_model_initialization():
    model = EarlyStageModel(
        current_cash=1000000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    assert model.current_cash == 1000000
    assert model.monthly_burn_rate == 50000
    assert model.projected_ebitda_positive_months == 12

def test_early_stage_model_validation():
    with pytest.raises(ValidationError):
        EarlyStageModel(
            current_cash=-1000000,
            monthly_burn_rate=50000,
            projected_ebitda_positive_months=12,
            target_ebitda=2000000,
            ebitda_multiple=8,
            discount_rate=0.15,
            total_debt=500000
        )

def test_calculate_liquidity_runway():
    model = EarlyStageModel(
        current_cash=1000000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    assert model.calculate_liquidity_runway() == 20.0
    assert model.liquidity_runway == 20.0

def test_calculate_implied_valuation():
    model = EarlyStageModel(
        current_cash=1000000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    valuation = model.calculate_implied_valuation()
    assert valuation["Future Enterprise Value"] == 16000000.0
    # present_ev = 16000000 / ((1 + 0.15) ^ (12 / 12)) = 16000000 / 1.15 = 13913043.48
    assert valuation["Present Enterprise Value"] == 13913043.48

def test_evaluate_ltv_framework():
    model = EarlyStageModel(
        current_cash=1000000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    ltv_metrics = model.evaluate_ltv_framework(1000000)
    assert ltv_metrics["Pro Forma Debt"] == 1500000.0
    # 1500000 / 13913043.48 * 100 = 10.78
    assert ltv_metrics["LTV (%)"] == 10.78

def test_evaluate_ltv_framework_negative_loan():
    model = EarlyStageModel(
        current_cash=1000000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    with pytest.raises(ValueError):
        model.evaluate_ltv_framework(-1000)

def test_check_interim_covenant_protection_pass():
    model = EarlyStageModel(
        current_cash=1000000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    passed, msg = model.check_interim_covenant_protection(500000)
    assert passed is True
    assert "PASS" in msg

def test_check_interim_covenant_protection_breach_cash():
    model = EarlyStageModel(
        current_cash=400000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    passed, msg = model.check_interim_covenant_protection(500000)
    assert passed is False
    assert "BREACH: Current cash (400000.0) is below minimum liquidity covenant" in msg

def test_check_interim_covenant_protection_breach_runway():
    model = EarlyStageModel(
        current_cash=1000000,
        monthly_burn_rate=100000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    passed, msg = model.check_interim_covenant_protection(500000)
    assert passed is False
    assert "WARNING/BREACH: Liquidity runway" in msg

def test_check_interim_covenant_protection_negative_covenant():
    model = EarlyStageModel(
        current_cash=1000000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    with pytest.raises(ValueError):
        model.check_interim_covenant_protection(-1000)

def test_generate_risk_summary_safe():
    model = EarlyStageModel(
        current_cash=1000000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    summary = model.generate_risk_summary(minimum_liquidity_covenant=500000)
    assert "[RISK:SAFE]" in summary

def test_generate_risk_summary_at_risk():
    model = EarlyStageModel(
        current_cash=400000,
        monthly_burn_rate=50000,
        projected_ebitda_positive_months=12,
        target_ebitda=2000000,
        ebitda_multiple=8,
        discount_rate=0.15,
        total_debt=500000
    )
    summary = model.generate_risk_summary(minimum_liquidity_covenant=500000)
    assert "[RISK:AT_RISK]" in summary
