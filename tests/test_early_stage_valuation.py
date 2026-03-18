import pytest

from src.early_stage_valuation import EarlyStageModel


def test_early_stage_model_initialization():
    model = EarlyStageModel(
        current_cash=1000000.0,
        monthly_burn_rate=100000.0,
        projected_ebitda_positive_months=12,
        target_ebitda=5000000.0,
        ebitda_multiple=10.0,
        discount_rate=0.20,
        total_debt=0.0
    )
    assert model.current_cash == 1000000.0
    assert model.monthly_burn_rate == 100000.0
    assert model.projected_ebitda_positive_months == 12
    assert model.target_ebitda == 5000000.0
    assert model.ebitda_multiple == 10.0
    assert model.discount_rate == 0.20
    assert model.total_debt == 0.0

def test_early_stage_model_invalid_initialization():
    # Negative cash
    with pytest.raises(ValueError):
        EarlyStageModel(-100.0, 100000.0, 12, 5000000.0, 10.0, 0.20)
    
    # Non-positive burn rate
    with pytest.raises(ValueError):
        EarlyStageModel(1000000.0, 0.0, 12, 5000000.0, 10.0, 0.20)
        
    with pytest.raises(ValueError):
        EarlyStageModel(1000000.0, -10.0, 12, 5000000.0, 10.0, 0.20)

    # Negative projected months
    with pytest.raises(ValueError):
        EarlyStageModel(1000000.0, 100000.0, -5, 5000000.0, 10.0, 0.20)
        
    # Non-positive target EBITDA
    with pytest.raises(ValueError):
        EarlyStageModel(1000000.0, 100000.0, 12, 0.0, 10.0, 0.20)
        
    with pytest.raises(ValueError):
        EarlyStageModel(1000000.0, 100000.0, 12, -500.0, 10.0, 0.20)

    # Non-positive ebitda multiple
    with pytest.raises(ValueError):
        EarlyStageModel(1000000.0, 100000.0, 12, 5000000.0, 0.0, 0.20)
        
    # Invalid discount rate
    with pytest.raises(ValueError):
        EarlyStageModel(1000000.0, 100000.0, 12, 5000000.0, 10.0, -0.1)
    with pytest.raises(ValueError):
        EarlyStageModel(1000000.0, 100000.0, 12, 5000000.0, 10.0, 1.1)

    # Negative debt
    with pytest.raises(ValueError):
        EarlyStageModel(1000000.0, 100000.0, 12, 5000000.0, 10.0, 0.20, -100.0)


def test_calculate_liquidity_runway():
    model = EarlyStageModel(
        current_cash=1500000.0,
        monthly_burn_rate=150000.0,
        projected_ebitda_positive_months=12,
        target_ebitda=5000000.0,
        ebitda_multiple=10.0,
        discount_rate=0.20
    )
    runway = model.calculate_liquidity_runway()
    assert runway == 10.0

def test_calculate_implied_valuation():
    model = EarlyStageModel(
        current_cash=1000000.0,
        monthly_burn_rate=100000.0,
        projected_ebitda_positive_months=24, # 2 years
        target_ebitda=5000000.0,
        ebitda_multiple=10.0,
        discount_rate=0.20
    )
    val = model.calculate_implied_valuation()
    
    assert val["Future Enterprise Value"] == 50000000.0 # 5M * 10
    
    # Present EV = 50M / (1 + 0.2)^2 = 50M / 1.44 = 34722222.22
    assert val["Present Enterprise Value"] == 34722222.22

def test_evaluate_ltv_framework():
    model = EarlyStageModel(
        current_cash=1000000.0,
        monthly_burn_rate=100000.0,
        projected_ebitda_positive_months=24, # 2 years
        target_ebitda=5000000.0,
        ebitda_multiple=10.0,
        discount_rate=0.20,
        total_debt=5000000.0
    )
    ltv_metrics = model.evaluate_ltv_framework(proposed_loan_amount=5000000.0)
    
    # Total Pro Forma Debt = 5M + 5M = 10M
    assert ltv_metrics["Pro Forma Debt"] == 10000000.0
    
    # Present EV = 34722222.22
    # LTV = 10M / 34.72M = 28.8%
    assert ltv_metrics["LTV (%)"] == 28.8

def test_evaluate_ltv_framework_invalid_loan():
    model = EarlyStageModel(1000000.0, 100000.0, 12, 5000000.0, 10.0, 0.20)
    with pytest.raises(ValueError):
        model.evaluate_ltv_framework(-1000.0)

def test_check_interim_covenant_protection_pass():
    model = EarlyStageModel(
        current_cash=2000000.0,
        monthly_burn_rate=100000.0,
        projected_ebitda_positive_months=12,
        target_ebitda=5000000.0,
        ebitda_multiple=10.0,
        discount_rate=0.20
    )
    # Runway is 20 months, needs 12. Current cash 2M, min is 500k.
    status, msg = model.check_interim_covenant_protection(500000.0)
    assert status is True
    assert "PASS" in msg

def test_check_interim_covenant_protection_breach_min_cash():
    model = EarlyStageModel(
        current_cash=400000.0, # Below 500k min
        monthly_burn_rate=10000.0, # Long runway but low cash balance
        projected_ebitda_positive_months=12,
        target_ebitda=5000000.0,
        ebitda_multiple=10.0,
        discount_rate=0.20
    )
    status, msg = model.check_interim_covenant_protection(500000.0)
    assert status is False
    assert "BREACH: Current cash" in msg

def test_check_interim_covenant_protection_breach_runway():
    model = EarlyStageModel(
        current_cash=1000000.0,
        monthly_burn_rate=100000.0, # Runway = 10 months
        projected_ebitda_positive_months=12, # Needs 12
        target_ebitda=5000000.0,
        ebitda_multiple=10.0,
        discount_rate=0.20
    )
    # Meets min cash, but fails runway
    status, msg = model.check_interim_covenant_protection(500000.0)
    assert status is False
    assert "WARNING/BREACH: Liquidity runway" in msg
    assert "~200000.0" in msg # Short by 2 months * 100k
    
def test_check_interim_covenant_protection_invalid_covenant():
    model = EarlyStageModel(1000000.0, 100000.0, 12, 5000000.0, 10.0, 0.20)
    with pytest.raises(ValueError):
        model.check_interim_covenant_protection(-100.0)
