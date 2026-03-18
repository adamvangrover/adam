import pytest

from src.config import RATING_MAP
from src.credit_risk import CreditSponsorModel


def test_credit_sponsor_initialization():
    model = CreditSponsorModel(enterprise_value=1000.0, total_debt=500.0, ebitda=100.0, interest_expense=20.0)
    assert model.ev == 1000.0
    assert model.debt == 500.0
    assert model.ebitda == 100.0
    assert model.interest == 20.0

def test_credit_sponsor_invalid_initialization():
    with pytest.raises(ValueError):
        CreditSponsorModel(enterprise_value=-1000.0, total_debt=500.0, ebitda=100.0, interest_expense=20.0)
    with pytest.raises(ValueError):
        CreditSponsorModel(enterprise_value=1000.0, total_debt=-500.0, ebitda=100.0, interest_expense=20.0)
    with pytest.raises(ValueError):
        CreditSponsorModel(enterprise_value=1000.0, total_debt=500.0, ebitda=-100.0, interest_expense=20.0)
    with pytest.raises(ValueError):
        CreditSponsorModel(enterprise_value=1000.0, total_debt=500.0, ebitda=100.0, interest_expense=-20.0)

def test_calculate_metrics():
    model = CreditSponsorModel(enterprise_value=1000.0, total_debt=500.0, ebitda=100.0, interest_expense=20.0)
    metrics = model.calculate_metrics()

    assert metrics["Leverage (x)"] == 5.0
    assert metrics["LTV (%)"] == 50.0
    assert metrics["FCCR (x)"] == 4.5 # (100 - 10) / 20 = 4.5

def test_determine_regulatory_rating():
    model = CreditSponsorModel(enterprise_value=1000.0, total_debt=500.0, ebitda=100.0, interest_expense=20.0)

    # Simulate high quality metrics
    metrics = {"Leverage (x)": 2.5, "LTV (%)": 25.0, "FCCR (x)": 3.0}
    rating = model.determine_regulatory_rating(metrics)
    assert rating == RATING_MAP[3.0] # IG

    # Simulate poor quality metrics
    metrics = {"Leverage (x)": 8.0, "LTV (%)": 80.0, "FCCR (x)": 0.5}
    rating = model.determine_regulatory_rating(metrics)
    assert rating == RATING_MAP[8.0] # Substandard

def test_determine_regulatory_rating_missing_keys():
    model = CreditSponsorModel(enterprise_value=1000.0, total_debt=500.0, ebitda=100.0, interest_expense=20.0)
    metrics = {"Leverage (x)": 2.5, "LTV (%)": 25.0}
    with pytest.raises(KeyError):
        model.determine_regulatory_rating(metrics)

def test_perform_downside_stress():
    model = CreditSponsorModel(enterprise_value=1000.0, total_debt=500.0, ebitda=100.0, interest_expense=20.0)
    metrics, rating = model.perform_downside_stress(stress_factor=0.20)

    # 100 * (1 - 0.20) = 80 EBITDA
    assert metrics["Leverage (x)"] == 6.25 # 500 / 80 = 6.25
    assert metrics["LTV (%)"] == 62.5 # 500 / 800 = 62.5%
    assert metrics["FCCR (x)"] == 3.6 # (80 * 0.9) / 20 = 3.6

    # Lev: 6.25 < 7.5 and FCCR: 3.6 > 1.0 => B- (6.0)
    assert rating == RATING_MAP[6.0]

def test_perform_downside_stress_invalid_factor():
    model = CreditSponsorModel(enterprise_value=1000.0, total_debt=500.0, ebitda=100.0, interest_expense=20.0)
    with pytest.raises(ValueError):
        model.perform_downside_stress(stress_factor=1.5)

def test_snc_check():
    model_non_snc = CreditSponsorModel(enterprise_value=1000.0, total_debt=50_000_000, ebitda=100.0, interest_expense=20.0)
    model_snc = CreditSponsorModel(enterprise_value=1000.0, total_debt=150_000_000, ebitda=100.0, interest_expense=20.0)

    assert model_non_snc.snc_check() == "Non-SNC"
    assert model_snc.snc_check() == "SNC REPORTING REQUIRED: Flag for Regulatory Review"
