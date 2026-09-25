import pytest
from afos.core_types import Obligor, Facility
from afos.engine import calculate_expected_loss, assess_facility_risk, arbitrate_dual_models

def test_calculate_expected_loss():
    obligor = Obligor(
        obligor_id="OBL-001",
        name="Acme Corp",
        probability_of_default=0.05,
        sector="Technology"
    )
    facility = Facility(
        facility_id="FAC-001",
        obligor_id="OBL-001",
        exposure_amount=1_000_000.0,
        loss_given_default=0.40,
        collateral_value=500_000.0
    )

    el = calculate_expected_loss(obligor, facility)
    assert el == pytest.approx(20000.0)

def test_assess_facility_risk():
    obligor = Obligor(
        obligor_id="OBL-001",
        name="Acme Corp",
        probability_of_default=0.02,
        sector="Technology"
    )
    facility = Facility(
        facility_id="FAC-001",
        obligor_id="OBL-001",
        exposure_amount=1_000_000.0,
        loss_given_default=0.40,
        collateral_value=500_000.0
    )

    result = assess_facility_risk(obligor, facility)
    assert result.expected_loss == pytest.approx(8000.0)
    assert result.facility_rating == "AAA"
    assert result.provenance is not None
    assert not result.provenance.observed_drift

def test_obligor_mismatch():
    obligor = Obligor(
        obligor_id="OBL-001",
        name="Acme",
        probability_of_default=0.05,
        sector="Tech"
    )
    facility = Facility(
        facility_id="FAC-001",
        obligor_id="OBL-002",
        exposure_amount=100.0,
        loss_given_default=0.5,
        collateral_value=0.0
    )
    with pytest.raises(ValueError):
        calculate_expected_loss(obligor, facility)

def test_assess_facility_risk_unrated():
    obligor = Obligor(
        obligor_id="OBL-001",
        name="Acme",
        probability_of_default=0.05,
        sector="Tech"
    )
    facility = Facility(
        facility_id="FAC-001",
        obligor_id="OBL-001",
        exposure_amount=0.0,
        loss_given_default=0.5,
        collateral_value=0.0
    )
    result = assess_facility_risk(obligor, facility)
    assert result.facility_rating == "UNRATED"

def test_assess_facility_risk_bbb_bb_c():
    obligor = Obligor(
        obligor_id="OBL-001",
        name="Acme",
        probability_of_default=1.0,
        sector="Tech"
    )
    facility = Facility(
        facility_id="FAC-001",
        obligor_id="OBL-001",
        exposure_amount=100.0,
        loss_given_default=0.03,
        collateral_value=0.0
    )
    result = assess_facility_risk(obligor, facility)
    assert result.facility_rating == "BBB"

    facility.loss_given_default = 0.08
    result = assess_facility_risk(obligor, facility)
    assert result.facility_rating == "BB"

    facility.loss_given_default = 0.15
    result = assess_facility_risk(obligor, facility)
    assert result.facility_rating == "C"

def test_arbitrate_dual_models_no_flag():
    result = arbitrate_dual_models(pd_alpha=0.10, pd_beta=0.12, theta=0.050, lambda_penalty=1.5)
    assert result.bidirectional_disparity == pytest.approx(0.02)
    assert result.arbitration_flag_triggered is False
    assert result.one_sided_downside_spread == pytest.approx(0.02)
    assert result.capital_buffer_penalty == pytest.approx(0.03)

def test_arbitrate_dual_models_flag_triggered():
    result = arbitrate_dual_models(pd_alpha=0.10, pd_beta=0.18, theta=0.050, lambda_penalty=2.0)
    assert result.bidirectional_disparity == pytest.approx(0.08)
    assert result.arbitration_flag_triggered is True
    assert result.one_sided_downside_spread == pytest.approx(0.08)
    assert result.capital_buffer_penalty == pytest.approx(0.16)

def test_arbitrate_dual_models_downside_spread():
    result = arbitrate_dual_models(pd_alpha=0.15, pd_beta=0.10, theta=0.050, lambda_penalty=1.0)
    assert result.bidirectional_disparity == pytest.approx(0.05)
    assert result.arbitration_flag_triggered is False
    assert result.one_sided_downside_spread == pytest.approx(0.0)
    assert result.capital_buffer_penalty == pytest.approx(0.0)
