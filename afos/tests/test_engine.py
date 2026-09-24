import pytest
from afos.core_types import Obligor, Facility
from afos.engine import calculate_expected_loss, assess_facility_risk

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
