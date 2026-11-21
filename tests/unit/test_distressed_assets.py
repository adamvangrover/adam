import pytest
from pydantic import ValidationError
from core.schemas.distressed_assets import SupplyChainDependency, DistressTrigger, SectorOracleInput

def test_supply_chain_dependency_valid():
    scd = SupplyChainDependency(
        company_id="comp_A",
        supplier_id="supp_B",
        criticality_score=0.9,
        dependency_type="Semiconductors"
    )
    assert scd.company_id == "comp_A"
    assert scd.supplier_id == "supp_B"
    assert scd.criticality_score == 0.9

def test_supply_chain_dependency_invalid_score():
    with pytest.raises(ValidationError):
        SupplyChainDependency(
            company_id="comp_A",
            supplier_id="supp_B",
            criticality_score=1.5,  # Score should be between 0.0 and 1.0
            dependency_type="Semiconductors"
        )

def test_distress_trigger_valid():
    dt = DistressTrigger(
        trigger_id="trig_1",
        trigger_type="Covenant Breach",
        severity="High",
        description="Debt to EBITDA ratio exceeded 5.0x",
        affected_entities=["comp_A", "comp_B"]
    )
    assert dt.trigger_id == "trig_1"
    assert dt.severity == "High"
    assert "comp_A" in dt.affected_entities

def test_distress_trigger_invalid():
    with pytest.raises(ValidationError):
        DistressTrigger(
            trigger_id="trig_1",
            trigger_type="Covenant Breach",
            severity="High",
            # Missing description
        )

def test_sector_oracle_input_valid():
    soi = SectorOracleInput(
        sector="TMT",
        monitoring_parameters={"max_leverage": 6.0},
        target_entities=["comp_X", "comp_Y"]
    )
    assert soi.sector == "TMT"
    assert soi.monitoring_parameters["max_leverage"] == 6.0

def test_sector_oracle_input_invalid():
    with pytest.raises(ValidationError):
        SectorOracleInput(
            # Missing sector
            monitoring_parameters={"max_leverage": 6.0}
        )
