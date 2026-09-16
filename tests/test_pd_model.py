import pytest
from scripts.realtime_pd_model import assess_credit_risk

def test_assess_credit_risk():
    context = {
        "financials": {"total_assets": 1000, "total_liabilities": 500, "ebitda": 100, "interest_expense": 20},
        "facility": {"lgd": 0.4, "collateral_value": 200, "exposure": 600}
    }
    output = assess_credit_risk("ENT-123", context)
    assert output.confidence == 0.95
    assert output.provenance_trace is not None
    assert output.metadata["entity_id"] == "ENT-123"
    assert "obligor_pd" in output.metadata
    assert "facility_expected_loss" in output.metadata
    assert not output.observed_drift
    assert output.metadata["obligor_pd"] > 0
    assert output.metadata["facility_expected_loss"] > 0
