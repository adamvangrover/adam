import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import jsonschema
from typing import Dict, Any
import asyncio

from src.governance.gatekeeper import GovernanceGatekeeper, GovernanceError, ProvenanceHeader

# Sample schema for testing
TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "value": {"type": "number"}
    },
    "required": ["status", "value"]
}

def check_grounding(inference_output: Dict[str, Any]) -> bool:
    """
    Test helper to verify W3C PROV-O compliance (grounding).
    Checks that every output contains a reference to its source data object.
    """
    try:
        provenance = inference_output.get("provenance_trace", {})
        header = ProvenanceHeader(**provenance)
        return bool(header.source_data_object)
    except Exception:
        return False

def get_valid_provenance():
    return {
        "git_commit_hash": "abcdef1234567890",
        "timestamp": "2023-10-27T10:00:00Z",
        "content_hash": "hash123",
        "jsonLogic_version": "1.0",
        "confidence_score": 0.95,
        "derivation_path": "/path/to/data",
        "source_data_object": "data_id_123"
    }

@pytest.fixture
def valid_provenance():
    return get_valid_provenance()

def test_valid_inference(valid_provenance):
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    valid_input = {
        "provenance_trace": valid_provenance,
        "data": {"status": "ok", "value": 42}
    }

    assert check_grounding(valid_input)
    result = gatekeeper.validate_inference(valid_input)
    assert result == valid_input

def test_missing_provenance():
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    invalid_input = {
        "data": {"status": "ok", "value": 42}
    }

    with pytest.raises(GovernanceError, match="Missing 'provenance_trace'"):
        gatekeeper.validate_inference(invalid_input)

def test_invalid_schema(valid_provenance):
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    invalid_input = {
        "provenance_trace": valid_provenance,
        "data": {"status": "ok"} # missing 'value'
    }

    with pytest.raises(GovernanceError, match="Schema validation failed"):
        gatekeeper.validate_inference(invalid_input)

def test_poisoned_data_low(valid_provenance):
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    provenance = valid_provenance.copy()
    provenance["confidence_score"] = -0.1
    invalid_input = {
        "provenance_trace": provenance,
        "data": {"status": "ok", "value": 42}
    }

    with pytest.raises(GovernanceError, match="Poisoned data detected"):
        gatekeeper.validate_inference(invalid_input)

def test_poisoned_data_high(valid_provenance):
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    provenance = valid_provenance.copy()
    provenance["confidence_score"] = 1.1
    invalid_input = {
        "provenance_trace": provenance,
        "data": {"status": "ok", "value": 42}
    }

    with pytest.raises(GovernanceError, match="Poisoned data detected"):
        gatekeeper.validate_inference(invalid_input)

def test_missing_data(valid_provenance):
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    invalid_input = {
        "provenance_trace": valid_provenance,
    }

    with pytest.raises(GovernanceError, match="Missing 'data' payload"):
        gatekeeper.validate_inference(invalid_input)

@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    status=st.text(),
    value=st.floats(allow_nan=False, allow_infinity=False)
)
def test_fuzz_valid_schema(status, value):
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    fuzzed_input = {
        "provenance_trace": get_valid_provenance(),
        "data": {"status": status, "value": value}
    }

    assert check_grounding(fuzzed_input)
    result = gatekeeper.validate_inference(fuzzed_input)
    assert result == fuzzed_input

@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    confidence=st.one_of(
        st.floats(max_value=-0.0001, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0001, allow_nan=False, allow_infinity=False)
    )
)
def test_fuzz_poisoned_data(confidence):
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    provenance = get_valid_provenance()
    provenance["confidence_score"] = confidence
    invalid_input = {
        "provenance_trace": provenance,
        "data": {"status": "ok", "value": 42}
    }

    with pytest.raises(GovernanceError, match="Poisoned data detected"):
        gatekeeper.validate_inference(invalid_input)

def test_heal_drift(valid_provenance):
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    input_with_drift = {
        "provenance_trace": valid_provenance,
        "data": {"status": "ok", "value": 42},
        "observed_drift": True
    }
    result = gatekeeper.heal_drift(input_with_drift)
    assert result["observed_drift"] is False
    assert result["revalidation_triggered"] is True

@pytest.mark.asyncio
async def test_async_validate_inference(valid_provenance):
    gatekeeper = GovernanceGatekeeper(schema=TEST_SCHEMA)
    valid_input = {
        "provenance_trace": valid_provenance,
        "data": {"status": "ok", "value": 42}
    }
    result = await gatekeeper.async_validate_inference(valid_input)
    assert result == valid_input
