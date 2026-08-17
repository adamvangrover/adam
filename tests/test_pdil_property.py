import pytest
import hashlib
import json
from hypothesis import given, strategies as st
from unittest.mock import patch, MagicMock
from src.pdil.middleware import (
    DriftIntelligenceLayer,
    SecurityGovernanceGatekeeper,
    JsonLogicGovernanceGatekeeper,
    GovernanceError
)
from src.pdil.models import ProvenanceHeader

# Helper to generate valid base input
def create_valid_input(confidence=0.9, payload=None, source="https://example.com/data"):
    if payload is None:
        payload = {"name": "test", "value": 123}

    payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    computed_hash = hashlib.sha256(payload_json).hexdigest()

    return {
        "provenance_trace": {
            "git_commit_hash": "abc123def456",
            "timestamp": "2023-10-25T12:00:00Z",
            "content_hash": computed_hash,
            "jsonLogic_version": "1.0.0",
            "confidence_score": confidence,
            "derivation_path": "test_path",
            "source_data_object": source
        },
        "data": payload
    }

@given(st.floats(min_value=0.0, max_value=0.4999))
def test_poison_check_fails_on_low_confidence(confidence):
    gatekeeper = JsonLogicGovernanceGatekeeper(rules={"==": [1, 1]})

    inference_output = create_valid_input(confidence=confidence)

    with pytest.raises(GovernanceError, match="Poisoned data detected") as exc_info:
        gatekeeper.validate_inference(inference_output)

    assert exc_info.value.provenance == inference_output.get("provenance_trace")
    assert "Erroneous input provenance:" in str(exc_info.value)

@given(st.floats(min_value=0.5, max_value=1.0))
def test_poison_check_passes_on_high_confidence(confidence):
    gatekeeper = JsonLogicGovernanceGatekeeper(rules={"==": [{"var": "name"}, "test"]})

    inference_output = create_valid_input(confidence=confidence)

    # Should not raise GovernanceError related to confidence
    try:
        gatekeeper.validate_inference(inference_output)
    except GovernanceError as e:
        assert "Poisoned data detected" not in str(e)

@given(st.text(min_size=1))
def test_content_hash_mismatch_fails(bad_hash):
    gatekeeper = JsonLogicGovernanceGatekeeper(rules={"==": [1, 1]})

    inference_output = create_valid_input()
    inference_output["provenance_trace"]["content_hash"] = bad_hash

    with pytest.raises(GovernanceError, match="Provenance violation: content_hash mismatch") as exc_info:
        gatekeeper.validate_inference(inference_output)

    assert exc_info.value.provenance == inference_output.get("provenance_trace")

@patch("socket.gethostbyname")
@patch("urllib.request.build_opener")
def test_security_gatekeeper_valid_source(mock_build_opener, mock_gethostbyname):
    # Mock DNS resolution to return a public IP
    mock_gethostbyname.return_value = "8.8.8.8"

    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_opener = MagicMock()
    mock_opener.open.return_value.__enter__.return_value = mock_response
    mock_build_opener.return_value = mock_opener

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "number"}
        }
    }
    gatekeeper = SecurityGovernanceGatekeeper(schema=schema)

    # Needs an allowed domain
    inference_output = create_valid_input(source="https://api.github.com/data")

    # Should pass without raising exception
    result = gatekeeper.validate_inference(inference_output)
    assert result == inference_output

    # Verify mocks were called
    mock_gethostbyname.assert_called_with("api.github.com")
    mock_build_opener.assert_called()

@patch("socket.gethostbyname")
def test_security_gatekeeper_private_ip_fails(mock_gethostbyname):
    # Mock DNS resolution to return a private IP
    mock_gethostbyname.return_value = "192.168.1.1"

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "number"}
        }
    }
    gatekeeper = SecurityGovernanceGatekeeper(schema=schema)

    inference_output = create_valid_input(source="https://api.github.com/data")

    with pytest.raises(GovernanceError, match="resolves to a private IP") as exc_info:
        gatekeeper.validate_inference(inference_output)

    assert exc_info.value.provenance == inference_output.get("provenance_trace")

@patch("socket.gethostbyname")
@patch("urllib.request.build_opener")
def test_security_gatekeeper_http_error_fails(mock_build_opener, mock_gethostbyname):
    # Mock DNS resolution to return a public IP
    mock_gethostbyname.return_value = "8.8.8.8"

    # Mock HTTP response to return a 404 error
    mock_response = MagicMock()
    mock_response.getcode.return_value = 404
    mock_opener = MagicMock()
    mock_opener.open.return_value.__enter__.return_value = mock_response
    mock_build_opener.return_value = mock_opener

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "number"}
        }
    }
    gatekeeper = SecurityGovernanceGatekeeper(schema=schema)

    inference_output = create_valid_input(source="https://api.github.com/data")

    with pytest.raises(GovernanceError, match="Source data object unreachable: HTTP 404") as exc_info:
        gatekeeper.validate_inference(inference_output)

    assert exc_info.value.provenance == inference_output.get("provenance_trace")

@patch("socket.gethostbyname")
@patch("urllib.request.build_opener")
@given(
    payload=st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.integers()
    ).filter(lambda x: len(x) > 0)
)
def test_security_gatekeeper_schema_validation(mock_build_opener, mock_gethostbyname, payload):
    # Mock DNS resolution to return a public IP
    mock_gethostbyname.return_value = "8.8.8.8"

    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_opener = MagicMock()
    mock_opener.open.return_value.__enter__.return_value = mock_response
    mock_build_opener.return_value = mock_opener
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "valid_key": {"type": "string"}
        }
    }
    gatekeeper = SecurityGovernanceGatekeeper(schema=schema)

    inference_output = create_valid_input(payload=payload)

    with pytest.raises(GovernanceError, match="Schema validation failed") as exc_info:
        gatekeeper.validate_inference(inference_output)

    assert exc_info.value.provenance == inference_output.get("provenance_trace")


@st.composite
def inference_outputs(draw):
    payload = draw(st.dictionaries(keys=st.text(min_size=1), values=st.integers() | st.text()))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    source = draw(st.sampled_from(["https://example.com/data", "https://api.github.com/data", "http://bad.com"]))

    return create_valid_input(confidence=confidence, payload=payload, source=source)


@given(inference_outputs())
def test_drift_intelligence_layer_detects_and_heals(inference_output):
    # Setup mock gatekeeper to just return input back
    mock_gatekeeper = MagicMock()
    mock_gatekeeper.validate_inference.return_value = inference_output

    layer = DriftIntelligenceLayer(mock_gatekeeper)

    # Intentionally use a bad hash to trigger drift
    historical_hash = "invalid_hash_to_force_drift"

    # Layer should detect drift, heal it, and call gatekeeper
    result = layer.detect_and_heal_drift(inference_output, historical_hash)

    # Healing means 'observed_drift' is False and 'revalidation_triggered' is True
    assert result.get("observed_drift") is False
    assert result.get("revalidation_triggered") is True
    mock_gatekeeper.validate_inference.assert_called_once()
