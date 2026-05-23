import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from core.pdil.integration_layer import GovernanceGatekeeper, GovernanceError

VALID_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "score": {"type": "number"}
    },
    "required": ["analysis", "score"]
}

# Generators for ProvenanceHeader fields
git_hashes = st.text(alphabet="0123456789abcdef", min_size=40, max_size=40)
timestamps = st.text(min_size=1) # simplified ISO 8601 mock
content_hashes = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)
versions = st.text(min_size=1)
paths = st.text(min_size=1)

@given(
    st.dictionaries(
        keys=st.sampled_from(["analysis", "score"]),
        values=st.one_of(st.text(), st.integers()),
    ),
    st.floats(min_value=-10.0, max_value=10.0)
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_governance_gatekeeper_fuzz(payload, confidence):
    gatekeeper = GovernanceGatekeeper(schema=VALID_SCHEMA)

    inference_output = {
        "provenance_trace": {
            "git_commit_hash": "a"*40,
            "timestamp": "2023-10-27T10:00:00Z",
            "content_hash": "b"*64,
            "jsonLogic_version": "1.0",
            "confidence_score": confidence,
            "derivation_path": "test_path"
        },
        "data": payload
    }

    # It must raise GovernanceError if the data is poisoned (confidence score out of bounds)
    # or if the payload fails schema validation
    is_valid_payload = isinstance(payload.get("analysis"), str) and isinstance(payload.get("score"), (int, float)) and "analysis" in payload and "score" in payload
    is_valid_confidence = 0.0 <= confidence <= 1.0

    if not is_valid_payload or not is_valid_confidence:
        with pytest.raises(GovernanceError):
            gatekeeper.validate_inference(inference_output)
    else:
        # Should pass
        result = gatekeeper.validate_inference(inference_output)
        assert result == inference_output

def test_missing_provenance_trace():
    gatekeeper = GovernanceGatekeeper(schema=VALID_SCHEMA)
    inference_output = {
        "data": {"analysis": "Looks good", "score": 0.8}
    }

    with pytest.raises(GovernanceError, match="Missing 'provenance_trace'"):
        gatekeeper.validate_inference(inference_output)

def test_invalid_provenance_trace():
    gatekeeper = GovernanceGatekeeper(schema=VALID_SCHEMA)
    inference_output = {
        "provenance_trace": {
            "git_commit_hash": "a"*40,
            # missing timestamp
            "content_hash": "b"*64,
            "jsonLogic_version": "1.0",
            "confidence_score": 0.9,
            "derivation_path": "test_path"
        },
        "data": {"analysis": "Looks good", "score": 0.8}
    }

    with pytest.raises(GovernanceError, match="Invalid ProvenanceHeader"):
        gatekeeper.validate_inference(inference_output)

def test_valid_inference():
    gatekeeper = GovernanceGatekeeper(schema=VALID_SCHEMA)
    inference_output = {
        "provenance_trace": {
            "git_commit_hash": "a"*40,
            "timestamp": "2023-10-27T10:00:00Z",
            "content_hash": "b"*64,
            "jsonLogic_version": "1.0",
            "confidence_score": 0.9,
            "derivation_path": "test_path"
        },
        "data": {"analysis": "Looks good", "score": 0.8}
    }

    result = gatekeeper.validate_inference(inference_output)
    assert result == inference_output
