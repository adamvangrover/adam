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


def check_grounding(inference_output: dict):
    if "provenance_trace" not in inference_output or "source_data_object" not in inference_output["provenance_trace"]:
        raise GovernanceError("Output fails W3C PROV-O compliance: missing source_data_object in provenance_trace")
    return True

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
            "derivation_path": "test_path",
            "source_data_object": "fuzz_test_source"
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
            "derivation_path": "test_path",
            "source_data_object": "test_source"
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
            "derivation_path": "test_path",
            "source_data_object": "test_source"
        },
        "data": {"analysis": "Looks good", "score": 0.8}
    }

    result = gatekeeper.validate_inference(inference_output)
    assert result == inference_output

def test_check_grounding():
    # Valid output
    valid_output = {
        "provenance_trace": {
            "source_data_object": "market_data_v1"
        }
    }
    assert check_grounding(valid_output) == True

    # Invalid output (missing source_data_object)
    invalid_output = {
        "provenance_trace": {
            "git_commit_hash": "a"*40
        }
    }
    with pytest.raises(GovernanceError, match="missing source_data_object"):
        check_grounding(invalid_output)

    # Invalid output (missing provenance_trace)
    invalid_output_no_trace = {
        "data": {"score": 1}
    }
    with pytest.raises(GovernanceError, match="missing source_data_object"):
        check_grounding(invalid_output_no_trace)
