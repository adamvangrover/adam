import pytest
import skills.epistemic_grounding as grounding


def test_grounding_valid_artifact_reference():
    input_data = {
        "correlation_id": "eval-grounding-test-01",
        "commit_sha": "HEAD",
        "assertions": [
            {
                "claim_id": "c1",
                "text": "The core Rust library defines BasisPoints.",
                "source_refs": ["adam-core/src/lib.rs"]
            }
        ]
    }

    res = grounding.run(input_data)
    assert res["status"] == "PASSED"
    assert res["promotion_allowed"] is True
    assert res["claims"][0]["epistemic_status"] == "VERIFIED"


def test_grounding_missing_artifact_reference():
    input_data = {
        "correlation_id": "eval-grounding-test-02",
        "commit_sha": "HEAD",
        "assertions": [
            {
                "claim_id": "c2",
                "text": "This assertion references a non-existent file.",
                "source_refs": ["does/not/exist.txt"]
            }
        ]
    }

    res = grounding.run(input_data)
    assert res["status"] == "REVIEW_REQUIRED"
    assert res["promotion_allowed"] is False
    assert res["claims"][0]["epistemic_status"] == "UNVERIFIED"
    assert "ARTIFACT_NOT_FOUND:does/not/exist.txt" in res["claims"][0]["reason_codes"]
