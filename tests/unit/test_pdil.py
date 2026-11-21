import pytest
import hashlib
from src.pdil.models import ProvenanceHeader
from src.schemas.core_types import compute_deterministic_hash


def test_provenance_header_hash_state():
    header = ProvenanceHeader(
        git_commit_hash="abc123def456",
        timestamp="2023-10-25T12:00:00Z",
        content_hash="dummy_hash",
        jsonLogic_version="1.0.0",
        confidence_score=0.95,
        derivation_path="test_path",
        source_data_object="test_source"
    )

    expected_hash = hashlib.sha256(b"abc123def456:1.0.0").hexdigest()
    assert header.hash_state() == expected_hash

    # Test stability: same input yields same output
    assert header.hash_state() == expected_hash

def test_compute_deterministic_hash():
    data = {"b": 2, "a": 1}
    expected_json = '{"a":1,"b":2}'
    expected_hash = hashlib.sha256(expected_json.encode('utf-8')).hexdigest()

    assert compute_deterministic_hash(data) == expected_hash

    # Order shouldn't matter
    data2 = {"a": 1, "b": 2}
    assert compute_deterministic_hash(data2) == expected_hash
