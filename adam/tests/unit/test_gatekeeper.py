import pytest
from pydantic import ValidationError
from hypothesis import given, strategies as st
from src.schemas.provenance import ProvenanceHeader, AgentOutput
from src.governance.gatekeeper import GovernanceGatekeeper

def check_grounding(output: AgentOutput) -> bool:
    """Helper to verify grounding provenance as per the test suite requirement."""
    return bool(output.provenance.source_data_reference)

def test_provenance_header_hashing():
    header = ProvenanceHeader(
        git_commit_hash="abcdef123456",
        jsonLogic_version="1.2.0",
        source_data_reference="s3://bucket/10k_filing.pdf"
    )
    assert header.hash is not None
    assert isinstance(header.hash, str)
    assert len(header.hash) == 64  # SHA256 length

def test_agent_output_validation():
    # Valid output
    header = ProvenanceHeader(
        git_commit_hash="abcdef123456",
        jsonLogic_version="1.2.0",
        source_data_reference="s3://bucket/10k_filing.pdf"
    )
    output = AgentOutput(
        provenance=header,
        answer="Credit rating downgraded.",
        confidence=0.9,
        sources=["s3://bucket/10k_filing.pdf"]
    )
    assert output.confidence == 0.9
    assert check_grounding(output) is True

    # Invalid output (low confidence)
    with pytest.raises(ValidationError):
        AgentOutput(
            provenance=header,
            answer="Credit rating downgraded.",
            confidence=0.4, # Below 0.5 threshold
            sources=["s3://bucket/10k_filing.pdf"]
        )

@given(st.floats(min_value=0.5, max_value=1.0))
def test_gatekeeper_approval(confidence_score):
    """Property-based test for the Gatekeeper using Hypothesis."""
    gatekeeper = GovernanceGatekeeper(approval_threshold=0.85)

    header = ProvenanceHeader(
        git_commit_hash="test",
        jsonLogic_version="test",
        source_data_reference="test"
    )

    output = AgentOutput(
        provenance=header,
        answer="Test",
        confidence=confidence_score
    )

    # Needs approval if confidence < 0.85
    needs_approval = gatekeeper.require_approval(output)
    if confidence_score < 0.85:
        assert needs_approval is True
    else:
        assert needs_approval is False

def test_gatekeeper_process_request():
    gatekeeper = GovernanceGatekeeper()

    raw_payload = {
        "provenance": {
            "git_commit_hash": "abcd",
            "jsonLogic_version": "1.0",
            "source_data_reference": "ref"
        },
        "answer": "Buy",
        "confidence": 0.95,
        "sources": []
    }

    result = gatekeeper.process_agent_request(raw_payload)
    assert result is not None
    assert result["approved"] is True
    assert result["decision"] == "Buy"
    assert len(gatekeeper.ledger) == 1

def test_gatekeeper_process_quarantined_request():
    gatekeeper = GovernanceGatekeeper()

    raw_payload = {
        "provenance": {
            "git_commit_hash": "abcd",
            "jsonLogic_version": "1.0",
            "source_data_reference": "ref"
        },
        "answer": "Hold",
        "confidence": 0.6, # Between 0.5 and 0.85 -> requires review
        "sources": []
    }

    result = gatekeeper.process_agent_request(raw_payload)
    assert result is not None
    assert result["approved"] is False # Quarantined
    assert result["decision"] == "Hold"
    assert len(gatekeeper.ledger) == 1
