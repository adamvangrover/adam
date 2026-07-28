import pytest
from src.schemas.core_types import AgentInput, AgentOutput
from src.pdil.models import ProvenanceHeader

def test_agent_input_instantiation():
    """Test AgentInput correctly handles data payload without legacy fields."""
    input_data = AgentInput(
        data={"metric": "value"},
        context="System simulation context"
    )
    assert input_data.data == {"metric": "value"}
    assert input_data.context == "System simulation context"

def test_agent_output_instantiation_and_grounding():
    """Test AgentOutput correctly uses ProvenanceHeader and validates W3C PROV-O."""
    header = ProvenanceHeader(
        git_commit_hash="abcdef123456",
        timestamp="2023-10-25T12:00:00Z",
        content_hash="hash123",
        jsonLogic_version="1.0",
        confidence_score=0.95,
        derivation_path="sim -> calc -> output",
        source_data_object="doc_456"
    )

    output_data = AgentOutput(
        provenance_trace=header,
        data={"result": "success"},
        observed_drift=False
    )

    assert output_data.data == {"result": "success"}
    assert output_data.observed_drift is False
    assert output_data.check_grounding() is True

def test_agent_output_failing_grounding():
    """Test check_grounding fails when source_data_object is missing/empty."""
    header = ProvenanceHeader(
        git_commit_hash="abcdef123456",
        timestamp="2023-10-25T12:00:00Z",
        content_hash="hash123",
        jsonLogic_version="1.0",
        confidence_score=0.95,
        derivation_path="sim -> calc -> output",
        source_data_object="" # Empty string to fail grounding
    )

    output_data = AgentOutput(
        provenance_trace=header,
        data={"result": "success"},
        observed_drift=False
    )

    assert output_data.check_grounding() is False
