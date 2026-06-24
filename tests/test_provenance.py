import pytest
from adam_v3.kernel.schema import AgentInput, AgentOutput
from src.pdil.models import ProvenanceHeader

def test_provenance_trace_set():
    header = ProvenanceHeader(
        git_commit_hash="abc",
        timestamp="2023",
        content_hash="123",
        jsonLogic_version="1.0",
        confidence_score=0.9,
        derivation_path="path",
        source_data_object="source"
    )
    output = AgentOutput(status='success', result={'key': 'value'}, provenance_trace=header)
    assert output.provenance_trace == header

def check_grounding(output: AgentOutput) -> bool:
    return bool(output.provenance_trace.source_data_object)

def test_check_grounding():
    header = ProvenanceHeader(
        git_commit_hash="abc",
        timestamp="2023",
        content_hash="123",
        jsonLogic_version="1.0",
        confidence_score=0.9,
        derivation_path="path",
        source_data_object="source"
    )
    output = AgentOutput(status='success', result={'key': 'value'}, provenance_trace=header)
    assert check_grounding(output) is True

from hypothesis import given, strategies as st

@given(
    status=st.text(),
    key=st.text(),
    val=st.text()
)
def test_check_grounding_property(status, key, val):
    header = ProvenanceHeader(
        git_commit_hash="abc",
        timestamp="2023",
        content_hash="123",
        jsonLogic_version="1.0",
        confidence_score=0.95,
        derivation_path="path",
        source_data_object="http://example.com/data"
    )
    output = AgentOutput(status=status, result={key: val}, provenance_trace=header)
    assert check_grounding(output) is True
