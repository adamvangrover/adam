import pytest
from adam_v3.kernel.schema import AgentInput, AgentOutput


def test_provenance_trace_optional():
    output = AgentOutput(status='success', result={'key': 'value'})
    assert output.provenance_trace is None


def test_provenance_trace_set():
    output = AgentOutput(status='success', result={'key': 'value'},
        provenance_trace='trace_123')
    assert output.provenance_trace == 'trace_123'


def check_grounding(output: AgentOutput):
    assert output.provenance_trace is not None
