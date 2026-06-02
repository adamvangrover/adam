import pytest
import os
import sys

# Ensure project root is in PYTHONPATH for testing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from adam_finance.math import calculate_option_greeks, run_monte_carlo_pipeline
from adam_interfaces.protocols import SwarmToGraphBoundary, SwarmOutput, GraphInput
from adam_swarm.orchestrator import FastMCPToolMapper
from adam_governance.state_control import ProofOfThoughtLogger

def test_calculate_option_greeks():
    result = calculate_option_greeks(100.0, 100.0, 1.0, 0.2, 0.05)
    assert isinstance(result, dict)
    assert "delta" in result
    assert result["delta"] == 0.5

def test_run_monte_carlo_pipeline():
    result = run_monte_carlo_pipeline(100.0, 5, 1)
    assert isinstance(result, list)
    assert len(result) == 5
    assert result[0] == 105.0

def test_fastmcp_tool_mapper():
    mapper = FastMCPToolMapper()
    mapper.map_tool("test_tool", {"type": "string"})
    schema = mapper.get_schema("test_tool")
    assert schema == {"type": "string"}

    missing = mapper.get_schema("missing_tool")
    assert missing == {}

def test_proof_of_thought_logger():
    logger = ProofOfThoughtLogger()
    payload = {"key": "value"}
    result = logger.log_payload(payload)

    assert result["entity"] == "LLM_Output"
    assert result["wasGeneratedBy"] == "Adam_Swarm_Agent"
    assert "content_hash" in result
    assert result["payload"] == payload

def test_swarm_to_graph_boundary_abstract():
    # Attempting to instantiate the ABC should raise TypeError
    with pytest.raises(TypeError):
        SwarmToGraphBoundary()

    class ConcreteBoundary(SwarmToGraphBoundary):
        def transform(self, swarm_output: SwarmOutput) -> GraphInput:
            class MockGraphInput:
                nodes = {"A": swarm_output.data}
                edges = []
            return MockGraphInput()

    class MockSwarmOutput:
        data = "test_data"
        confidence = 0.99

    boundary = ConcreteBoundary()
    result = boundary.transform(MockSwarmOutput())
    assert result.nodes["A"] == "test_data"

from adam_governance.state_control import MilestoneLogger

def test_milestone_logger():
    logger = MilestoneLogger()
    # Add milestone with high conviction and low complexity
    logger.add_milestone("Process A", {}, complexity=0.1, conviction=0.9)
    # Add milestone with low conviction and high complexity
    logger.add_milestone("Process B", {}, complexity=0.8, conviction=0.2)

    best_process = logger.get_most_efficient_process()
    assert best_process["name"] == "Process A"
    assert "timestamp" in best_process

from adam_finance.math import calculate_margin_of_error, evaluate_iteration_need

def test_margin_of_error():
    moe = calculate_margin_of_error(10.0, 100)
    assert round(moe, 2) == 1.96

    need_iter = evaluate_iteration_need(moe, 1.0)
    assert need_iter is True

    no_iter = evaluate_iteration_need(0.01, 0.05)
    assert no_iter is False
