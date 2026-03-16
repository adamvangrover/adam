import pytest
import core.intelligence.nssf.mocks # Patch environment first
from core.intelligence.nssf.mocks import torch
from core.intelligence.nssf.synapse import QuantumSynapseLayer
from core.intelligence.nssf.liquid import LiquidSemanticEncoder
from core.intelligence.nssf.framer import FrameCollapser, RefinedExecutionPlan
from core.intelligence.nssf.main import NSSFOrchestrator
from core.risk_engine.quantum_model import QuantumRiskEngine
from core.schemas.v23_5_schema import ExecutionPlan

def test_synapse_layer():
    engine = QuantumRiskEngine()
    # Ensure it uses mock backend
    input_dim = 4
    output_dim = 2
    layer = QuantumSynapseLayer(input_dim, output_dim, backend_engine=engine)

    # Create a mock input (Batch=1, Dim=4)
    x = torch.randn(1, input_dim)
    output = layer(x)

    assert output is not None
    # Matmul (1,4) x (4,2) -> (1,2)
    assert output.shape == (1, output_dim)

def test_liquid_encoder():
    engine = QuantumRiskEngine()
    encoder = LiquidSemanticEncoder(input_dim=10, hidden_dim=5, backend_engine=engine)

    plan = ExecutionPlan(
        plan_id="test_plan",
        steps=[],
        estimated_complexity="LOW"
    )

    state = encoder.encode_plan(plan)
    assert state is not None
    assert state.shape == (1, 5)

    # Test Determinism
    state2 = encoder.encode_plan(plan)
    # Check if tensors are equal (mock check)
    assert str(state) == str(state2) # Basic check for mock

def test_frame_collapser():
    collapser = FrameCollapser()
    plan = ExecutionPlan(
        plan_id="test_plan",
        steps=[],
        estimated_complexity="LOW"
    )

    # Test 1: Low prob -> HOLD
    result = collapser.collapse(plan, 0.4, {})
    assert result.collapser_verdict == "HOLD"

    # Test 2: High prob -> EXECUTE
    result = collapser.collapse(plan, 0.6, {})
    assert result.collapser_verdict == "EXECUTE"

    # Test 3: Constraint Logic (Volatility Gate)
    # Volatility > 0.5 triggers ABORT
    result = collapser.collapse(plan, 0.9, {"volatility": 0.6})
    assert result.collapser_verdict == "ABORT"
    assert "VOLATILITY_GATE" in result.applied_constraints

    # Test 4: Financial Rules (Liquidity)
    result = collapser.collapse(plan, 0.9, {"liquidity_risk": "High"})
    assert result.collapser_verdict == "ABORT"
    assert "LIQUIDITY_CRUNCH_ABORT" in result.applied_constraints

def test_orchestrator():
    nssf = NSSFOrchestrator()
    # Mocks ensure this runs without real Graph/Risk engines
    result = nssf.process_query("Test Query")

    assert isinstance(result, RefinedExecutionPlan)
    assert result.plan_id is not None
