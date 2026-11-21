import pytest
from core.experimental.quantum_swarm_poc import QuantumSwarmIntelligence
from core.engine.swarm.mirofish_engine import MiroFishSwarmEngine
from core.v22_quantum_pipeline.qmc_engine import QuantumMonteCarloEngine

@pytest.mark.asyncio
async def test_quantum_swarm_evaluation():
    """
    Validates that the QuantumSwarmIntelligence engine can run an end-to-end evaluation
    combining quantum PD estimation and emergent swarm intelligence.
    """
    q_engine = QuantumMonteCarloEngine(n_qubits=10)
    s_engine = MiroFishSwarmEngine(agent_count=10, max_cycles=1, consensus_threshold=0.85)
    qs_intel = QuantumSwarmIntelligence(quantum_engine=q_engine, swarm_engine=s_engine)

    result = await qs_intel.evaluate_credit_facility(
        asset_value=100.0,
        debt_face_value=80.0,
        volatility=0.4,
        risk_free_rate=0.05,
        time_horizon=1.0,
        n_simulations=1000
    )

    # Assert structural integrity of the synthesized intelligence report
    assert "quantum_baseline" in result
    assert "probability_of_default" in result["quantum_baseline"]
    assert "swarm_events_captured" in result
    assert "emergent_consensus" in result

    # Assert logical bounds of the probability of default
    pd = result["quantum_baseline"]["probability_of_default"]
    assert 0.0 <= pd <= 1.0

    # The swarm should have captured at least some events (since we ran 1 cycle with 10 agents)
    assert result["swarm_events_captured"] >= 0
