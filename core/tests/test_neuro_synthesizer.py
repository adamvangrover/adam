import pytest
import asyncio
from core.v30_architecture.neuro_quantum.ontology import SemanticLabel, FIBOConcept, MarketRegime
from core.v30_architecture.neuro_quantum.synthesizer import StateSynthesizer, HolisticStateTuple
from core.v30_architecture.neuro_quantum.liquid_net import LiquidNeuralNetwork
from core.engine.swarm.neuro_worker import NeuroQuantumWorker
from core.engine.swarm.pheromone_board import PheromoneBoard

def test_fibo_mapping():
    assert FIBOConcept.LOAN.value == "fibo-loan-ln-ln:Loan"
    assert MarketRegime.STAGFLATIONARY_DIVERGENCE.value == "Stagflationary_Divergence"

def test_synthesizer_regime_detection():
    synth = StateSynthesizer()

    # 1. Text based
    regime = synth._identify_regime("stagflation divergence", 0.0)
    assert regime == MarketRegime.STAGFLATIONARY_DIVERGENCE

    # 2. Entropy based (High Entropy -> Geopolitical/Chaos)
    regime_chaos = synth._identify_regime("random text", 0.5)
    assert regime_chaos == MarketRegime.GEOPOLITICAL_ESCALATION

def test_synthesizer_fibo_extraction():
    synth = StateSynthesizer()
    prompt = "We have a syndicated loan and a derivative swap"
    concepts = synth._extract_fibo(prompt)

    assert FIBOConcept.SYNDICATED_LOAN in concepts
    assert FIBOConcept.DERIVATIVE in concepts
    assert FIBOConcept.SWAP in concepts

def test_synthesizer_entropy_calc():
    synth = StateSynthesizer()
    net = LiquidNeuralNetwork()
    net.add_neuron("n1")
    # Add synapse with uncertainty 0.2
    net.add_synapse("input", "n1", mean_weight=1.0, uncertainty=0.2)
    # Add synapse with uncertainty 0.4
    net.add_synapse("n1", "n1", mean_weight=0.5, uncertainty=0.4)

    # Average should be (0.2 + 0.4) / 2 = 0.3
    entropy = synth.calculate_quantum_entropy(net)
    assert abs(entropy - 0.3) < 0.0001

@pytest.mark.asyncio
async def test_neuro_worker_holistic_output():
    board = PheromoneBoard()
    worker = NeuroQuantumWorker(board)

    task_data = {
        "id": "sim_holistic",
        "type": "SIMULATION",
        "context": "stagflation and syndicated loan failure",
        "steps": 2
    }

    await worker.execute_task(task_data)

    results = await board.sniff("NEURO_RESULT")
    assert len(results) > 0
    res = results[0].data

    assert "holistic_state" in res
    h_state = res["holistic_state"]

    assert h_state["market_regime"] == MarketRegime.STAGFLATIONARY_DIVERGENCE.value
    assert FIBOConcept.SYNDICATED_LOAN.value in h_state["fibo_concepts"]
    assert "quantum_entropy" in h_state
