import pytest
import numpy as np
import asyncio
from unittest.mock import MagicMock, AsyncMock
from core.v30_architecture.neuro_quantum.synapse import QuantumSynapse
from core.v30_architecture.neuro_quantum.liquid_net import LiquidNeuralNetwork
from core.v30_architecture.neuro_quantum.framer import NeuroSymbolicFramer
from core.engine.swarm.neuro_worker import NeuroQuantumWorker
from core.engine.swarm.pheromone_board import PheromoneBoard

def test_quantum_synapse_measurement():
    """Verify that the synapse returns variable weights centered around mean."""
    synapse = QuantumSynapse(weight_mean=1.0, uncertainty=0.1)

    measurements = [synapse.measure() for _ in range(100)]
    mean_val = np.mean(measurements)
    std_val = np.std(measurements)

    # Check that the mean is close to 1.0
    assert 0.9 < mean_val < 1.1
    # Check that there is variance (stochastic behavior)
    assert std_val > 0.05

def test_synapse_update():
    """Verify parameter updates."""
    synapse = QuantumSynapse(weight_mean=0.5, uncertainty=0.1)
    synapse.update_parameters(new_mean=2.0, new_uncertainty=0.01)

    assert synapse.weight_mean == 2.0
    assert synapse.uncertainty == 0.01

def test_liquid_network_integration():
    """Verify that network steps evolve state."""
    net = LiquidNeuralNetwork()
    net.add_neuron("n1", tau=1.0, bias=0.0)
    # Add input connection
    net.add_synapse("input_a", "n1", mean_weight=1.0, uncertainty=0.0)

    # Run one step with input_a = 1.0
    # dx/dt = -(0 - 0)/1 + tanh(1*1) = tanh(1) approx 0.76
    # state = 0 + 0.76 * 0.1 = 0.076
    states = net.compute_step({"input_a": 1.0}, dt=0.1)

    assert "n1" in states
    assert states["n1"] > 0.0
    assert states["n1"] < 0.1 # approximate check

def test_framer_context():
    framer = NeuroSymbolicFramer()
    inputs = framer.frame_context("The market is in Crisis mode.")

    # "crisis" maps to "input_risk" -> 1.0
    assert inputs.get("input_risk") == 1.0
    assert inputs.get("input_stability") == 0.0

def test_framer_overlay():
    net = LiquidNeuralNetwork()
    net.add_neuron("n1", tau=1.0)
    net.add_synapse("input_a", "n1", mean_weight=0.5, uncertainty=0.5)

    framer = NeuroSymbolicFramer()

    # Script to reduce uncertainty (determinism)
    script = {
        "name": "Safety Protocol",
        "rules": [
            {
                "target_neuron": "n1",
                "action": "reduce_uncertainty",
                "value": 0.001
            }
        ]
    }

    framer.overlay_script(net, script)

    # Check if uncertainty was reduced
    synapse = net.neurons["n1"].incoming_synapses["input_a"]
    assert synapse.uncertainty == 0.001

@pytest.mark.asyncio
async def test_neuro_worker_task():
    """Verify worker execution flow."""
    board = PheromoneBoard()
    # Mock deposit to avoid real locking/decay logic overhead if possible,
    # but PheromoneBoard is simple enough to run real.

    worker = NeuroQuantumWorker(board)

    task_data = {
        "id": "task_123",
        "context": "growth and stability",
        "steps": 5,
        "dt": 0.1
    }

    await worker.execute_task(task_data)

    # Check if result was deposited
    results = await board.sniff("NEURO_RESULT")
    assert len(results) > 0
    res_data = results[0].data

    assert res_data["task_id"] == "task_123"
    assert "final_state" in res_data
