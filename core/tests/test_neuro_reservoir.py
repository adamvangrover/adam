import pytest
import numpy as np
import asyncio
from core.v30_architecture.neuro_quantum.ontology import SemanticLabel, EnvironmentDefinition
from core.v30_architecture.neuro_quantum.reservoir import QuantumReservoirClassifier
from core.v30_architecture.neuro_quantum.trainer import LivePromptTrainer
from core.engine.swarm.neuro_worker import NeuroQuantumWorker
from core.engine.swarm.pheromone_board import PheromoneBoard

def test_ontology_loading():
    """Verify ontology structures."""
    env = EnvironmentDefinition(name="HighVol", noise_level=0.5)
    assert env.noise_level == 0.5
    assert SemanticLabel.RISK.value == "RISK"

def test_reservoir_training_flow():
    """Verify that the reservoir can fit and predict."""
    clf = QuantumReservoirClassifier(reservoir_size=5)

    prompts = ["massive risk crash", "huge growth surge", "steady stable flat", "risk drop"]
    labels = [SemanticLabel.RISK, SemanticLabel.GROWTH, SemanticLabel.STABILITY, SemanticLabel.RISK]

    # Train
    clf.fit(prompts, labels)

    # Predict (on training data for simplicity check)
    pred = clf.predict("massive risk")

    # Depending on random initialization and small data, accuracy isn't guaranteed,
    # but it should return a valid SemanticLabel string
    assert isinstance(pred, str)
    assert pred in [l.value for l in SemanticLabel]

def test_live_trainer_generation():
    trainer = LivePromptTrainer()
    prompts = ["Market is in chaos", "Steady growth"]

    p_out, l_out = trainer.generate_dataset(prompts)

    assert len(p_out) == 2
    assert len(l_out) == 2
    assert l_out[0] == SemanticLabel.CHAOS
    # "Steady growth" contains "growth", so heuristic picks GROWTH first likely, or STABILITY if logic matches "Steady" first.
    # Let's check the code: "growth" check is before "stable".
    assert l_out[1] == SemanticLabel.GROWTH

@pytest.mark.asyncio
async def test_neuro_worker_training_task():
    board = PheromoneBoard()
    worker = NeuroQuantumWorker(board)

    # 1. Train Task
    train_data = {
        "id": "train_001",
        "type": "TRAIN_RESERVOIR",
        "prompts": ["risk alert", "growth spike", "chaos everywhere"]
    }

    await worker.execute_task(train_data)

    results = await board.sniff("TRAINING_RESULT")
    assert len(results) > 0
    assert results[0].data["status"] == "trained"

    # 2. Simulation Task (now utilizing the model)
    sim_data = {
        "id": "sim_001",
        "type": "SIMULATION",
        "context": "risk alert incoming",
        "steps": 5
    }

    await worker.execute_task(sim_data)

    sim_results = await board.sniff("NEURO_RESULT")
    assert len(sim_results) > 0
    res = sim_results[0].data

    # Check if prediction field exists
    assert "prediction" in res
    # Since we trained on "risk alert" -> RISK, the prediction for "risk alert incoming" should ideally be RISK
    # but verifying existence is enough for integration test
