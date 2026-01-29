import asyncio
import logging
from typing import Dict, Any, List
from core.engine.swarm.worker_node import SwarmWorker
from core.engine.swarm.pheromone_board import PheromoneBoard
from core.v30_architecture.neuro_quantum.liquid_net import LiquidNeuralNetwork
from core.v30_architecture.neuro_quantum.framer import NeuroSymbolicFramer
from core.v30_architecture.neuro_quantum.reservoir import QuantumReservoirClassifier
from core.v30_architecture.neuro_quantum.trainer import LivePromptTrainer
from core.v30_architecture.neuro_quantum.synthesizer import StateSynthesizer

logger = logging.getLogger(__name__)

class NeuroQuantumWorker(SwarmWorker):
    """
    Swarm Worker specialized in running Neuro-Quantum simulations.
    It builds Liquid Neural Networks, overlays deterministic scripts,
    and returns probabilistic state evolutions.

    Updated to support 'TASK_TRAIN_RESERVOIR' using live prompt sets.
    """
    def __init__(self, board: PheromoneBoard, role: str = "neuro_quantum"):
        super().__init__(board, role)
        self.framer = NeuroSymbolicFramer()
        self.trainer = LivePromptTrainer()
        self.synthesizer = StateSynthesizer()
        # Persisting the model in memory for this worker instance (mock persistence)
        self.reservoir_model = None

    async def execute_task(self, data: Dict[str, Any]):
        task_type = data.get("type", "SIMULATION")

        if task_type == "TRAIN_RESERVOIR":
            await self._execute_training(data)
        else:
            await self._execute_simulation(data)

    async def _execute_training(self, data: Dict[str, Any]):
        task_id = data.get("id", "unknown_train_task")
        prompts = data.get("prompts", [])

        logger.info(f"NeuroWorker {self.id} starting training on {len(prompts)} prompts.")

        # Generate labeled dataset
        train_prompts, train_labels = self.trainer.generate_dataset(prompts)

        # Initialize and Train Reservoir
        # In a real distributed system, we might load an existing model here.
        self.reservoir_model = QuantumReservoirClassifier(reservoir_size=20)
        self.reservoir_model.fit(train_prompts, train_labels)

        logger.info(f"NeuroWorker {self.id} training complete.")

        result = {
            "task_id": task_id,
            "status": "trained",
            "samples": len(train_prompts),
            "worker": self.id
        }
        await self.board.deposit("TRAINING_RESULT", result, intensity=10.0, source=self.id)

    async def _execute_simulation(self, data: Dict[str, Any]):
        task_id = data.get("id", "unknown_task")
        context = data.get("context", "")
        script = data.get("script", {})
        steps = data.get("steps", 10)
        dt = data.get("dt", 0.1)

        logger.info(f"NeuroWorker {self.id} starting simulation for task {task_id}")

        # If we have a trained model, we could optionally classify the context first
        prediction = None
        if self.reservoir_model:
            prediction = self.reservoir_model.predict(context)
            logger.info(f"Reservoir predicted context type: {prediction}")

        # 1. Build Network (Mock topology for now, or dynamic based on script)
        # In a full implementation, the topology might come from the 'script' or 'context'
        net = LiquidNeuralNetwork()

        # Add Input Nodes (virtual placeholders managed by Framer)
        # Add Hidden/Output Nodes
        net.add_neuron("risk_accumulator", tau=5.0, bias=-0.5)
        net.add_neuron("market_stability", tau=10.0, bias=0.5)

        # Connect Inputs to Neurons
        # (weight_mean, uncertainty) -> quantum synapse
        net.add_synapse("input_risk", "risk_accumulator", 1.0, 0.2)
        net.add_synapse("input_stability", "market_stability", 0.8, 0.1)

        # Cross connections
        net.add_synapse("risk_accumulator", "market_stability", -2.0, 0.5) # High risk destabilizes market (inhibitory)

        # 2. Frame Context & Script
        input_vector = self.framer.frame_context(context)
        self.framer.overlay_script(net, script)

        # 3. Run Simulation
        # Repeat input vector for the duration of the simulation (constant stimulus)
        inputs_series = [input_vector for _ in range(steps)]

        history = net.run(inputs_series, dt=dt)

        final_state = history[-1]

        # 4. Holistic State Synthesis
        holistic_state = self.synthesizer.synthesize(net, context, final_state)

        # Serialize for transport (dataclass to dict)
        holistic_state_dict = {
            "neural_state_vector": holistic_state.neural_state_vector,
            "quantum_entropy": holistic_state.quantum_entropy,
            "market_regime": holistic_state.market_regime.value,
            "fibo_concepts": [f.value for f in holistic_state.fibo_concepts]
        }

        result = {
            "task_id": task_id,
            "final_state": final_state,
            "holistic_state": holistic_state_dict,
            "trajectory_sample": [h["market_stability"] for h in history], # Just sending one trace for brevity
            "prediction": prediction,
            "worker": self.id,
            "status": "completed"
        }

        await self.board.deposit("NEURO_RESULT", result, intensity=10.0, source=self.id)
