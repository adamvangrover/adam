from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .synapse import QuantumSynapse

@dataclass
class LiquidNeuron:
    id: str
    state: float = 0.0
    tau: float = 1.0  # Time constant
    bias: float = 0.0
    # Inputs: map of source_id -> QuantumSynapse
    incoming_synapses: Dict[str, QuantumSynapse] = field(default_factory=dict)

class LiquidNeuralNetwork:
    """
    Implements a Liquid Time-Constant (LTC) Network where neuron states evolve
    according to differential equations, and connections are mediated by QuantumSynapses.
    """
    def __init__(self):
        self.neurons: Dict[str, LiquidNeuron] = {}

    def add_neuron(self, neuron_id: str, tau: float = 1.0, bias: float = 0.0):
        self.neurons[neuron_id] = LiquidNeuron(id=neuron_id, tau=tau, bias=bias)

    def add_synapse(self, source_id: str, target_id: str, mean_weight: float, uncertainty: float):
        if target_id not in self.neurons:
            raise ValueError(f"Target neuron {target_id} does not exist.")

        # Source can be an input ID (not a neuron) or another neuron
        synapse = QuantumSynapse(weight_mean=mean_weight, uncertainty=uncertainty)
        self.neurons[target_id].incoming_synapses[source_id] = synapse

    def compute_step(self, inputs: Dict[str, float], dt: float = 0.1) -> Dict[str, float]:
        """
        Performs one time-step of integration using Euler method.
        x(t+dt) = x(t) + (-x(t)/tau + S(t)) * dt
        """
        new_states = {}

        for n_id, neuron in self.neurons.items():
            # 1. Calculate synaptic input S(t)
            synaptic_input = 0.0

            # Collapse quantum states for this step (stochastic measurement)
            for src_id, synapse in neuron.incoming_synapses.items():
                weight = synapse.measure()

                # Input value
                if src_id in inputs:
                    val = inputs[src_id]
                elif src_id in self.neurons:
                    val = self.neurons[src_id].state # Recurrent connection
                else:
                    val = 0.0 # Missing input

                synaptic_input += weight * val

            # 2. Compute derivative dx/dt
            # dx/dt = -(x - bias) / tau + S(t)
            # Using tanh activation on the synaptic input is common in LTCs to keep it bounded
            # but keeping it linear for now per simplified spec, or adding simple sigmoid

            # Non-linearity on the input sum
            activated_input = np.tanh(synaptic_input)

            dx_dt = -(neuron.state - neuron.bias) / neuron.tau + activated_input

            # 3. Update state
            new_state = neuron.state + dx_dt * dt
            new_states[n_id] = new_state

        # Update all neurons simultaneously
        for n_id, val in new_states.items():
            self.neurons[n_id].state = val

        return new_states

    def run(self, inputs_series: List[Dict[str, float]], dt: float = 0.1) -> List[Dict[str, float]]:
        """
        Runs the network over a sequence of inputs.
        """
        history = []
        for inputs in inputs_series:
            states = self.compute_step(inputs, dt)
            history.append(states.copy())
        return history
