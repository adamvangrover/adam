from core.intelligence.nssf.mocks import torch, nn
from core.risk_engine.quantum_model import QuantumRiskEngine

class QuantumSynapseLayer(nn.Module):
    """
    Implements a Quantum Synapse Layer that simulates a Variational Quantum Circuit (VQC).
    It inherits from torch.nn.Module to integrate with standard Deep Learning pipelines.
    """
    def __init__(self, input_dim: int, output_dim: int, backend_engine: QuantumRiskEngine):
        super().__init__()
        self.backend = backend_engine

        # Variational parameters (theta) for the quantum circuit.
        # These represent rotation angles in the PQC (Parameterized Quantum Circuit).
        self.theta = nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, x):
        """
        Defines the computation performed at every call.
        """
        # Check if the backend is 'real' (Qiskit available) or we are in simulation/fallback
        if self.backend.qiskit_available:
            return self._mock_quantum_forward(x)
        else:
            return self._classical_shadow_fallback(x)

    def _mock_quantum_forward(self, x):
        """
        Simulate the non-linear activation of a quantum measurement when Qiskit is present
        but we are running a simulation of the VQC for speed or testing.

        Equation: Output ~ cos^2(x * theta) which approximates Rabi oscillations.
        """
        linear_part = torch.matmul(x, self.theta)
        # Apply the quantum-like non-linearity
        return torch.cos(linear_part) ** 2

    def _classical_shadow_fallback(self, x):
        """
        Fallback mode when quantum backend is offline.
        Uses a classical linear transformation with added noise to simulate
        quantum uncertainty (decoherence).
        """
        out = torch.matmul(x, self.theta)
        # Add 'quantum noise'
        noise = torch.randn_like(out) * 0.01
        return out + noise
