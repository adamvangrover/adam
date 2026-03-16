from typing import List, Any
import logging
from core.intelligence.nssf.mocks import torch, nn
from core.schemas.v23_5_schema import ExecutionPlan
from core.intelligence.nssf.synapse import QuantumSynapseLayer
from core.risk_engine.quantum_model import QuantumRiskEngine

logger = logging.getLogger(__name__)

# Mock torchdiffeq
try:
    from torchdiffeq import odeint
except ImportError:
    logger.warning("torchdiffeq not found. Using NSSF MockODE solver.")
    def odeint(func, y0, t):
        # Simple Euler integration mock
        dt = 0.1 # explicit mock step
        if hasattr(t, 'shape') and len(t.shape) > 0 and t.shape[0] > 1:
             dt = t[1] - t[0]

        y = y0
        results = [y0]

        steps = 1
        if hasattr(t, 'shape'):
            steps = t.shape[0]
        elif isinstance(t, list):
            steps = len(t)

        for _ in range(steps-1):
            dy = func(t, y) # t is mocked here
            y = y + dy * dt
            results.append(y)

        return torch.tensor(results)

class LiquidODEFunc(nn.Module):
    """
    The differential equation defining the Liquid Neural Network dynamics.
    dy/dt = -y/tau + S(y)
    where S is the Quantum Synapse Layer.
    """
    def __init__(self, hidden_dim: int, quantum_layer: QuantumSynapseLayer):
        super().__init__()
        self.quantum_layer = quantum_layer
        self.tau = nn.Parameter(torch.randn(1)) # Time constant

    def forward(self, t, y):
        # Damping term: -y / tau
        # Ensure tau is positive and non-zero
        tau_val = torch.abs(self.tau) + 0.1
        damping = -1.0 * (y * (1.0 / tau_val)) # Mock math

        synaptic_input = self.quantum_layer(y)
        return damping + synaptic_input

class LiquidSemanticEncoder(nn.Module):
    """
    Encodes execution plans into a liquid state representation using Neural ODEs.
    """
    def __init__(self, input_dim: int, hidden_dim: int, backend_engine: QuantumRiskEngine):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Synapse layer defining the derivative
        self.synapse = QuantumSynapseLayer(hidden_dim, hidden_dim, backend_engine)
        self.ode_func = LiquidODEFunc(hidden_dim, self.synapse)

        # Projection for plan steps (Text -> Vector)
        # In a real system, this would be a BERT/Transformer embedding.
        # Here we mock it as a linear projection.
        self.projection = nn.Module()
        self.projection.weight = nn.Parameter(torch.randn(input_dim, hidden_dim))

    def encode_plan(self, plan: ExecutionPlan) -> Any:
        """
        Converts ExecutionPlan to a time-series input and runs the LNN.
        Returns the final state vector (Liquid Context).
        """
        steps = plan.steps
        num_steps = len(steps)

        if num_steps == 0:
            # Return a deterministic "zero" or baseline state instead of random noise
            return torch.tensor([0.1] * self.hidden_dim).view(1, self.hidden_dim)

        # 1. Vectorize the Plan (Deterministic)
        # We treat each step as a timepoint.
        # We hash the step content to generate a stable vector.
        step_vectors = []
        for step in steps:
            step_vectors.append(self._hash_step_to_vector(step))

        # 2. Integrate
        # We solve the ODE from t=0 to t=num_steps
        times = torch.tensor([float(i) for i in range(num_steps + 1)])

        # Initial state (at t=0) - Deterministic seed based on plan ID or constant
        seed_val = abs(hash(plan.plan_id)) % 100 / 100.0
        y0 = torch.tensor([seed_val] * self.hidden_dim).view(1, self.hidden_dim)

        # Run ODE solver
        # For this additive implementation, we just run the solver over the duration.
        # A more complex Liquid Network would inject the 'step_vectors' as continuous input I(t).
        # We will assume the ODE function captures the intrinsic dynamics.

        states = odeint(self.ode_func, y0, times)

        # Return the final state
        return states[-1] if hasattr(states, '__getitem__') else states

    def _hash_step_to_vector(self, step: Any) -> Any:
        """
        Hashes a PlanStep (or dict) into a deterministic input vector.
        """
        # Support both Pydantic model and Dict
        if hasattr(step, 'action'):
            content = f"{step.action}:{step.target_entity}:{step.step_id}"
        else:
             content = f"{step.get('action', '')}:{step.get('target_entity', '')}:{step.get('step_id', '')}"

        # Simple non-crypto hash to generate a seed
        h = abs(hash(content))

        # Pseudo-random generation seeded by hash
        # Since we use mocks or torch, we can't easily set global seed without side effects.
        # We'll construct the vector manually from the hash.

        vector = []
        for i in range(self.input_dim):
            # Shift hash to get different values for each dimension
            val = ((h >> (i % 8)) & 0xFF) / 255.0
            vector.append(val)

        return torch.tensor(vector).view(1, self.input_dim)
