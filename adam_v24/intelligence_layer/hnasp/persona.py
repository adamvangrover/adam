import math
from typing import Tuple

class BayesACTPersona:
    """
    HNASP: Probabilistic Personality via BayesACT.
    Maintains an agent's identity as a vector in Evaluation-Potency-Activity (EPA) space.
    """
    def __init__(self, name: str, epa_vector: Tuple[float, float, float]):
        self.name = name
        self.fundamental_identity = epa_vector # The "True Self"
        self.transient_state = epa_vector      # Current emotional state
        self.deflection_history = []

    def interact(self, user_epa: Tuple[float, float, float]) -> str:
        """
        Simulate an interaction. If user EPA conflicts with agent EPA,
        calculate deflection and adjust response to minimize it.
        """
        deflection = self._calculate_deflection(self.transient_state, user_epa)
        self.deflection_history.append(deflection)

        # Simple heuristic: If user is aggressive (Low Evaluation, High Potency),
        # Agent maintains professional distance (Neutral Evaluation, Controlled Potency)
        # instead of mirroring aggression.

        response_tone = "Neutral"
        if user_epa[0] < -1.0: # Negative Evaluation
            response_tone = "De-escalating"
            # Restore balance
            self.transient_state = self.fundamental_identity

        return f"Agent {self.name} responds in {response_tone} tone. (Deflection: {deflection:.2f})"

    def _calculate_deflection(self, agent_epa, user_epa):
        # Euclidean distance in EPA space
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(agent_epa, user_epa)))

# Example Personas
# Risk Architect: High Evaluation, High Potency, Low Activity (Stable, Good, Powerful)
RISK_ARCHITECT_EPA = (2.5, 2.0, -0.5)
