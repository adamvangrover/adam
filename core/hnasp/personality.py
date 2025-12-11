import numpy as np
from typing import Dict, Any, List, Optional
from core.schemas.hnasp import EPAVector, Identity, PersonaState

class BayesACTEngine:
    def __init__(self):
        # Placeholder for EPA dictionary
        self.epa_dictionary = {
            "greet": {"E": 2.0, "P": 1.0, "A": 1.0},
            "attack": {"E": -2.0, "P": 1.5, "A": 1.5},
            "agree": {"E": 1.5, "P": 0.5, "A": 0.0},
            "default": {"E": 0.0, "P": 0.0, "A": 0.0}
        }

    def _epa_to_array(self, epa: EPAVector) -> np.ndarray:
        return np.array([epa.E, epa.P, epa.A])

    def _array_to_epa(self, arr: np.ndarray) -> EPAVector:
        return EPAVector(E=float(arr[0]), P=float(arr[1]), A=float(arr[2]))

    def calculate_deflection(self, fundamental: EPAVector, transient: EPAVector) -> float:
        """
        Calculates deflection (squared Euclidean distance) between fundamental and transient sentiments.
        D = || F - T ||^2
        """
        f = self._epa_to_array(fundamental)
        t = self._epa_to_array(transient)
        diff = f - t
        return float(np.dot(diff, diff))

    def get_interaction_epa(self, text: str) -> EPAVector:
        """
        Mock implementation of text-to-EPA mapping.
        In a real system, this would lookup the verb/object in an ACT dictionary.
        """
        text_lower = text.lower()
        if "hello" in text_lower or "hi" in text_lower:
            vals = self.epa_dictionary["greet"]
        elif "stupid" in text_lower or "hate" in text_lower:
            vals = self.epa_dictionary["attack"]
        elif "yes" in text_lower or "agree" in text_lower:
            vals = self.epa_dictionary["agree"]
        else:
            vals = self.epa_dictionary["default"]

        return EPAVector(**vals)

    def update_persona_state(self, state: PersonaState, incoming_text: str, role: str = "user") -> PersonaState:
        """
        Updates the persona state based on an incoming interaction.
        """
        # 1. Calculate EPA of the incoming act
        interaction_epa = self.get_interaction_epa(incoming_text)

        # 2. Update Transient Impression
        # (Simplified: New Transient = Interaction EPA for now.
        # In full ACT, it's a fusion of Actor, Behavior, Object)
        if role == "user":
            # If user acts, it affects the User's transient identity and the Agent's view of the situation
            if state.identities.user.transient_epa is None:
                state.identities.user.transient_epa = interaction_epa
            else:
                # Simple moving average for now
                current = self._epa_to_array(state.identities.user.transient_epa)
                new_val = self._epa_to_array(interaction_epa)
                updated = 0.7 * current + 0.3 * new_val
                state.identities.user.transient_epa = self._array_to_epa(updated)

        # 3. Calculate Deflection for the Agent (Self)
        # Assuming the interaction happened *to* the agent.
        # Ideally we calculate the impression formed on the agent.
        # For this prototype, we'll assume the interaction EPA *is* the impression on the agent's transient state
        # if the user attacks, the agent feels "attacked" (Low E).

        if state.identities.self.fundamental_epa:
            # How does this interaction make the agent feel?
            # If attacked, transient moves to Low E.
            # We mock this by "blending" the interaction EPA into the self transient
            if state.identities.self.transient_epa is None:
                 state.identities.self.transient_epa = state.identities.self.fundamental_epa

            # Update dynamics
            f_vec = state.identities.self.fundamental_epa
            t_vec = state.identities.self.transient_epa

            # Simplified interaction update:
            # The agent's transient state is pulled towards the interaction nature
            # (e.g. if user is negative, agent's state might become negative/defensive)
            t_arr = self._epa_to_array(t_vec)
            i_arr = self._epa_to_array(interaction_epa)

            # Update transient:
            # We'll say the transient is the average of previous transient and the incoming act
            new_t_arr = 0.8 * t_arr + 0.2 * i_arr
            state.identities.self.transient_epa = self._array_to_epa(new_t_arr)

            # Recalculate Deflection
            deflection = self.calculate_deflection(f_vec, state.identities.self.transient_epa)
            state.dynamics.current_deflection = deflection

            # Calculate Target Behavior
            # The agent wants to minimize deflection.
            # Ideally it finds a behavior B such that new Transient matches Fundamental.
            # We'll just set target to Fundamental for now (simplistic restoration)
            state.dynamics.target_behavior_epa = f_vec

        return state
