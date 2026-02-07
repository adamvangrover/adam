from typing import List, Dict, Any
from .model import OSWMTransformer, WorldModelState

class OSWMInference:
    """
    Handles inference for the One-Shot World Model.
    """

    def __init__(self):
        self.model = OSWMTransformer()

    def predict_next_state(self, current_state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """
        Predicts the next world state given the current state and an action.
        """
        # 1. Tokenize State + Action
        # Mock tokenization
        input_ids = [1, 2, 3]

        # 2. Model Forward Pass
        logits = self.model.forward(input_ids)

        # 3. Decode Output
        # Simulate state transition
        next_state = current_state.copy()
        next_state['step'] = current_state.get('step', 0) + 1

        if action == "stimulate_economy":
            next_state['gdp'] = current_state.get('gdp', 100) * 1.02
            next_state['inflation'] = current_state.get('inflation', 0.02) + 0.005
        elif action == "raise_rates":
            next_state['gdp'] = current_state.get('gdp', 100) * 0.99
            next_state['inflation'] = current_state.get('inflation', 0.02) - 0.005

        return next_state
