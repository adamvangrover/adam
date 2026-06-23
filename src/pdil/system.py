from typing import Dict, Any

class SystemStateManager:
    """Manages meta-state, layer state, and context across the PDIL architecture."""
    def __init__(self):
        self.global_state: Dict[str, Any] = {}
        self.layer_states: Dict[str, Dict[str, Any]] = {}

    def update_layer_state(self, layer_name: str, state_delta: Dict[str, Any]):
        """Updates the state context of a specific architectural layer."""
        if layer_name not in self.layer_states:
            self.layer_states[layer_name] = {}
        self.layer_states[layer_name].update(state_delta)

    def get_context(self, layer_name: str) -> Dict[str, Any]:
        """Retrieves combined context for a layer."""
        context = self.global_state.copy()
        context.update(self.layer_states.get(layer_name, {}))
        return context
