import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

class MemoryMixin:
    """
    A mixin to add simple file-based memory persistence to agents.
    Useful for saving state across sessions or crashes.
    """

    def load_memory(self, file_path: str = None) -> Dict[str, Any]:
        """
        Loads agent memory from a JSON file.

        Args:
            file_path: Optional custom path. If None, uses default based on agent ID.
        """
        if not file_path:
            agent_id = getattr(self, "name", "unknown_agent")
            file_path = f"data/memory/{agent_id}_memory.json"

        if not os.path.exists(file_path):
            logging.info(f"No memory file found at {file_path}. Starting fresh.")
            return {}

        try:
            with open(file_path, 'r') as f:
                memory = json.load(f)
            logging.info(f"Loaded memory from {file_path}")

            # If agent has a 'state' attribute (HNASP), try to merge or restore key fields
            if hasattr(self, 'state') and hasattr(self.state, 'logic_layer'):
                if "state_variables" in memory:
                    self.state.logic_layer.state_variables.update(memory["state_variables"])

            # Also update legacy context
            if hasattr(self, 'context'):
                self.context.update(memory.get("context", {}))

            return memory
        except Exception as e:
            logging.error(f"Failed to load memory: {e}")
            return {}

    def save_memory(self, file_path: str = None, extra_data: Dict[str, Any] = None):
        """
        Saves agent state to a JSON file.

        Args:
            file_path: Optional custom path.
            extra_data: Additional data to save beyond state variables.
        """
        if not file_path:
            agent_id = getattr(self, "name", "unknown_agent")
            # Ensure directory exists
            os.makedirs("data/memory", exist_ok=True)
            file_path = f"data/memory/{agent_id}_memory.json"

        data_to_save = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": getattr(self, "name", "unknown"),
            "context": getattr(self, "context", {}),
            "state_variables": {}
        }

        if hasattr(self, 'state') and hasattr(self.state, 'logic_layer'):
            data_to_save["state_variables"] = self.state.logic_layer.state_variables

        if extra_data:
            data_to_save.update(extra_data)

        try:
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            logging.info(f"Saved memory to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save memory: {e}")

    # --- Additive Enhancements for v23.5 ---

    def remember(self, key: str, value: Any):
        """
        Convenience method to update internal state/context memory immediately.
        Does not persist to disk (use save_memory for that).
        """
        if hasattr(self, 'context'):
            self.context[key] = value

        if hasattr(self, 'state') and hasattr(self.state, 'logic_layer'):
            self.state.logic_layer.state_variables[key] = value

    def recall(self, key: str) -> Any:
        """
        Retrieves a value from internal state/context memory.
        """
        # Check HNASP state first
        if hasattr(self, 'state') and hasattr(self.state, 'logic_layer'):
            val = self.state.logic_layer.state_variables.get(key)
            if val is not None:
                return val

        # Fallback to legacy context
        if hasattr(self, 'context'):
            return self.context.get(key)

        return None
