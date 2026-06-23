from typing import Dict, Any
import time

class ModelLifecycleManager:
    """Manages model and software lifecycle states."""
    def __init__(self):
        self.active_models: Dict[str, Dict[str, Any]] = {}

    def register_model(self, model_id: str, version: str, deprecation_date: float):
        """Registers a model version in the lifecycle."""
        self.active_models[model_id] = {
            "version": version,
            "deprecation_date": deprecation_date,
            "status": "ACTIVE"
        }

    def check_lifecycle(self, model_id: str) -> str:
        """Checks if a model is nearing deprecation or obsolete."""
        if model_id not in self.active_models:
            return "UNKNOWN_MODEL"

        model_info = self.active_models[model_id]
        if time.time() > model_info["deprecation_date"]:
            model_info["status"] = "DEPRECATED"
            return "DEPRECATED"
        return "ACTIVE"
