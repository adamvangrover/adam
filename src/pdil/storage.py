import json

class DriftStorageBackend:
    """Storage backend for keeping historical drift data."""
    def __init__(self, storage_path: str = "drift_data.json"):
        self.storage_path = storage_path
        self.data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_drift(self, execution_id: str, drift_info: dict):
        self.data[execution_id] = drift_info
        with open(self.storage_path, 'w') as f:
            json.dump(self.data, f)
