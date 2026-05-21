import datetime
import json
import os
from typing import Dict, Any

class HumanValidationGate:
    """
    Middleware for the Human Validation Gate in the RLHF loop.
    Captures differences between LLM generated JSON and human edited JSON,
    saving them to a corrections ledger for future few-shot prompts.
    """
    def __init__(self, ledger_path: str = "corrections_ledger.json"):
        self.ledger_path = ledger_path

    def _find_differences(self, original: Dict[str, Any], edited: Dict[str, Any], path="") -> list:
        differences = []

        for key in original:
            current_path = f"{path}.{key}" if path else key

            if key not in edited:
                differences.append({"path": current_path, "action": "deleted", "original": original[key]})
            elif isinstance(original[key], dict) and isinstance(edited[key], dict):
                differences.extend(self._find_differences(original[key], edited[key], current_path))
            elif original[key] != edited[key]:
                differences.append({"path": current_path, "action": "modified", "original": original[key], "edited": edited[key]})

        for key in edited:
            current_path = f"{path}.{key}" if path else key
            if key not in original:
                differences.append({"path": current_path, "action": "added", "edited": edited[key]})

        return differences

    def process_validation(self, original_payload: dict, edited_payload: dict):
        """
        Process the validation step by comparing original and edited payloads.
        If differences exist, record them to the ledger.
        """
        differences = self._find_differences(original_payload, edited_payload)

        if differences:
            self._record_corrections(differences)

        return edited_payload

    def _record_corrections(self, differences: list):
        """Append differences to the corrections ledger."""
        ledger = []
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, 'r') as f:
                try:
                    ledger = json.load(f)
                except json.JSONDecodeError:
                    pass

        ledger.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "differences": differences
        })

        with open(self.ledger_path, 'w') as f:
            json.dump(ledger, f, indent=4)
