"""
Manifest definitions for ADAM Gold Standard Certification.
"""
import json
from typing import Dict, Any

def generate_manifest(manifest_type: str, data: Dict[str, Any], output_path: str):
    """
    Generates a standard JSON manifest.
    """
    manifest_data = {
        "manifest_type": manifest_type,
        "content": data
    }
    with open(output_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    return manifest_data
