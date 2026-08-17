import hashlib
import json
from typing import Any

def generate_prov_hash(data: Any) -> str:
    """
    Generate a cryptographic hash for a provenance node.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
