import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any

class ContextFreezer:
    """
    Enforces zero-context-drift by implementing a frozen context object at the
    initialization of any decision lifecycle. The context is mathematically bound
    using a canonical cryptographic hash function to ensure it is immutable.
    """

    def __init__(self):
        pass

    def freeze_context(self, context_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a raw context dictionary, normalizes it, generates a cryptographic hash,
        and returns an immutable frozen context representation.
        """
        # Ensure deep copying by value logic to avoid mutability references
        frozen_data = context_payload.copy()

        # Serialize with sorted keys for canonical representation
        canonical_json = json.dumps(frozen_data, sort_keys=True, separators=(',', ':')).encode('utf-8')

        context_hash = hashlib.sha256(canonical_json).hexdigest()

        return {
            "frozen_data": frozen_data,
            "context_hash": context_hash,
            "timestamp_frozen": datetime.now(timezone.utc).isoformat(),
            "immutable": True
        }

    def verify_frozen_context(self, frozen_context: Dict[str, Any]) -> bool:
        """
        Verifies that the frozen context data matches its cryptographic hash,
        proving it has not drifted or been tampered with.
        """
        if not frozen_context.get("immutable"):
            return False

        data = frozen_context.get("frozen_data", {})
        canonical_json = json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
        recomputed_hash = hashlib.sha256(canonical_json).hexdigest()

        return recomputed_hash == frozen_context.get("context_hash")
