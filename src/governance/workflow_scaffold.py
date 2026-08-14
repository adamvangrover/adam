import time
import json
import hashlib
from typing import Dict, Any
from src.governance.gatekeeper import GovernanceGatekeeper
from src.pdil.models import ProvenanceHeader

def main():
    schema = {
        "type": "object",
        "properties": {
            "decision": {"type": "string"},
            "amount": {"type": "number"}
        },
        "required": ["decision", "amount"]
    }

    gatekeeper = GovernanceGatekeeper(schema=schema)

    payload = {"decision": "APPROVE", "amount": 1000000}
    payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    content_hash = hashlib.sha256(payload_json).hexdigest()

    header = ProvenanceHeader(
        git_commit_hash="mock_hash_123",
        timestamp=str(time.time()),
        content_hash=content_hash,
        jsonLogic_version="1.0.0",
        confidence_score=0.95,
        derivation_path="system_1_agent -> pdil -> gatekeeper",
        source_data_object="https://example.com"
    )

    inference_output = {
        "provenance_trace": header.model_dump(),
        "data": payload
    }

    print("--- Entry Gate ---")
    validated_entry = gatekeeper.entry_gate(inference_output)
    print(f"Validated Entry: {validated_entry['data']['decision']}")

    print("--- Execution (Mock) ---")
    print("Rust Deterministic Kernel Processing...")

    print("--- Exit Gate ---")
    validated_exit = gatekeeper.exit_gate(validated_entry)
    print(f"Validated Exit: {validated_exit['data']['decision']}")

if __name__ == "__main__":
    main()
