import hashlib
import json
import time
from typing import Dict, Any, List, Optional


class AuditLedgerTamperedError(Exception):
    """Raised when cryptographic provenance chain verification fails due to log tampering."""
    pass


class ProvenanceReplayLedger:
    """
    W3C PROV-O-compatible provenance ledger linking material assertions, extractions,
    formulas, and state transition requests to canonical source hashes through chained SHA-256 digests.
    Generates RDF Turtle graphs and detects audit trail tampering.
    """

    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self):
        self.chain: List[Dict[str, Any]] = []

    def record_event(
        self,
        event_type: str,
        entity_id: str,
        actor_id: str,
        payload: Dict[str, Any],
        source_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        previous_chain_hash = self.chain[-1]["current_chain_hash"] if self.chain else self.GENESIS_HASH
        timestamp = time.time()

        raw_payload_str = json.dumps(payload, sort_keys=True)
        payload_sha256 = hashlib.sha256(raw_payload_str.encode("utf-8")).hexdigest()

        block_data = f"{previous_chain_hash}:{event_type}:{entity_id}:{actor_id}:{payload_sha256}:{timestamp}"
        current_chain_hash = hashlib.sha256(block_data.encode("utf-8")).hexdigest()

        event_node = {
            "index": len(self.chain),
            "previous_chain_hash": previous_chain_hash,
            "current_chain_hash": current_chain_hash,
            "timestamp": timestamp,
            "event_type": event_type,
            "entity_id": entity_id,
            "actor_id": actor_id,
            "source_hash": source_hash or self.GENESIS_HASH,
            "payload_sha256": payload_sha256,
            "payload": payload
        }
        self.chain.append(event_node)
        return event_node

    def verify_ledger_integrity(self) -> bool:
        if not self.chain:
            return True

        for i, node in enumerate(self.chain):
            expected_prev = self.GENESIS_HASH if i == 0 else self.chain[i - 1]["current_chain_hash"]
            if node["previous_chain_hash"] != expected_prev:
                return False

            raw_payload_str = json.dumps(node["payload"], sort_keys=True)
            recalculated_payload_sha256 = hashlib.sha256(raw_payload_str.encode("utf-8")).hexdigest()
            if recalculated_payload_sha256 != node["payload_sha256"]:
                return False

            block_data = f"{node['previous_chain_hash']}:{node['event_type']}:{node['entity_id']}:{node['actor_id']}:{node['payload_sha256']}:{node['timestamp']}"
            recalculated_chain_hash = hashlib.sha256(block_data.encode("utf-8")).hexdigest()
            if recalculated_chain_hash != node["current_chain_hash"]:
                return False

        return True

    def export_w3c_provo_turtle(self) -> str:
        turtle_lines = [
            "@prefix prov: <http://www.w3.org/ns/prov#> .",
            "@prefix adam: <https://adam.financial/ontologies/prov#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            ""
        ]

        for node in self.chain:
            e_id = f"adam:entity_{node['entity_id'].replace('-', '_')}"
            a_id = f"adam:agent_{node['actor_id'].replace('/', '_').replace('-', '_')}"
            act_id = f"adam:activity_{node['index']}_{node['event_type']}"

            turtle_lines.extend([
                f"{e_id} a prov:Entity ;",
                f'    prov:value "{node["payload_sha256"]}" ;',
                f'    adam:chainHash "{node["current_chain_hash"]}" .',
                "",
                f"{a_id} a prov:Agent ;",
                f'    prov:label "{node["actor_id"]}" .',
                "",
                f"{act_id} a prov:Activity ;",
                f'    prov:wasAssociatedWith {a_id} ;',
                f'    prov:generated {e_id} .',
                ""
            ])

        return "\n".join(turtle_lines)
