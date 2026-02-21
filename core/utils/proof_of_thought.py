import hashlib
import json
import os
import time
from typing import Dict, Any, List

class ProofOfThoughtLogger:
    """
    Project OMEGA: Pillar 2 - The Trust Engine.
    Implements 'Proof of Thought' (PoT) by hashing analytical steps into an immutable chain.
    """
    def __init__(self, ledger_path: str = "showcase/data/proof_of_thought_ledger.json"):
        self.ledger_path = ledger_path
        self._ensure_ledger()

    def _ensure_ledger(self):
        if not os.path.exists(os.path.dirname(self.ledger_path)):
            os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w") as f:
                # Genesis Block
                genesis = {
                    "index": 0,
                    "timestamp": time.time(),
                    "agent": "SYSTEM",
                    "thought": "GENESIS_BLOCK",
                    "previous_hash": "0" * 64,
                    "hash": self._calculate_hash(0, time.time(), "SYSTEM", "GENESIS_BLOCK", "0" * 64)
                }
                json.dump([genesis], f, indent=2)

    def _calculate_hash(self, index, timestamp, agent, thought, previous_hash):
        payload = f"{index}{timestamp}{agent}{thought}{previous_hash}".encode()
        return hashlib.sha256(payload).hexdigest()

    def log_thought(self, agent: str, thought: str, metadata: Dict[str, Any] = None):
        """
        Logs a thought step, hashing it and linking it to the previous entry.
        """
        chain = self._load_chain()
        last_block = chain[-1]

        new_index = last_block["index"] + 1
        timestamp = time.time()
        previous_hash = last_block["hash"]

        thought_content = json.dumps({"thought": thought, "metadata": metadata or {}}, sort_keys=True)
        new_hash = self._calculate_hash(new_index, timestamp, agent, thought_content, previous_hash)

        block = {
            "index": new_index,
            "timestamp": timestamp,
            "agent": agent,
            "thought": thought_content,
            "previous_hash": previous_hash,
            "hash": new_hash
        }

        chain.append(block)
        self._save_chain(chain)
        return new_hash

    def verify_chain(self) -> bool:
        """
        Verifies the cryptographic integrity of the ledger.
        """
        chain = self._load_chain()
        for i in range(1, len(chain)):
            current = chain[i]
            previous = chain[i-1]

            if current["previous_hash"] != previous["hash"]:
                print(f"Broken Link at Index {current['index']}")
                return False

            recalc_hash = self._calculate_hash(
                current["index"],
                current["timestamp"],
                current["agent"],
                current["thought"],
                current["previous_hash"]
            )

            if current["hash"] != recalc_hash:
                print(f"Invalid Hash at Index {current['index']}")
                return False

        return True

    def _load_chain(self) -> List[Dict]:
        try:
            with open(self.ledger_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._ensure_ledger()
            return self._load_chain()

    def _save_chain(self, chain: List[Dict]):
        with open(self.ledger_path, "w") as f:
            json.dump(chain, f, indent=2)
