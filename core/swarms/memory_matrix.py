import json
import os
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime

class MemoryMatrix:
    """
    A persistent, distributed shared memory for the Agent Swarm.
    Allows agents to share insights, consensus, and state across different execution sessions.

    Concept: "The Matrix" is a JSON-backed KV store that serves as the 'Collective Unconscious' of the swarm.
    """

    def __init__(self, storage_path: str = "data/swarm_memory_matrix.json"):
        self.storage_path = storage_path
        self.memory_store: Dict[str, Any] = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"meta": {"created_at": datetime.utcnow().isoformat()}, "nodes": {}}
        return {"meta": {"created_at": datetime.utcnow().isoformat()}, "nodes": {}}

    def _save_memory(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory_store, f, indent=2)

    def write_consensus(self, topic: str, insight: str, agent_id: str, confidence: float):
        """
        Writes a consensus point to the matrix.
        """
        key = hashlib.sha256(topic.encode()).hexdigest()

        if key not in self.memory_store["nodes"]:
            self.memory_store["nodes"][key] = {
                "topic": topic,
                "insights": [],
                "consensus_score": 0.0
            }

        node = self.memory_store["nodes"][key]

        # Add insight
        node["insights"].append({
            "agent": agent_id,
            "content": insight,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Recalculate generic consensus (simple average of confidence)
        total_conf = sum(i["confidence"] for i in node["insights"])
        node["consensus_score"] = total_conf / len(node["insights"])

        self._save_memory()

    def query_matrix(self, topic_keyword: str) -> List[Dict[str, Any]]:
        """
        Retrieves insights related to a keyword.
        """
        results = []
        for key, node in self.memory_store["nodes"].items():
            if topic_keyword.lower() in node["topic"].lower():
                results.append(node)
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self.memory_store["nodes"]),
            "last_updated": datetime.utcnow().isoformat()
        }
