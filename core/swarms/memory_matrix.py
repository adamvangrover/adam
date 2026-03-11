import json
import os
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

class MemoryMatrix:
    """
    A persistent, distributed shared memory for the Agent Swarm.
    Allows agents to share insights, consensus, and state across different execution sessions.

    Concept: "The Matrix" is a JSON-backed KV store that serves as the 'Collective Unconscious' of the swarm.
    """

    def __init__(self, storage_path: str = "data/swarm_memory_matrix.json"):
        self.storage_path = storage_path
        self.memory_store: Dict[str, Any] = self._load_memory()
        self.logger = logging.getLogger(__name__)

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                # Log error but don't crash, return empty structure
                if hasattr(self, 'logger'):
                    self.logger.error(f"Failed to load memory: {e}")
                return {"meta": {"created_at": datetime.utcnow().isoformat()}, "nodes": {}}
        return {"meta": {"created_at": datetime.utcnow().isoformat()}, "nodes": {}}

    def _save_memory(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.memory_store, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")

    def write_consensus(self, topic: str, insight: str, agent_id: str, confidence: float):
        """
        Writes a consensus point to the matrix.
        """
        key = hashlib.sha256(topic.encode()).hexdigest()

        if "nodes" not in self.memory_store:
            self.memory_store["nodes"] = {}

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
        if "nodes" in self.memory_store:
            for key, node in self.memory_store["nodes"].items():
                if topic_keyword.lower() in node["topic"].lower():
                    results.append(node)
        return results

    def get_all_topics(self) -> List[str]:
        """
        Returns a list of all active topics in the matrix.
        """
        topics = []
        if "nodes" in self.memory_store:
            for node in self.memory_store["nodes"].values():
                topics.append(node.get("topic", "Unknown"))
        return topics

    def get_insights_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Retrieves all insights for a specific topic (exact match on topic name).
        """
        key = hashlib.sha256(topic.encode()).hexdigest()
        if "nodes" in self.memory_store and key in self.memory_store["nodes"]:
             return self.memory_store["nodes"][key].get("insights", [])
        return []

    def prune_stale_insights(self, max_age_hours: int = 24):
        """
        Removes insights older than the specified age.
        If a topic has no insights left, the topic node itself is removed.
        """
        if "nodes" not in self.memory_store:
            return

        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        keys_to_remove = []

        for key, node in self.memory_store["nodes"].items():
            original_count = len(node.get("insights", []))

            # Filter insights
            node["insights"] = [
                insight for insight in node.get("insights", [])
                if datetime.fromisoformat(insight["timestamp"]) > cutoff_time
            ]

            # If insights were removed, update consensus score
            if len(node["insights"]) != original_count:
                if node["insights"]:
                    total_conf = sum(i["confidence"] for i in node["insights"])
                    node["consensus_score"] = total_conf / len(node["insights"])
                else:
                    keys_to_remove.append(key)

        # Remove empty nodes
        for key in keys_to_remove:
            del self.memory_store["nodes"][key]

        self._save_memory()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self.memory_store.get("nodes", {})),
            "last_updated": datetime.utcnow().isoformat()
        }
