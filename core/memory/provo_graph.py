# core/memory/provo_graph.py

import logging
from typing import Dict, Any, List
import uuid
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ProvoGraph:
    """
    Implements a PROV-O compliant Knowledge Graph for audit trails.
    Tracks entities, activities, and agents.
    """
    def __init__(self, storage_file="data/provo_graph.json"):
        self.storage_file = storage_file
        self.nodes = {}
        self.edges = []
        self.ips = {} # Investment Policy Statement
        self._load()

    def _load(self):
        try:
            with open(self.storage_file, 'r') as f:
                data = json.load(f)
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", [])
                self.ips = data.get("ips", {})
        except FileNotFoundError:
            pass

    def _save(self):
        with open(self.storage_file, 'w') as f:
            json.dump({
                "nodes": self.nodes,
                "edges": self.edges,
                "ips": self.ips
            }, f, indent=2)

    def store_ips(self, ips_data: Dict[str, Any]):
        """Stores the Investment Policy Statement."""
        self.ips = ips_data
        self.log_activity("UpdateIPS", "System", "IPS updated")
        self._save()

    def get_ips(self) -> Dict[str, Any]:
        """Retrieves the Investment Policy Statement."""
        return self.ips

    def log_activity(self, activity_type: str, agent_id: str, details: str, inputs: List[str] = None):
        """
        Logs an activity in PROV-O format.
        """
        activity_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        node = {
            "id": activity_id,
            "type": "prov:Activity",
            "label": activity_type,
            "agent": agent_id,
            "details": details,
            "timestamp": timestamp
        }
        self.nodes[activity_id] = node

        if inputs:
            for inp in inputs:
                self.edges.append({
                    "source": activity_id,
                    "target": inp,
                    "relation": "prov:used"
                })

        self._save()
        return activity_id

    def log_entity(self, entity_id: str, entity_type: str, data: Dict[str, Any], generated_by: str = None):
        """Logs a new data entity."""
        timestamp = datetime.now().isoformat()
        node = {
            "id": entity_id,
            "type": "prov:Entity",
            "entity_type": entity_type,
            "data": data,
            "timestamp": timestamp
        }
        self.nodes[entity_id] = node

        if generated_by:
             self.edges.append({
                    "source": entity_id,
                    "target": generated_by,
                    "relation": "prov:wasGeneratedBy"
                })
        self._save()
        return entity_id

    def query_provenance(self, entity_id: str) -> List[Dict]:
        """Returns the provenance trace for an entity."""
        trace = []
        # Simple immediate parent search
        for edge in self.edges:
            if edge['source'] == entity_id:
                trace.append(edge)
                target = self.nodes.get(edge['target'])
                if target:
                    trace.append(target)
        return trace
