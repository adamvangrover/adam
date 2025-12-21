import json
import logging
import os
from typing import Any, Dict, List

import networkx as nx

try:
    import jsonschema
except ImportError:
    jsonschema = None

from core.v23_graph_engine.unified_knowledge_graph import UnifiedKnowledgeGraph

logger = logging.getLogger(__name__)

class OdysseyKnowledgeGraph(UnifiedKnowledgeGraph):
    """
    Odyssey-specific extension of the UnifiedKnowledgeGraph.
    Adds support for FIBO schema validation and risk detection algorithms.
    """
    def __init__(self):
        super().__init__()
        self.odyssey_schema = None
        self._load_odyssey_schema()

    def _load_odyssey_schema(self):
        schema_path = "data/odyssey_fibo_schema.json"
        if os.path.exists(schema_path):
            try:
                with open(schema_path, "r") as f:
                    self.odyssey_schema = json.load(f)
                logger.info("Loaded Odyssey FIBO Schema.")
            except Exception as e:
                logger.error(f"Failed to load Odyssey schema: {e}")
        else:
            logger.warning(f"Odyssey Schema file not found at {schema_path}")

    def ingest_odyssey_entity(self, entity_data: Dict[str, Any]):
        """
        Ingests a specialized Odyssey Entity with strict schema validation.
        """
        if self.odyssey_schema and jsonschema:
            try:
                jsonschema.validate(instance=entity_data, schema=self.odyssey_schema)
            except jsonschema.ValidationError as e:
                logger.error(f"Schema Validation Failed: {e}")
                raise ValueError(f"Entity does not conform to Odyssey FIBO Schema: {e}")
        elif not jsonschema:
            logger.warning("jsonschema not installed, skipping validation.")

        node_id = entity_data.get("@id")
        if not node_id:
             raise ValueError("Entity missing @id")

        self.graph.add_node(
            node_id,
            type=entity_data.get("@type"),
            legal_name=entity_data.get("legalName"),
            data=entity_data
        )

        if "hasCreditFacility" in entity_data:
            for facility in entity_data["hasCreditFacility"]:
                fac_id = facility.get("@id")
                if fac_id:
                    self.graph.add_node(
                        fac_id,
                        type="CreditFacility",
                        facility_type=facility.get("facilityType"),
                        has_jcrew_blocker=facility.get("hasJCrewBlocker", False),
                        interest_coverage_ratio=facility.get("interestCoverageRatio")
                    )
                    self.graph.add_edge(node_id, fac_id, relation="BORROWS")

    def detect_fractured_ouroboros(self) -> List[List[str]]:
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return [c for c in cycles if len(c) > 1]
        except Exception as e:
            logger.error(f"Cycle detection failed: {e}")
            return []

    def detect_j_crew_maneuver(self, entity_id: str) -> Dict[str, Any]:
        risk_report = {"detected": False, "details": []}

        if entity_id not in self.graph:
            return risk_report

        unrestricted_subs = []
        for n in self.graph.successors(entity_id):
            node_type = self.graph.nodes[n].get("type")
            if node_type == "lending:UnrestrictedSubsidiary":
                unrestricted_subs.append(n)

        if not unrestricted_subs:
            return risk_report

        facilities = []
        for n in self.graph.successors(entity_id):
             # Assuming relation is stored in edge attributes.
             # Original UnifiedKnowledgeGraph stores relation in edge data.
             if self.graph[entity_id][n].get("relation") == "BORROWS":
                 facilities.append(n)

        vulnerable_facilities = []
        for fac in facilities:
            has_blocker = self.graph.nodes[fac].get("has_jcrew_blocker", False)
            if not has_blocker:
                vulnerable_facilities.append(fac)

        if vulnerable_facilities:
            risk_report["detected"] = True
            risk_report["details"].append(f"Entity has {len(unrestricted_subs)} Unrestricted Subsidiaries.")
            risk_report["details"].append(f"Found {len(vulnerable_facilities)} Vulnerable Facilities missing J.Crew Blockers: {vulnerable_facilities}")

        return risk_report
