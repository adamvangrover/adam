import networkx as nx
import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class UnifiedKnowledgeGraph:
    """
    The central brain of the system.
    Merges:
    1. Financial Entities (Companies, Sectors)
    2. System Entities (Agents, Tools - via RepoGraph)
    3. Memory Artifacts (Past Analysis)
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def ingest_repo_graph(self, repo_graph_data: Dict[str, Any]):
        """Ingests the Repo Graph (Self-Awareness)."""
        logger.info("Ingesting Repo Graph...")
        subgraph = nx.node_link_graph(repo_graph_data)
        self.graph = nx.compose(self.graph, subgraph)
        logger.info(f"Graph now has {self.graph.number_of_nodes()} nodes.")

    def ingest_financial_data(self, companies: List[Dict[str, Any]]):
        """Ingests financial entities."""
        for company in companies:
            node_id = company.get("symbol") or company.get("company_id")
            if not node_id: continue

            self.graph.add_node(
                node_id,
                type="Company",
                sector=company.get("sector"),
                description=company.get("description")
            )
            # Add sector node
            if company.get("sector"):
                self.graph.add_node(company["sector"], type="Sector")
                self.graph.add_edge(node_id, company["sector"], relation="belongs_to")

    def ingest_memory_vectors(self, memory_entries: List[Dict[str, Any]]):
        """Ingests past analysis as Memory Nodes."""
        for entry in memory_entries:
            node_id = f"Memory::{entry['company_id']}::{entry['timestamp']}"
            self.graph.add_node(
                node_id,
                type="Memory",
                summary=entry.get("analysis_summary"),
                timestamp=entry.get("timestamp")
            )
            # Link to Company
            if entry.get("company_id"):
                self.graph.add_edge(node_id, entry["company_id"], relation="analyzes")

    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        # Placeholder for graph traversal/search
        # For now, return neighbors of a node if query matches a node ID
        if query in self.graph:
            return [
                {"node": n, "relation": self.graph[query][n].get("relation")}
                for n in self.graph.neighbors(query)
            ]
        return []

    def save_snapshot(self, filepath: str = "data/knowledge_graph_snapshot.json"):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
