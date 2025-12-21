from core.system.repo_graph import RepoGraphBuilder
from core.v23_graph_engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.system.memory_manager import VectorMemoryManager
import logging
import json
import networkx as nx

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Orchestrates the creation of the Comprehensive Memory.
    """

    def __init__(self):
        self.ukg = UnifiedKnowledgeGraph()
        self.memory_manager = VectorMemoryManager()
        self.repo_builder = RepoGraphBuilder(root_dir=".")

    def consolidate(self):
        logger.info("Starting Memory Consolidation...")

        # 1. Build Self-Awareness (Repo Graph)
        repo_graph = self.repo_builder.build()
        self.ukg.ingest_repo_graph(nx.node_link_data(repo_graph))

        # 2. Ingest Long-Term Memory
        history = self.memory_manager.load_history()
        self.ukg.ingest_memory_vectors(history)

        # 3. Ingest Financial Knowledge (from Memory companies)
        # In a real system, we would pull from a master entity DB.
        # Here we extract unique companies from history to populate the graph.
        companies = {}
        for h in history:
            cid = h.get("company_id")
            if cid and cid not in companies:
                # In a real scenario, we'd fetch details. Here we placeholder.
                companies[cid] = {"symbol": cid, "sector": "Technology", "description": "Extracted from Memory"}

        self.ukg.ingest_financial_data(list(companies.values()))

        # 4. Save Snapshot
        self.ukg.save_snapshot()
        logger.info("Memory Consolidation Complete.")

    def generate_system_manifest(self) -> str:
        """Generates a text description of the system state."""
        graph_data = nx.node_link_data(self.ukg.graph)
        node_types = {}
        for node in graph_data['nodes']:
            ntype = node.get('type', 'Unknown')
            node_types[ntype] = node_types.get(ntype, 0) + 1

        manifest = f"# System Manifest\n\n"
        manifest += f"## Knowledge Graph Stats\n"
        manifest += f"- Total Nodes: {len(graph_data['nodes'])}\n"
        edges = graph_data.get('links', graph_data.get('edges', []))
        manifest += f"- Total Edges: {len(edges)}\n"
        manifest += "\n## Node Types\n"
        for t, c in node_types.items():
            manifest += f"- {t}: {c}\n"

        return manifest
