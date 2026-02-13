import networkx as nx
import logging
from typing import List, Dict, Any

class GraphEngine:
    """
    Simulates Neo4j GraphRAG with NetworkX fallback.
    Protocol: Enterprise Knowledge Graph
    """
    def __init__(self):
        self.graph = nx.Graph()
        self._build_mock_graph()

    def _build_mock_graph(self):
        """
        Populate mock graph with entities and relationships.
        """
        # --- TechCorp Inc. ---
        self.graph.add_node("TechCorp Inc.", type="Borrower", risk="High")
        self.graph.add_node("TechGlobal Holdings", type="Guarantor", risk="Medium")
        self.graph.add_edge("TechCorp Inc.", "TechGlobal Holdings", relationship="SUBSIDIARY_OF")

        self.graph.add_node("ChipFoundry Taiwan", type="Supplier", risk="Low")
        self.graph.add_edge("TechCorp Inc.", "ChipFoundry Taiwan", relationship="MAJOR_SUPPLIER")

        self.graph.add_node("NextGen AI Ltd.", type="Investment", risk="Speculative")
        self.graph.add_edge("TechGlobal Holdings", "NextGen AI Ltd.", relationship="INVESTED_IN")

        # --- Apple Inc. ---
        self.graph.add_node("Apple Inc.", type="Borrower", risk="Low")
        self.graph.add_node("Foxconn Technology Group", type="Supplier", risk="Medium")
        self.graph.add_edge("Apple Inc.", "Foxconn Technology Group", relationship="STRATEGIC_PARTNER")

        self.graph.add_node("TSMC", type="Supplier", risk="Low")
        self.graph.add_edge("Apple Inc.", "TSMC", relationship="SOLE_SOURCE_CHIPS")

        self.graph.add_node("Qualcomm Inc.", type="Supplier", risk="Medium")
        self.graph.add_edge("Apple Inc.", "Qualcomm Inc.", relationship="LITIGATION_HISTORY")
        self.graph.add_edge("Qualcomm Inc.", "Broadcom", relationship="COMPETITOR") # Multi-hop

        # --- Tesla Inc. ---
        self.graph.add_node("Tesla Inc.", type="Borrower", risk="Medium")
        self.graph.add_node("SpaceX", type="Related Party", risk="Medium")
        self.graph.add_edge("Tesla Inc.", "SpaceX", relationship="SHARED_CEO")

        self.graph.add_node("Panasonic", type="Supplier", risk="Low")
        self.graph.add_edge("Tesla Inc.", "Panasonic", relationship="JV_PARTNER")

        self.graph.add_node("Twitter (X)", type="Related Party", risk="High")
        self.graph.add_edge("Tesla Inc.", "Twitter (X)", relationship="MARGIN_LOAN_COLLATERAL")

        # --- JPMorgan Chase ---
        self.graph.add_node("JPMorgan Chase", type="Borrower", risk="Low") # Bank as borrower (Repo/Line)
        self.graph.add_node("Federal Reserve", type="Regulator", risk="None")
        self.graph.add_edge("JPMorgan Chase", "Federal Reserve", relationship="PRIMARY_DEALER")

        self.graph.add_node("First Republic Assets", type="Acquisition", risk="Medium")
        self.graph.add_edge("JPMorgan Chase", "First Republic Assets", relationship="ACQUIRED_FROM_FDIC")


    def query_relationships(self, entity_name: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Finds connected entities up to `depth`.
        """
        # Fuzzy match for demo if exact name not found
        target = entity_name
        if target not in self.graph:
            # Try finding substring match
            for node in self.graph.nodes():
                if entity_name.lower() in node.lower() or node.lower() in entity_name.lower():
                    target = node
                    break

        if target not in self.graph:
            return []

        # Simple BFS
        subgraph = nx.ego_graph(self.graph, target, radius=depth)
        results = []
        for node in subgraph.nodes():
            if node == target:
                continue

            # Find path
            try:
                path = nx.shortest_path(self.graph, target, node)
                path_desc = " -> ".join(path)
                data = self.graph.nodes[node]
                results.append({
                    "entity": node,
                    "type": data.get("type", "Unknown"),
                    "risk_level": data.get("risk", "Unknown"),
                    "path": path_desc
                })
            except nx.NetworkXNoPath:
                continue

        return results

# Global Instance
graph_engine = GraphEngine()
