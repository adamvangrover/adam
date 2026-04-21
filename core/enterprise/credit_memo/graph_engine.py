import networkx as nx
import os
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
import logging
from typing import List, Dict, Any

class GraphEngine:
    """
    Simulates Neo4j GraphRAG with NetworkX fallback.
    Protocol: Enterprise Knowledge Graph
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.neo4j_driver = None

        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")

        if GraphDatabase and uri and user and password:
            try:
                self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
                # Verify connectivity
                self.neo4j_driver.verify_connectivity()
                logging.getLogger(__name__).info("Successfully connected to Neo4j database. Using real graph engine.")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to connect to Neo4j ({e}). Falling back to mock graph.")
                self.neo4j_driver = None
                self._build_mock_graph()
        else:
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

        # --- Apple Inc. ---
        self.graph.add_node("Apple Inc.", type="Borrower", risk="Low")

        self.graph.add_node("Google (Alphabet)", type="Counterparty", risk="High")
        self.graph.add_edge("Apple Inc.", "Google (Alphabet)", relationship="SEARCH_AGREEMENT_ANTITRUST")

        self.graph.add_node("European Commission", type="Regulator", risk="High")
        self.graph.add_edge("Apple Inc.", "European Commission", relationship="DMA_COMPLIANCE_PROBE")

        self.graph.add_node("Foxconn", type="Supplier", risk="Medium")
        self.graph.add_edge("Apple Inc.", "Foxconn", relationship="MANUFACTURING_PARTNER")

        self.graph.add_node("TSMC", type="Supplier", risk="Low")
        self.graph.add_edge("Apple Inc.", "TSMC", relationship="SOLE_SOURCE_CHIPS")

        # --- Tesla Inc. ---
        self.graph.add_node("Tesla Inc.", type="Borrower", risk="Medium")

        self.graph.add_node("SpaceX", type="Related Party", risk="Medium")
        self.graph.add_edge("Tesla Inc.", "SpaceX", relationship="SHARED_CEO_ELON_MUSK")

        self.graph.add_node("X (Twitter)", type="Related Party", risk="High")
        self.graph.add_edge("Tesla Inc.", "X (Twitter)", relationship="MARGIN_LOAN_COLLATERAL")

        self.graph.add_node("xAI", type="Related Party", risk="High")
        self.graph.add_edge("Tesla Inc.", "xAI", relationship="AI_MODEL_LICENSING")

        self.graph.add_node("BYD", type="Competitor", risk="Medium")
        self.graph.add_edge("Tesla Inc.", "BYD", relationship="MARKET_SHARE_RIVALRY")

        # --- JPMorgan Chase ---
        self.graph.add_node("JPMorgan Chase", type="Borrower", risk="Low") # Bank as borrower

        self.graph.add_node("Federal Reserve", type="Regulator", risk="None")
        self.graph.add_edge("JPMorgan Chase", "Federal Reserve", relationship="PRIMARY_DEALER_GSIB")

        self.graph.add_node("FDIC", type="Regulator", risk="None")
        self.graph.add_edge("JPMorgan Chase", "FDIC", relationship="FIRST_REPUBLIC_ACQUISITION")

        self.graph.add_node("Visa Inc.", type="Partner", risk="Low")
        self.graph.add_edge("JPMorgan Chase", "Visa Inc.", relationship="CO_BRAND_CARD_AGREEMENT")


    def query_relationships(self, entity_name: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Finds connected entities up to `depth`.
        """
        if self.neo4j_driver:
            try:
                query = (
                    "MATCH path = (n {name: $entity_name})-[*1..$depth]-(m) "
                    "RETURN [node in nodes(path) | node.name] AS path, "
                    "labels(m)[0] AS type, m.risk AS risk_level"
                )
                with self.neo4j_driver.session() as session:
                    result = session.run(query, entity_name=entity_name, depth=depth)
                    results = []
                    for record in result:
                        path = " -> ".join(record["path"])
                        results.append({
                            "entity": record["path"][-1],
                            "type": record["type"],
                            "risk_level": record["risk_level"],
                            "path": path
                        })
                return results
            except Exception as e:
                logging.getLogger(__name__).warning(f"Neo4j query failed ({e}). Falling back to mock graph.")
                # We do not switch to mock mid-query unless we build it first.
                if len(self.graph.nodes) == 0:
                    self._build_mock_graph()

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
