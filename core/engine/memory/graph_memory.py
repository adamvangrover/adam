import logging
import networkx as nx
import json
import os
from typing import List, Dict, Any

class GraphMemoryManager:
    """
    Manages entity relationship graphs using NetworkX.
    Acts as an in-memory substitute for a Neo4j database, allowing us to map
    Systemic Risks and Supply Chain linkages dynamically.
    """
    
    def __init__(self, data_path: str = "./data/relationship_graph.json"):
        self.data_path = data_path
        self.graph = nx.DiGraph()
        
        logging.info("Initializing GraphMemoryManager (NetworkX)...")
        self._load_seed_data()
        
    def _load_seed_data(self):
        """
        Loads the initial intelligence from the predefined JSON structure or establishes a base seed.
        """
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    
                # We expect data to be a list of edges [Source, Relationship, Target]
                for edge in data:
                    if len(edge) == 3:
                        source, rel, target = edge
                        self.add_relationship(source, rel, target)
                        
                logging.info(f"Loaded {self.graph.number_of_edges()} relationships from {self.data_path}")
            except Exception as e:
                logging.error(f"Failed to load graph data: {e}. Falling back to default seed.")
                self._apply_default_seed()
        else:
            logging.info("Creating net-new Graph Memory Seed.")
            self._apply_default_seed()
            self.save_graph()
            
    def _apply_default_seed(self):
        # 1. Supply Chain Subgraph
        self.add_relationship("NVDA", "depends_on", "TSMC")
        self.add_relationship("TSMC", "depends_on", "ASML")
        self.add_relationship("ASML", "vulnerable_to", "Geopolitical Restrictions")
        
        # 2. Systemic Risk Subgraph (From Priority 0 / Omnibus Report)
        self.add_relationship("Shadow Banking", "exposes", "Tier 1 Banks")
        self.add_relationship("Shadow Banking", "funds", "Direct Lenders")
        self.add_relationship("Direct Lenders", "highly_exposed_to", "Interest Rate Volatility")
        
        # 3. Equities / Macro
        self.add_relationship("AAPL", "depends_on", "Consumer Spending")
        self.add_relationship("Consumer Spending", "inversely_correlated_with", "Inflation")

    def add_relationship(self, source_entity: str, relationship_type: str, target_entity: str, attributes: Dict[str, Any] = None):
        """
        Adds a semantic relationship between two entities.
        """
        self.graph.add_edge(source_entity, target_entity, relation=relationship_type, **(attributes or {}))

    def save_graph(self):
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        # Format edges as [Source, Relation, Target] list for JSON
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append([u, data.get('relation', 'connected_to'), v])
            
        with open(self.data_path, 'w') as f:
            json.dump(edges, f, indent=4)
            
    def query_relationships(self, start_entity: str, radius: int = 1) -> List[Dict[str, str]]:
        """
        Given a starting entity, traverse the graph up to `radius` depth to find dependencies and risks.
        Returns a list of structured path dictionaries.
        """
        logging.info(f"GraphMemory querying dependencies for {start_entity} (Radius: {radius})")
        
        if start_entity not in self.graph:
            logging.warning(f"Entity '{start_entity}' not found in active Graph Memory.")
            return []
            
        # Extract ego graph (subgraph neighbors up to radius)
        subgraph = nx.ego_graph(self.graph, start_entity, radius=radius, undirected=False)
        
        results = []
        # Build human readable relationship chains
        for u, v, data in subgraph.edges(data=True):
            rel = data.get('relation', 'connected_to')
            results.append({
                "source": u,
                "relationship": rel,
                "target": v,
                "synthesized_narrative": f"[{u}] -> ({rel}) -> [{v}]"
            })
            
        return results
