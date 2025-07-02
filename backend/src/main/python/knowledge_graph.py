from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Re-using models from models.py is ideal, but for tool environment simplicity,
# we might redefine or assume they are available.
# from .models import Company, Industry, Driver, MacroEnvironmentFactor # etc.

@dataclass
class Node:
    id: str
    label: str  # e.g., 'Company', 'Industry', 'Driver'
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    source_id: str
    target_id: str
    relationship_type: str  # e.g., 'BELONGS_TO_INDUSTRY', 'AFFECTED_BY_DRIVER'
    properties: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        # For quick lookups
        self.adj: Dict[str, List[Edge]] = {} # From source to list of edges
        self.rev_adj: Dict[str, List[Edge]] = {} # To target from list of edges

    def add_node(self, node_id: str, label: str, properties: Optional[Dict[str, Any]] = None):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(id=node_id, label=label, properties=properties or {})
            self.adj[node_id] = []
            self.rev_adj[node_id] = []
        else:
            # Optionally update properties if node exists
            if properties:
                self.nodes[node_id].properties.update(properties)
        return self.nodes[node_id]

    def add_edge(self, source_id: str, target_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None):
        if source_id not in self.nodes:
            # Or raise error: print(f"Warning: Source node {source_id} not found for edge.")
            return
        if target_id not in self.nodes:
            # Or raise error: print(f"Warning: Target node {target_id} not found for edge.")
            return

        edge = Edge(source_id=source_id, target_id=target_id, relationship_type=relationship_type, properties=properties or {})
        self.edges.append(edge)
        self.adj[source_id].append(edge)
        self.rev_adj[target_id].append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[Node]:
        """ Get outgoing neighbors """
        neighbors = []
        if node_id in self.adj:
            for edge in self.adj[node_id]:
                if relationship_type is None or edge.relationship_type == relationship_type:
                    neighbors.append(self.nodes[edge.target_id])
        return neighbors

    def get_incoming_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[Node]:
        """ Get incoming neighbors """
        neighbors = []
        if node_id in self.rev_adj:
            for edge in self.rev_adj[node_id]:
                if relationship_type is None or edge.relationship_type == relationship_type:
                    neighbors.append(self.nodes[edge.source_id])
        return neighbors

    def find_nodes_by_label(self, label: str) -> List[Node]:
        return [node for node in self.nodes.values() if node.label == label]

    def __str__(self):
        return f"KnowledgeGraph(Nodes: {len(self.nodes)}, Edges: {len(self.edges)})"

# Example usage (will be integrated into the data loader)
if __name__ == '__main__':
    kg = KnowledgeGraph()
    kg.add_node("AAPL", "Company", {"name": "Apple Inc."})
    kg.add_node("IND_TECH", "Industry", {"name": "Technology"})
    kg.add_edge("AAPL", "IND_TECH", "BELONGS_TO_INDUSTRY")

    print(kg)
    print(kg.get_node("AAPL"))
    print(kg.get_neighbors("AAPL", "BELONGS_TO_INDUSTRY"))
