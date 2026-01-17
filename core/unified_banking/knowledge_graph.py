import networkx as nx
import json
import logging
from typing import Dict, List, Any

class BankingKnowledgeGraph:
    """
    Graph Engine for the Unified Banking World Model.
    Models relationships between Entities, Assets, and Orders to perform Contagion Analysis.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger(__name__)

    def load_from_simulation_log(self, log_path: str):
        """
        Ingests the simulation log (Gold Layer) to build the graph.
        """
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)

            # 1. Create Nodes for Static Entities
            self.graph.add_node("FAMILY_OFFICE_A", type="Entity", role="Client", sector="Wealth")
            self.graph.add_node("IB_MARKET_MAKER", type="Entity", role="Desk", sector="InvestmentBank")
            self.graph.add_node("JPM_HY_CREDIT", type="Asset", asset_class="Credit", rating="HY")

            # 2. Process Ticks/Orders to create Edges
            ticks = data.get("ticks", [])

            # Edge: Entity holds Asset (Inventory)
            # Market Maker Inventory Edge with dynamic attribute
            final_inventory = ticks[-1]["mm_inventory_end"]
            self.graph.add_edge("IB_MARKET_MAKER", "JPM_HY_CREDIT", relation="HOLDS", quantity=final_inventory)

            # Edge: Client Sold Asset
            # Aggregated flow
            total_sold = sum(o["filled"] for o in data.get("orders", []))
            self.graph.add_edge("FAMILY_OFFICE_A", "JPM_HY_CREDIT", relation="SOLD", quantity=total_sold)

            # 3. Add Scenario Context
            scenario = data["metadata"]["scenario"]
            self.graph.add_node(scenario, type="Scenario", severity="High")

            # Connect Scenario to Asset (Shock)
            # Since price dropped, Scenario IMPACTS Asset negative
            self.graph.add_edge(scenario, "JPM_HY_CREDIT", relation="STRESSES", impact="Price_Depreciation")

            self.logger.info(f"Graph loaded with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

        except Exception as e:
            self.logger.error(f"Failed to load graph: {e}")

    def analyze_contagion(self, start_node: str, depth: int = 2) -> Dict[str, Any]:
        """
        Performs a traversal to find impacted nodes.
        Example: If 'JPM_HY_CREDIT' is stressed, who holds it?
        """
        if start_node not in self.graph:
            return {"error": "Node not found"}

        impacted = {}
        # Simple BFS
        successors = list(nx.bfs_tree(self.graph, start_node, depth_limit=depth))

        # Also check predecessors (Who interacts with this node?)
        # For contagion, if Asset falls, Holders are hurt (Predecessors in a HOLDS relationship)
        # But my edge direction was Holder -> Asset. So I need in_edges.

        holders = []
        for u, v, data in self.graph.in_edges(start_node, data=True):
            if data.get("relation") == "HOLDS":
                holders.append({"entity": u, "exposure": data.get("quantity")})

        return {
            "source": start_node,
            "downstream_impacts": successors,
            "direct_holders_at_risk": holders
        }

    def export_json(self) -> Dict[str, Any]:
        """
        Exports graph to JSON compatible with Vis.js or D3 force-directed.
        """
        return nx.node_link_data(self.graph)

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    kg = BankingKnowledgeGraph()
    kg.load_from_simulation_log("showcase/data/unified_banking_log.json")

    print("\n--- Contagion Analysis: JPM_HY_CREDIT ---")
    analysis = kg.analyze_contagion("JPM_HY_CREDIT")
    print(json.dumps(analysis, indent=2))
