import json
import os
import networkx as nx
from typing import Dict, List

class ContagionEngine:
    """
    Models second-order risk propagation using a directed graph of sector dependencies.
    """

    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sector_relationships.json')
        self.graph = self._build_graph()

    def _build_graph(self):
        g = nx.DiGraph()
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                for link in data.get('relationships', []):
                    g.add_edge(link['source'], link['target'], weight=link['weight'])
        except Exception as e:
            # Fallback graph
            g.add_edge("Energy", "Industrials", weight=0.5)
        return g

    def simulate_contagion(self, initial_shocks: Dict[str, float], decay: float = 0.5) -> Dict[str, Any]:
        """
        Propagates shocks through the network.

        Args:
            initial_shocks: Dict of {Sector: ImpactScore} (e.g. -0.5 for 50% distress)
            decay: Attenuation factor for secondary impacts.

        Returns:
            Dict containing 'final_impacts' and 'propagation_log'.
        """
        impacts = initial_shocks.copy()
        log = []

        # 1-Step Propagation
        # For every shocked node, transmit shock to neighbors

        second_order = {}

        for source, shock in initial_shocks.items():
            if source not in self.graph:
                continue

            # If shock is negative (distress), propagate it
            if shock < 0:
                neighbors = self.graph.successors(source)
                for target in neighbors:
                    edge_weight = self.graph[source][target]['weight']
                    # Transmitted Shock = Original Shock * Dependency Weight * Decay
                    transmission = shock * edge_weight * decay

                    current_impact = second_order.get(target, 0.0)
                    second_order[target] = current_impact + transmission

                    log.append(f"{source} distress ({shock}) impacted {target} by {transmission:.2f}")

        # Merge impacts
        final_impacts = impacts.copy()
        for sector, impact in second_order.items():
            existing = final_impacts.get(sector, 0.0)
            final_impacts[sector] = existing + impact # Additive distress

        return {
            "final_impacts": final_impacts,
            "log": log
        }
