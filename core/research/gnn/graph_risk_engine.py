from typing import List, Dict, Any
import math
import random

class GraphRiskEngine:
    """
    Simulates a GNN-based engine for predicting contagion risk in a financial network.
    """

    def __init__(self):
        self.embedding_dim = 64

    def predict_risk(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Simulates message passing to update node embeddings and predict risk score.
        """
        # Mock GNN Forward Pass
        # 1. Initialize Node Embeddings (Random for simulation)
        node_embeddings = {n['id']: [random.random() for _ in range(self.embedding_dim)] for n in nodes}

        # 2. Message Passing (Simulated)
        # In a real GNN: H_next = A * H_curr * W
        # Here we just aggregate neighbor "risk factors"

        risk_scores = {}
        for node in nodes:
            node_id = node['id']
            # Base risk from node attributes
            base_risk = random.uniform(0.1, 0.4)

            # Neighbor influence
            neighbor_risk = 0.0
            neighbors = [e for e in edges if e['source'] == node_id or e['target'] == node_id]

            for edge in neighbors:
                # Stronger link = higher contagion
                weight = edge.get('weight', 1.0)
                neighbor_risk += weight * 0.05

            total_risk = min(base_risk + neighbor_risk, 0.99)
            risk_scores[node_id] = total_risk

        return risk_scores
