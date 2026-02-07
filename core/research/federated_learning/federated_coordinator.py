from typing import Dict, List
from .federated_client import FederatedClient

class FederatedCoordinator:
    """
    Manages the FedAvg (Federated Averaging) algorithm.
    """

    def __init__(self, num_clients: int = 5):
        self.clients = [FederatedClient(f"Bank_{i}") for i in range(num_clients)]
        self.global_model = {
            "layer_1_weight": 0.5,
            "layer_1_bias": 0.1,
            "layer_2_weight": -0.2,
            "layer_2_bias": 0.05
        }
        self.round = 0

    def start_round(self) -> Dict[str, float]:
        self.round += 1
        client_updates = []

        # 1. Distribute Global Model
        for client in self.clients:
            # 2. Local Training
            updated_weights = client.train_epoch(self.global_model)
            client_updates.append(updated_weights)

        # 3. Aggregation (FedAvg)
        self.global_model = self._aggregate(client_updates)
        return self.global_model

    def _aggregate(self, updates: List[Dict[str, float]]) -> Dict[str, float]:
        if not updates:
            return self.global_model

        avg_weights = {}
        for key in self.global_model.keys():
            total = sum(u[key] for u in updates)
            avg_weights[key] = total / len(updates)

        return avg_weights
