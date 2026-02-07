from typing import Dict, List, Any
import random
import time

class FederatedClient:
    """
    Simulates a Bank/Institution training a local model on private data.
    """
    def __init__(self, client_id: str, data_size: int = 1000):
        self.client_id = client_id
        self.data_size = data_size
        self.local_model_weights = {}

    def train_epoch(self, global_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Simulates local training.
        """
        # In reality, this would run PyTorch/TensorFlow training loop
        # Here we simulate weight updates based on "local data characteristics"

        new_weights = global_weights.copy()

        # Simulate gradient descent
        for layer, weight in new_weights.items():
            # Apply some noise/drift representing local data optimization
            drift = random.uniform(-0.01, 0.01)
            new_weights[layer] = weight + drift

        self.local_model_weights = new_weights
        return new_weights

    def evaluate(self, weights: Dict[str, float]) -> float:
        """Returns validation accuracy/loss."""
        return random.uniform(0.85, 0.95)
