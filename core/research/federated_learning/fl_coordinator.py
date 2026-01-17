import torch
import copy
from .model import CreditRiskModel
from .fl_client import FederatedClient

class FederatedCoordinator:
    """
    Manages the Federated Learning process (aggregation).
    """
    def __init__(self, num_clients=3, input_dim=10):
        self.global_model = CreditRiskModel(input_dim=input_dim)
        self.clients = [FederatedClient(client_id=f"Bank_{i}", input_dim=input_dim) for i in range(num_clients)]
        self.round_history = []

    def aggregate_weights(self, client_weights):
        """
        FedAvg: Averages the weights from all clients.
        """
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(client_weights))
        return avg_weights

    def run_round(self, round_num):
        """Executes one round of Federated Learning."""
        print(f"--- FL Round {round_num} ---")
        global_weights = self.global_model.state_dict()

        client_weights_list = []
        round_losses = []
        round_accuracies = []

        for client in self.clients:
            # 1. Download global model
            client.set_weights(global_weights)

            # 2. Local Training
            loss = client.train_epoch(epochs=5)

            # 3. Upload local updates
            client_weights_list.append(client.get_weights())

            # Evaluate
            _, acc = client.evaluate()
            round_losses.append(loss)
            round_accuracies.append(acc)
            # print(f"  {client.client_id}: Loss={loss:.4f}, Acc={acc:.4f}")

        # 4. Aggregate
        new_global_weights = self.aggregate_weights(client_weights_list)
        self.global_model.load_state_dict(new_global_weights)

        avg_loss = sum(round_losses) / len(round_losses)
        avg_acc = sum(round_accuracies) / len(round_accuracies)

        self.round_history.append({
            "round": round_num,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_acc
        })
        print(f"  Aggregated: Avg Loss={avg_loss:.4f}, Avg Acc={avg_acc:.4f}")
        return avg_loss, avg_acc
