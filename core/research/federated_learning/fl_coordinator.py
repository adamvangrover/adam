import torch
import copy
from .model import CreditRiskModel
from .fl_client import FederatedClient, FinGraphFLClient
from core.research.gnn.model import GAT
from .privacy import MSGuard

class FederatedCoordinator:
    """
    Manages the Federated Learning process (aggregation). 

[Image of federated learning architecture]

    Supports 'Standard' (MLP) and 'FinGraphFL' (GAT + Privacy) modes.
    """

    def __init__(self, num_clients=3, input_dim=10, client_configs=None, mode="Standard"):
        self.mode = mode
        self.round_history = []

        # 1. Initialize Global Model based on Mode
        if self.mode == "FinGraphFL":
            print("Initializing FinGraphFL: GAT models + Differential Privacy + MSGuard")
            self.global_model = GAT(nfeat=input_dim, nhid=16, nclass=1)
        else:
            self.global_model = CreditRiskModel(input_dim=input_dim)

        # 2. Initialize Clients (Handling Configs + Mode)
        self.clients = []
        
        if self.mode == "FinGraphFL":
            # Graph/Privacy Mode
            # Uses FinGraphFLClient. Respects IDs from config if present, ignoring sector_bias 
            # as graph topology usually overrides simple bias params.
            if client_configs:
                for cfg in client_configs:
                    self.clients.append(FinGraphFLClient(client_id=cfg['id'], input_dim=input_dim))
            else:
                self.clients = [FinGraphFLClient(client_id=f"Bank_{i}", input_dim=input_dim) for i in range(num_clients)]
        
        else:
            # Standard Mode
            # Uses standard FederatedClient. Fully supports sector_bias from feature branch.
            if client_configs:
                for cfg in client_configs:
                    self.clients.append(FederatedClient(
                        client_id=cfg['id'],
                        input_dim=input_dim,
                        sector_bias=cfg.get('sector_bias')
                    ))
            else:
                self.clients = [FederatedClient(client_id=f"Bank_{i}", input_dim=input_dim) for i in range(num_clients)]

    def aggregate_weights(self, client_weights):
        """
        Aggregates weights. Applies MSGuard to filter malicious updates.
        """
        # Apply MSGuard Defense
        valid_weights = MSGuard.filter_updates(client_weights)

        if len(valid_weights) < len(client_weights):
            print(f"  [MSGuard] Blocked {len(client_weights) - len(valid_weights)} suspicious update(s).")

        if not valid_weights:
            print("  [Warning] All updates filtered! Fallback to raw averaging.")
            valid_weights = client_weights

        # FedAvg: Average the weights
        avg_weights = copy.deepcopy(valid_weights[0])
        for key in avg_weights.keys():
            # Check for float tensors to avoid aggregating LongTensors (if any, usually none in weights)
            if avg_weights[key].dtype in [torch.float32, torch.float64]:
                for i in range(1, len(valid_weights)):
                    avg_weights[key] += valid_weights[i][key]
                avg_weights[key] = torch.div(avg_weights[key], len(valid_weights))
            else:
                # For integer buffers (like batches_tracked), usually take the first one or max
                pass

        return avg_weights

    def run_round(self, round_num):
        """Executes one round of Federated Learning."""
        print(f"--- FL Round {round_num} ({self.mode}) ---")
        global_weights = self.global_model.state_dict()

        client_weights_list = []
        round_losses = []
        round_accuracies = []

        for client in self.clients:
            # 1. Download global model
            client.set_weights(global_weights)

            # 2. Local Training
            loss = client.train_epoch(epochs=5)

            # 3. Upload local updates (potentially noisy if FinGraphFL)
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