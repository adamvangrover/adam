import torch
import torch.optim as optim
import copy
from .model import CreditRiskModel
from core.research.gnn.model import GAT
from .privacy import PrivacyEngine

class FederatedClient:
    """
    Simulates a financial institution (Bank) participating in Federated Learning.
    Holds local, private data.
    """
    def __init__(self, client_id, data_size=100, input_dim=10):
        self.client_id = client_id
        self.data_size = data_size
        self.input_dim = input_dim
        self.model = CreditRiskModel(input_dim=input_dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = torch.nn.BCELoss()

        # Generate synthetic private data
        # Feature 0: High means risky
        self.X = torch.randn(data_size, input_dim)
        # Simple rule for ground truth: if feature 0 + feature 1 > 1 => Default (1)
        logits = self.X[:, 0] + self.X[:, 1]
        self.y = (torch.sigmoid(logits) > 0.5).float().view(-1, 1)

    def set_weights(self, global_state_dict):
        """Updates local model with global weights."""
        self.model.load_state_dict(global_state_dict)

    def train_epoch(self, epochs=1):
        """Trains locally for a few epochs."""
        self.model.train()
        epoch_loss = 0.0
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X)
            loss = self.criterion(outputs, self.y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / epochs

    def get_weights(self):
        """Returns local model weights."""
        return copy.deepcopy(self.model.state_dict())

    def evaluate(self):
        """Evaluates model on local data."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X)
            loss = self.criterion(outputs, self.y).item()
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == self.y).sum().item() / self.data_size
        return loss, accuracy

class FinGraphFLClient:
    """
    Advanced Client using Graph Neural Networks and Differential Privacy.
    """
    def __init__(self, client_id, data_size=100, input_dim=10, use_privacy=True):
        self.client_id = client_id
        self.data_size = data_size
        self.input_dim = input_dim

        # Use GAT for structural learning
        # Note: input_dim for GAT is nfeat
        self.model = GAT(nfeat=input_dim, nhid=16, nclass=1)

        # Adam optimizer as per report
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        self.criterion = torch.nn.BCELoss()

        self.privacy = PrivacyEngine(epsilon=0.5) if use_privacy else None

        # Synthetic Graph Data
        self.X = torch.randn(data_size, input_dim)

        # Generate random graph structure (Adjacency Matrix)
        # Dense for simplicity in this demo, but GAT handles it
        self.adj = torch.eye(data_size)
        # Add random edges (density 0.1)
        mask = torch.rand(data_size, data_size) < 0.1
        self.adj[mask] = 1.0
        # Make symmetric
        self.adj = (self.adj + self.adj.t()) > 0
        self.adj = self.adj.float()

        # Labels (depend on neighbors for graph effect)
        # y = sigmoid(X0 + Neighbor_X0)
        neighbor_sum = torch.mm(self.adj, self.X)
        logits = self.X[:, 0] + 0.5 * neighbor_sum[:, 0]
        self.y = (torch.sigmoid(logits) > 0.5).float().view(-1, 1)

    def set_weights(self, global_state_dict):
        """Updates local model with global weights."""
        self.model.load_state_dict(global_state_dict)

    def train_epoch(self, epochs=1):
        """Trains locally using GAT."""
        self.model.train()
        epoch_loss = 0.0
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X, self.adj)
            loss = self.criterion(outputs, self.y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / epochs

    def get_weights(self):
        """Returns local model weights with Differential Privacy noise."""
        weights = copy.deepcopy(self.model.state_dict())
        if self.privacy:
            weights = self.privacy.add_noise(weights)
        return weights

    def evaluate(self):
        """Evaluates model on local data."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X, self.adj)
            loss = self.criterion(outputs, self.y).item()
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == self.y).sum().item() / self.data_size
        return loss, accuracy
