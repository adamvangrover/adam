import torch
import torch.optim as optim
import copy
from .model import CreditRiskModel

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
