import torch
import torch.optim as optim
import copy
import numpy as np
from .model import CreditRiskModel

class FederatedClient:
    """
    Simulates a financial institution (Bank) participating in Federated Learning.
    Holds local, private data.
    """
    def __init__(self, client_id, data_size=100, input_dim=10, sector_bias=None):
        self.client_id = client_id
        self.data_size = data_size
        self.input_dim = input_dim
        self.model = CreditRiskModel(input_dim=input_dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = torch.nn.BCELoss()

        # Generate synthetic private data with Non-IID characteristics (Sector Bias)
        self.X = torch.randn(data_size, input_dim)

        if sector_bias is not None:
            # Shift the distribution of certain features based on sector bias
            # sector_bias is a dict {feature_index: shift_value}
            # e.g., {0: 2.0} means feature 0 is shifted by +2.0
            for idx, shift in sector_bias.items():
                if idx < input_dim:
                    self.X[:, idx] += shift

        # Ground Truth Logic
        # If Feature 0 (Leverage) + Feature 1 (Volatility) > 1 => Default
        # Adding some noise
        logits = self.X[:, 0] + self.X[:, 1] + torch.randn(data_size) * 0.1
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
