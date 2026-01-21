import torch
import torch.nn as nn
import torch.nn.functional as F

class CreditRiskModel(nn.Module):
    """
    A robust MLP for predicting credit risk (probability of default).
    Includes BatchNorm and Dropout for better generalization in FL settings.
    """
    def __init__(self, input_dim=10, hidden_dim=64):
        super(CreditRiskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        # Handle batch norm for single samples or small batches if needed
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = torch.sigmoid(self.fc3(x))
        return x
