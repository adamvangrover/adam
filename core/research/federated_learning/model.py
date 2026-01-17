import torch
import torch.nn as nn
import torch.nn.functional as F

class CreditRiskModel(nn.Module):
    """
    A simple MLP for predicting credit risk (probability of default).
    Input: Features (e.g., Debt-to-Income, Credit Score, Loan Amount, etc.)
    Output: Probability of Default (0-1)
    """
    def __init__(self, input_dim=10, hidden_dim=64):
        super(CreditRiskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
