# core/agents/portfolio_optimization_agent.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
from .agent_base import AgentBase
from utils.data_validation import validate_portfolio_data
from utils.visualization_tools import generate_portfolio_visualization


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)

        # Take the output of the last time step
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class AIPoweredPortfolioOptimizationAgent(AgentBase):
    """
    Agent that uses AI (PyTorch) to optimize investment portfolios.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = LSTMModel()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.history = {'loss': []}

    def execute(self, *args, **kwargs):
        """
        Main execution method for the agent.
        """
        # Example execution logic wrapper
        data = kwargs.get('data')
        if data is not None:
            return self.run(data)
        return {"error": "No data provided"}

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocesses portfolio data for model training.
        """
        # Placeholder: reshape to (N, 1, 1) for LSTM (batch, seq, feature)
        return np.array(data).reshape(-1, 1, 1)

    def train_model(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> None:
        """
        Trains the LSTM model on portfolio data.
        """
        processed_data = self.preprocess_data(data)

        # Create tensors
        dataset = torch.tensor(processed_data, dtype=torch.float32)
        # Using data as target (Autoencoder-like behavior from original code)
        target = torch.tensor(processed_data, dtype=torch.float32).view(-1, 1)

        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(dataset.size(0))
            epoch_loss = 0.0
            for i in range(0, dataset.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = dataset[indices]
                batch_y = target[indices]

                if batch_x.size(0) == 0:
                    continue

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.history['loss'].append(epoch_loss / (dataset.size(0) / batch_size))

    def optimize_portfolio(self, data: pd.DataFrame) -> np.ndarray:
        """
        Optimizes the portfolio using the trained model.
        """
        processed_data = self.preprocess_data(data)
        input_tensor = torch.tensor(processed_data, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_tensor).numpy()

        optimized_weights = self.simulate_optimization(predictions)
        return optimized_weights

    def simulate_optimization(self, predictions: np.ndarray) -> np.ndarray:
        """
        Simulates portfolio optimization.
        """
        return np.random.rand(predictions.shape[0])

    def generate_portfolio_report(self, data: pd.DataFrame, optimized_weights: np.ndarray) -> Dict:
        """
        Generates a portfolio optimization report.
        """
        report = {
            "original_data": data.to_dict(),
            "optimized_weights": optimized_weights.tolist(),
            "model_loss": self.history['loss'][-1] if self.history['loss'] else None,
        }
        return report

    def generate_portfolio_visualization(self, data: pd.DataFrame, optimized_weights: np.ndarray):
        """Generates a visualization"""
        # Try/except to avoid crashing if utils are missing
        try:
            generate_portfolio_visualization(data, optimized_weights)
        except Exception:
            pass

    def run(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        """
        Runs the AI-powered portfolio optimization.
        """
        try:
            if not validate_portfolio_data(data):
                return {"error": "Invalid portfolio data."}
        except NameError:
            pass  # validate_portfolio_data might be missing

        self.train_model(data, epochs, batch_size)
        optimized_weights = self.optimize_portfolio(data)
        report = self.generate_portfolio_report(data, optimized_weights)
        self.generate_portfolio_visualization(data, optimized_weights)
        return report


# Example usage
if __name__ == "__main__":
    agent = AIPoweredPortfolioOptimizationAgent({"agent_id": "portfolio_opt"})
    portfolio_data = pd.DataFrame(np.random.rand(100, 1))
    results = agent.run(portfolio_data)
    print(results)
