import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from core.data_sources.data_fetcher import DataFetcher
from .model import WorldModelTransformer
from .priors import PriorSampler

class OSWMInference:
    def __init__(self):
        # Using default d_model=512 from model.py
        self.model = WorldModelTransformer(input_dim=1, d_model=512, nhead=8, num_layers=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) # Reduced LR for larger model
        self.criterion = nn.MSELoss()
        self.data_fetcher = DataFetcher()

    def pretrain_on_synthetic_prior(self, steps=100, batch_size=4):
        """
        Trains the model on synthetic data (NNPrior + MomentumPrior) to learn general dynamics.
        """
        self.model.train()
        print(f"Pre-training OSWM on synthetic priors for {steps} steps...")

        for i in range(steps):
            # Sample batch of trajectories from priors
            # Shape: (Seq, Batch, Feat)
            y = PriorSampler.sample(batch_size=batch_size, steps=50, mixing_ratio=0.5)

            self.optimizer.zero_grad()

            # Predict next step (shifted)
            input_seq = y[:-1]
            target_seq = y[1:]

            output = self.model(input_seq)
            loss = self.criterion(output, target_seq)
            loss.backward()
            self.optimizer.step()

            if i % 10 == 0:
                print(f"  Step {i}/{steps}, Loss: {loss.item():.6f}")

    def load_market_context(self, ticker="SPY", period="1mo"):
        """
        Fetches historical data for a ticker and normalizes it for the model.
        Returns: normalized_data (list), scaler_params (dict)
        """
        print(f"Fetching market context for {ticker}...")
        history = self.data_fetcher.fetch_historical_data(ticker, period=period)

        if not history:
            print(f"Warning: No data found for {ticker}, using random context.")
            return np.random.randn(20).tolist(), {"mean": 0, "std": 1}

        prices = [d['close'] for d in history if d['close'] is not None]

        # Simple Z-score normalization
        mean = np.mean(prices)
        std = np.std(prices) + 1e-6
        normalized = [(p - mean) / std for p in prices]

        return normalized, {"mean": mean, "std": std}

    def generate_scenario(self, context_data, steps=20):
        """
        Uses in-context learning to predict future steps.
        context_data: List of floats (historical prices)
        """
        self.model.eval()
        # Prepare context
        # (Seq, Batch=1, Feat=1)
        context_tensor = torch.tensor(context_data, dtype=torch.float32).view(-1, 1, 1)

        predictions = []
        current_seq = context_tensor

        with torch.no_grad():
            for _ in range(steps):
                output = self.model(current_seq)
                next_val = output[-1, :, :] # Take the last prediction
                predictions.append(next_val.item())

                # Append to sequence for autoregressive generation
                current_seq = torch.cat((current_seq, next_val.unsqueeze(0)), dim=0)

        return predictions
