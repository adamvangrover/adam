import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from core.data_sources.data_fetcher import DataFetcher
from .model import WorldModelTransformer
from .priors import NeuralNetworkPrior, MomentumPrior

class OSWMInference:
    """
    One-Shot World Model Inference Engine. 
    
    Uses Prior-Fitted Networks (PFNs) trained on synthetic priors to adapt to real contexts.
    Combines high-capacity architecture (Feature Branch) with robust meta-learning priors (Main Branch).
    """
    def __init__(self):
        # Configuration:
        # Using expanded capacity from Feature Branch (512 dims) for complex regime modeling,
        # but retaining Dropout and Prior structures from Main Branch for generalization.
        self.model = WorldModelTransformer(
            input_dim=1, 
            d_model=512, 
            nhead=8, 
            num_layers=4, 
            dropout=0.1
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Real-world Data Source (Feature Branch)
        self.data_fetcher = DataFetcher()

        # Synthetic Priors for Meta-Training (Main Branch)
        self.nn_prior = NeuralNetworkPrior(input_dim=1, output_dim=1)
        self.momentum_prior = MomentumPrior(dim=1)

    def pretrain_on_synthetic_prior(self, steps=100, batch_size=8, seq_len=50):
        """
        Trains the PFN on synthetic priors (NN + Momentum).
        This teaches the model to 'learn how to learn' dynamics by alternating 
        between different data generating processes.
        """
        self.model.train()
        print(f"Pre-training OSWM on Synthetic Priors for {steps} steps...")

        for i in range(steps):
            self.optimizer.zero_grad()

            # Mix priors: Alternate between NN and Momentum (Main Branch Logic)
            # This ensures the model doesn't overfit to one specific type of physics.
            if i % 2 == 0:
                data = self.nn_prior.sample_batch(batch_size, seq_len)
            else:
                data = self.momentum_prior.sample_batch(batch_size, seq_len)

            # data shape: (Seq, Batch, Feat)
            input_seq = data[:-1]
            target_seq = data[1:]

            output = self.model(input_seq)
            loss = self.criterion(output, target_seq)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if (i + 1) % 20 == 0:
                print(f"  Step {i+1}/{steps}: Loss={loss.item():.4f}")

    def load_market_context(self, ticker="SPY", period="1mo"):
        """
        Fetches historical data for a ticker and normalizes it for the model.
        (Capability from Feature Branch)
        
        Returns: normalized_data (list), scaler_params (dict)
        """
        print(f"Fetching market context for {ticker}...")
        history = self.data_fetcher.fetch_historical_data(ticker, period=period)

        if not history:
            print(f"Warning: No data found for {ticker}, using random context.")
            return np.random.randn(20).tolist(), {"mean": 0, "std": 1}

        prices = [d['close'] for d in history if d['close'] is not None]

        # Simple Z-score normalization to match synthetic prior distribution
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
        # Prepare context: (Seq, Batch=1, Feat=1)
        context_tensor = torch.tensor(context_data, dtype=torch.float32).view(-1, 1, 1)

        current_seq = context_tensor
        predictions = []

        with torch.no_grad():
            for _ in range(steps):
                # We feed the whole sequence. PFN attends to context to infer the underlying rule.
                output = self.model(current_seq)
                next_val = output[-1, :, :] # Take the last prediction
                predictions.append(next_val.item())

                # Append to sequence for autoregressive generation
                current_seq = torch.cat((current_seq, next_val.unsqueeze(0)), dim=0)

        return predictions