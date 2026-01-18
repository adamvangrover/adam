import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .transformer import WorldModelTransformer
from .priors import NeuralNetworkPrior, MomentumPrior

class OSWMInference:
    """
    One-Shot World Model Inference Engine.
    Uses Prior-Fitted Networks (PFNs) trained on synthetic priors to adapt to real contexts.
    """
    def __init__(self):
        # Increased capacity for PFN behavior
        self.model = WorldModelTransformer(input_dim=1, d_model=64, nhead=4, num_layers=4, dropout=0.1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Synthetic Priors
        self.nn_prior = NeuralNetworkPrior(input_dim=1, output_dim=1)
        self.momentum_prior = MomentumPrior(dim=1)

    def pretrain_on_synthetic_prior(self, steps=100, batch_size=8, seq_len=50):
        """
        Trains the PFN on synthetic priors (NN + Momentum).
        This teaches the model to 'learn how to learn' dynamics.
        """
        self.model.train()
        print(f"Pre-training OSWM on Synthetic Priors for {steps} steps...")

        for i in range(steps):
            self.optimizer.zero_grad()

            # Mix priors: Alternate between NN and Momentum
            if i % 2 == 0:
                data = self.nn_prior.sample_batch(batch_size, seq_len)
            else:
                data = self.momentum_prior.sample_batch(batch_size, seq_len)

            # data: (Seq, Batch, Feat)
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

    def generate_scenario(self, context_data, steps=20):
        """
        Uses in-context learning to predict future steps.
        context_data: List of floats (historical prices)
        """
        self.model.eval()
        # Prepare context: (Seq, Batch=1, Feat=1)
        context_tensor = torch.tensor(context_data, dtype=torch.float32).view(-1, 1, 1)

        predictions = []
        current_seq = context_tensor

        with torch.no_grad():
            for _ in range(steps):
                # We feed the whole sequence. PFN attends to context.
                output = self.model(current_seq)
                next_val = output[-1, :, :] # Take the last prediction
                predictions.append(next_val.item())

                # Append to sequence for autoregressive generation
                current_seq = torch.cat((current_seq, next_val.unsqueeze(0)), dim=0)

        return predictions
