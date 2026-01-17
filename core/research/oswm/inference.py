import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .transformer import WorldModelTransformer

class OSWMInference:
    def __init__(self):
        self.model = WorldModelTransformer(input_dim=1, d_model=32, nhead=2, num_layers=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def pretrain_on_synthetic_prior(self, steps=100):
        """
        Trains the model on synthetic sine waves to learn basic physics/periodicity.
        """
        self.model.train()
        for _ in range(steps):
            # Generate random sine wave
            t = torch.linspace(0, 10, 50)
            freq = torch.rand(1) * 2 + 0.5
            y = torch.sin(freq * t).unsqueeze(1).unsqueeze(1) # (Seq, Batch, Feat)

            self.optimizer.zero_grad()
            # Predict next step (shifted)
            input_seq = y[:-1]
            target_seq = y[1:]

            output = self.model(input_seq)
            loss = self.criterion(output, target_seq)
            loss.backward()
            self.optimizer.step()

    def generate_scenario(self, context_data, steps=20):
        """
        Uses in-context learning to predict future steps.
        context_data: List of floats (historical prices)
        """
        self.model.eval()
        # Prepare context
        context_tensor = torch.tensor(context_data, dtype=torch.float32).view(-1, 1, 1)

        predictions = []
        current_seq = context_tensor

        with torch.no_grad():
            for _ in range(steps):
                output = self.model(current_seq)
                next_val = output[-1, :, :] # Take the last prediction
                predictions.append(next_val.item())

                # Append to sequence for autoregressive generation
                # Keep sequence length manageable if needed, but Transformer can handle it
                current_seq = torch.cat((current_seq, next_val.unsqueeze(0)), dim=0)

        return predictions
