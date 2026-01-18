import torch
import torch.nn as nn
import numpy as np

class Prior:
    """
    Base class for synthetic priors.
    """
    def sample_batch(self, batch_size, seq_len):
        raise NotImplementedError

class NeuralNetworkPrior(Prior):
    """
    Generates data from random neural networks (Random MLPs).
    Simulates complex, non-linear, but structured dynamics.
    """
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def sample_batch(self, batch_size, seq_len):
        """
        Generates a batch of trajectories from random dynamical systems.
        """
        # Random initial states: (Batch, 1, Input)
        current_state = torch.randn(batch_size, 1, self.input_dim)

        # Random Weights for the batch of networks
        # W1: (Batch, Input, Hidden)
        W1 = torch.randn(batch_size, self.input_dim, self.hidden_dim) / np.sqrt(self.input_dim)
        # W2: (Batch, Hidden, Output)
        W2 = torch.randn(batch_size, self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)

        trajectory = []

        for t in range(seq_len):
            trajectory.append(current_state)

            # Dynamics: x_{t+1} = x_t + 0.1 * Net(x_t)
            # Forward pass: h = tanh(x * W1)
            h = torch.bmm(current_state, W1) # (B, 1, H)
            h = torch.tanh(h)

            # out = h * W2
            update = torch.bmm(h, W2) # (B, 1, O)

            current_state = current_state + 0.1 * update

            # Clamp to prevent explosion
            current_state = torch.clamp(current_state, -10, 10)

        # Stack: (Seq, Batch, Feature) -> (Seq, Batch, Feature)
        # Trajectory list is list of (B, 1, F)
        # cat(dim=1) -> (B, Seq, F)
        # We need (Seq, Batch, F) for Transformer usually, but let's see consumer.
        data = torch.cat(trajectory, dim=1) # (B, Seq, F)
        data = data.permute(1, 0, 2) # (Seq, Batch, F)
        return data

class MomentumPrior(Prior):
    """
    Generates data based on simple physical laws (Harmonic Oscillator).
    P(t+1) = P(t) + V(t)*dt
    V(t+1) = V(t) - k*P(t)*dt
    """
    def __init__(self, dim=1):
        self.dim = dim

    def sample_batch(self, batch_size, seq_len):
        # Initial Position and Velocity
        pos = torch.randn(batch_size, 1, self.dim)
        vel = torch.randn(batch_size, 1, self.dim)

        # Random spring constants per batch: k ~ Uniform(0.1, 2.0)
        k = torch.rand(batch_size, 1, 1) * 1.9 + 0.1
        # Random friction/damping: c ~ Uniform(0, 0.1)
        c = torch.rand(batch_size, 1, 1) * 0.1

        trajectory = []
        dt = 0.1

        for t in range(seq_len):
            trajectory.append(pos)

            # Physics Update
            acc = -k * pos - c * vel
            vel = vel + acc * dt
            pos = pos + vel * dt

        data = torch.cat(trajectory, dim=1) # (B, Seq, F)
        data = data.permute(1, 0, 2) # (Seq, Batch, F)
        return data
