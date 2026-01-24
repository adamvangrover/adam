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
        Generates a batch of trajectories from random dynamical systems using vectorized operations.
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

        # Stack: (Seq, Batch, Feature)
        data = torch.cat(trajectory, dim=1) # (B, Seq, F)
        data = data.permute(1, 0, 2) # (Seq, Batch, F)
        return data

class MomentumPrior(Prior):
    """
    Generates data based on simple physical laws (Harmonic Oscillator). 

[Image of harmonic oscillator physics model]

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
            
            # Simple clamping
            pos = torch.clamp(pos, -1e6, 1e6)

        data = torch.cat(trajectory, dim=1) # (B, Seq, F)
        data = data.permute(1, 0, 2) # (Seq, Batch, F)
        return data

class OrnsteinUhlenbeckPrior(Prior):
    """
    Generates synthetic data using an Ornstein-Uhlenbeck process. 
    Simulates mean-reverting financial time series (e.g., volatility, interest rates).
    Equation: dx = theta * (mu - x) * dt + sigma * dW
    """
    def __init__(self, dim=1, dt=0.01):
        self.dim = dim
        self.dt = dt

    def sample_batch(self, batch_size, seq_len):
        # Random parameters per batch
        # Theta (reversion speed): Uniform(0.1, 0.65)
        theta = torch.rand(batch_size, 1, 1) * 0.55 + 0.1
        # Mu (long term mean): Normal(0, 1)
        mu = torch.randn(batch_size, 1, self.dim)
        # Sigma (volatility): Fixed 0.2 approx
        sigma = 0.2

        # Start near the mean
        x = mu + torch.randn(batch_size, 1, self.dim)
        
        trajectory = []
        sqrt_dt = np.sqrt(self.dt)

        for t in range(seq_len):
            trajectory.append(x)
            
            # Vectorized OU Update
            # dW is standard normal noise
            dW = torch.randn(batch_size, 1, self.dim)
            dx = theta * (mu - x) * self.dt + sigma * dW * sqrt_dt
            x = x + dx
            
            x = torch.clamp(x, -1e6, 1e6)

        data = torch.cat(trajectory, dim=1)
        data = data.permute(1, 0, 2)
        return data

class PriorSampler:
    """
    Samples trajectories from a mixture of priors using efficient batching.
    """
    def __init__(self):
        self.nn_prior = NeuralNetworkPrior()
        self.momentum_prior = MomentumPrior()
        self.ou_prior = OrnsteinUhlenbeckPrior()

    def sample(self, batch_size=1, steps=50):
        """
        Samples a mixed batch of trajectories.
        Distribution: 40% NN, 30% Momentum, 30% OU.
        """
        rand = np.random.rand()
        
        if rand < 0.4:
            return self.nn_prior.sample_batch(batch_size, steps)
        elif rand < 0.7:
            return self.momentum_prior.sample_batch(batch_size, steps)
        else:
            return self.ou_prior.sample_batch(batch_size, steps)