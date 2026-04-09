import torch
import torch.nn as nn
import numpy as np

class Prior:
    """
    Base class for synthetic priors.
    """
    def sample_batch(self, batch_size, seq_len, device='cpu'):
        raise NotImplementedError

class NeuralNetworkPrior(Prior):
    """Generates data from random neural networks (Random MLPs)."""
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def generate_trajectory(self, steps, start_val=0.0):
        # Fallback for old tests
        res = self.sample_batch(1, steps)
        return res[:, 0, 0].tolist()

    def sample_batch(self, batch_size, seq_len, device='cpu'):
        current_state = torch.randn(batch_size, 1, self.input_dim, device=device)

        W1 = torch.randn(batch_size, self.input_dim, self.hidden_dim, device=device) / np.sqrt(self.input_dim)
        W2 = torch.randn(batch_size, self.hidden_dim, self.output_dim, device=device) / np.sqrt(self.hidden_dim)

        # Pre-allocate tensor for performance: (Seq, Batch, Feature)
        data = torch.empty((seq_len, batch_size, self.output_dim), device=device)

        for t in range(seq_len):
            data[t] = current_state.squeeze(1)

            h = torch.tanh(torch.bmm(current_state, W1))
            update = torch.bmm(h, W2)

            current_state = current_state + 0.1 * update
            current_state = torch.clamp(current_state, -10, 10)

        return data

class MomentumPrior(Prior):
    """
    Generates data based on simple physical laws (Harmonic Oscillator). 

[Image of harmonic oscillator physics model]

    P(t+1) = P(t) + V(t)*dt
    V(t+1) = V(t) - (k*P(t) + c*V(t))*dt
    """
    def __init__(self, dim=1):
        self.dim = dim

    def sample_batch(self, batch_size, seq_len, device='cpu'):
        pos = torch.randn(batch_size, 1, self.dim, device=device)
        vel = torch.randn(batch_size, 1, self.dim, device=device)

    def sample_batch(self, batch_size, seq_len, device='cpu'):
        pos = torch.randn(batch_size, 1, self.dim, device=device)
        vel = torch.randn(batch_size, 1, self.dim, device=device)

        k = torch.rand(batch_size, 1, 1, device=device) * 1.9 + 0.1
        c = torch.rand(batch_size, 1, 1, device=device) * 0.1

        dt = 0.1
        data = torch.empty((seq_len, batch_size, self.dim), device=device)

        for t in range(seq_len):
            data[t] = pos.squeeze(1)

            acc = -k * pos - c * vel
            vel = vel + acc * dt
            pos = pos + vel * dt
            pos = torch.clamp(pos, -1e6, 1e6)

        return data

class NoisyWavePrior(Prior):
    """
    Replaces the pseudoscientific 'WavelengthSpeedOfLightPrior'.
    Generates a realistic, frequency-modulated sine wave with dynamic noise,
    clamped between 0 and 1.
    """
    def __init__(self, dim=1):
        self.dim = dim

    def sample_batch(self, batch_size, seq_len, device='cpu'):
        # Random base frequencies and phase shifts
        frequency = torch.rand(batch_size, 1, self.dim, device=device) * 5.0 + 1.0
        phase = torch.rand(batch_size, 1, self.dim, device=device) * 2 * np.pi

        data = torch.empty((seq_len, batch_size, self.dim), device=device)

        for t in range(seq_len):
            time_t = t * 0.1

            # Base wave dynamics
            base_val = 0.5 * (torch.sin(frequency * time_t + phase) + 1.0)

            # Add stochastic measurement noise
            noise = torch.randn_like(base_val) * 0.05
            val = torch.clamp(base_val + noise, 0.0, 1.0)

            data[t] = val.squeeze(1)

        return data

class OrnsteinUhlenbeckPrior(Prior):
    """Simulates mean-reverting financial time series (e.g., volatility)."""
    def __init__(self, dim=1, dt=0.01):
        self.dim = dim
        self.dt = dt

    def generate_trajectory(self, steps, start_val=0.0):
        # Fallback for old tests
        res = self.sample_batch(1, steps)
        return res[:, 0, 0].tolist()

    def sample_batch(self, batch_size, seq_len, device='cpu'):
        theta = torch.rand(batch_size, 1, 1, device=device) * 0.55 + 0.1
        mu = torch.randn(batch_size, 1, self.dim, device=device)
        sigma = 0.2

        x = mu + torch.randn(batch_size, 1, self.dim, device=device)
        sqrt_dt = np.sqrt(self.dt)

        data = torch.empty((seq_len, batch_size, self.dim), device=device)

        for t in range(seq_len):
            data[t] = x.squeeze(1)
            
            dW = torch.randn(batch_size, 1, self.dim, device=device)
            dx = theta * (mu - x) * self.dt + sigma * dW * sqrt_dt
            x = torch.clamp(x + dx, -1e6, 1e6)

        return data

class NoisyWavePrior(Prior):
    """
    Replaces the pseudoscientific 'WavelengthSpeedOfLightPrior'.
    Generates a realistic, frequency-modulated sine wave with dynamic noise,
    clamped between 0 and 1.
    """
    def __init__(self, dim=1):
        self.dim = dim

    def generate_trajectory(self, steps, start_val=0.0):
        # Fallback for old tests
        res = self.sample_batch(1, steps)
        return res[:, 0, 0].tolist()

    def sample_batch(self, batch_size, seq_len, device='cpu'):
        # Random base frequencies and phase shifts
        frequency = torch.rand(batch_size, 1, self.dim, device=device) * 5.0 + 1.0
        phase = torch.rand(batch_size, 1, self.dim, device=device) * 2 * np.pi

        data = torch.empty((seq_len, batch_size, self.dim), device=device)

        for t in range(seq_len):
            time_t = t * 0.1

            # Base wave dynamics
            base_val = 0.5 * (torch.sin(frequency * time_t + phase) + 1.0)

            # Add stochastic measurement noise
            noise = torch.randn_like(base_val) * 0.05
            val = torch.clamp(base_val + noise, 0.0, 1.0)

            data[t] = val.squeeze(1)

        return data

class PriorSampler:
    """Samples trajectories from a mixture of priors using efficient batching."""
    def __init__(self):
        self.nn_prior = NeuralNetworkPrior()
        self.momentum_prior = MomentumPrior()
        self.ou_prior = OrnsteinUhlenbeckPrior()
        self.wave_prior = NoisyWavePrior()

    @classmethod
    def sample(cls, batch_size=1, steps=50, device='cpu'):
        return cls().sample_mixed(batch_size, steps, device)

    def sample_mixed(self, batch_size=1, steps=50, device='cpu'):
        """
        Samples a truly mixed batch of trajectories.
        Distribution: 30% NN, 30% Momentum, 30% OU, 10% Wave.
        """
        probabilities = torch.tensor([0.3, 0.3, 0.3, 0.1], device=device)

        # Determine exact number of samples for each prior in this batch
        multinomial = torch.distributions.Multinomial(total_count=batch_size, probs=probabilities)
        counts = multinomial.sample().int().tolist()

        batches = []
        if counts[0] > 0:
            batches.append(self.nn_prior.sample_batch(counts[0], steps, device))
        if counts[1] > 0:
            batches.append(self.momentum_prior.sample_batch(counts[1], steps, device))
        if counts[2] > 0:
            batches.append(self.ou_prior.sample_batch(counts[2], steps, device))
        if counts[3] > 0:
            batches.append(self.wave_prior.sample_batch(counts[3], steps, device))

        # Concatenate along the batch dimension (dim=1)
        mixed_batch = torch.cat(batches, dim=1)
        
        # Shuffle the batch to ensure heterogenous distribution during training
        indices = torch.randperm(batch_size, device=device)
        return mixed_batch[:, indices, :]
