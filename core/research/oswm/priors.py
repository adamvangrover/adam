import torch
import torch.nn as nn
import numpy as np

class NNPrior(nn.Module):
    """
    Generates synthetic data using a randomly initialized Neural Network.
    This captures complex, non-linear dependencies.
    """
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, num_layers=3):
        super(NNPrior, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization to prevent numerical instability."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def generate_trajectory(self, steps=50, start_val=0.0):
        """Generates a trajectory by iteratively applying the network."""
        trajectory = []
        current = torch.tensor([[start_val]], dtype=torch.float32)

        with torch.no_grad():
            for _ in range(steps):
                # Apply clip to prevent explosion
                current = torch.clamp(current, -1e6, 1e6)
                trajectory.append(current.item())

                # Input to net is current state
                delta = self.net(current)
                # We can model this as x_t+1 = x_t + f(x_t) (ResNet style) or x_t+1 = f(x_t)
                # Using additive update for smoother trajectories often found in nature
                current = current + delta

        return trajectory


class MomentumPrior:
    """
    Generates synthetic data using simple physics equations (position + velocity).
    """
    def __init__(self, dt=0.1, friction=0.01, gravity=0.0):
        self.dt = dt
        self.friction = friction
        self.gravity = gravity

    def generate_trajectory(self, steps=50, start_pos=0.0, start_vel=None):
        if start_vel is None:
            start_vel = np.random.randn()

        pos = start_pos
        vel = start_vel
        trajectory = []

        for _ in range(steps):
            trajectory.append(pos)

            # Physics update
            # v = v - friction*v + gravity + random_noise
            acc = -self.friction * vel + self.gravity + np.random.normal(0, 0.1)
            vel = vel + acc * self.dt
            pos = pos + vel * self.dt

            # Clip
            pos = max(-1e6, min(1e6, pos))

        return trajectory

class OrnsteinUhlenbeckPrior:
    """
    Generates synthetic data using an Ornstein-Uhlenbeck process.
    This simulates mean-reverting financial time series (e.g., volatility, interest rates).
    Equation: dx = theta * (mu - x) * dt + sigma * dW
    """
    def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=0.01):
        self.theta = theta # Speed of reversion
        self.mu = mu       # Long-term mean
        self.sigma = sigma # Volatility parameter
        self.dt = dt

    def generate_trajectory(self, steps=50, start_val=None):
        if start_val is None:
            # Start near the mean with some variance
            start_val = self.mu + np.random.normal(0, 1.0)

        x = start_val
        trajectory = []

        for _ in range(steps):
            trajectory.append(x)

            # OU Update
            dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.normal(0, np.sqrt(self.dt))
            x = x + dx

            # Clip
            x = max(-1e6, min(1e6, x))

        return trajectory

class PriorSampler:
    """
    Samples trajectories from a mixture of priors.
    """
    @staticmethod
    def sample(batch_size=1, steps=50, mixing_ratio=0.5):
        trajectories = []
        for _ in range(batch_size):
            rand = np.random.rand()
            if rand < 0.4:
                # Use NNPrior (40%)
                prior = NNPrior(input_dim=1, output_dim=1)
                traj = prior.generate_trajectory(steps=steps, start_val=np.random.randn())
            elif rand < 0.7:
                # Use MomentumPrior (30%)
                prior = MomentumPrior()
                traj = prior.generate_trajectory(steps=steps, start_pos=np.random.randn())
            else:
                # Use OrnsteinUhlenbeckPrior (30%)
                prior = OrnsteinUhlenbeckPrior(theta=0.15 + np.random.rand()*0.5, mu=np.random.randn())
                traj = prior.generate_trajectory(steps=steps, start_val=None)

            trajectories.append(traj)

        # (Batch, Seq, 1)
        tensor_data = torch.tensor(trajectories, dtype=torch.float32).unsqueeze(-1)
        # Convert to (Seq, Batch, 1) for Transformer default
        return tensor_data.permute(1, 0, 2)
