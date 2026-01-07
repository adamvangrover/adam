import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 5.3 Reinforcement Learning for Parameter Tuning
# Asynchronous PPO agent to tune Gamma and Kappa

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Actions are continuous (Gamma, Kappa)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

class StrategyTuner:
    def __init__(self):
        # State: Volatility, Inventory, Spread, Imbalance
        self.state_dim = 4
        # Action: Gamma, Kappa
        self.action_dim = 2

        self.policy = PPOPolicy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mean, _ = self.policy(state_tensor)
        # In PPO we would sample from a distribution, simplified here
        return action_mean.detach().numpy()[0]

    def update(self, rollouts):
        # PPO update logic stub
        pass

if __name__ == "__main__":
    tuner = StrategyTuner()
    # Mock state: Volatility=0.02, Inventory=10, Spread=0.01, Imbalance=0.5
    state = np.array([0.02, 10.0, 0.01, 0.5])
    action = tuner.select_action(state)
    print(f"Recommended Parameters -> Gamma: {action[0]:.4f}, Kappa: {action[1]:.4f}")
