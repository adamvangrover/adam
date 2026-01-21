import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class RandomMLP(nn.Module):
    """
    A single random MLP for one dimension of the state in one environment.
    """
    def __init__(self, input_dim, hidden_dim=16, num_layers=3, activation_opts=['relu', 'tanh', 'sigmoid', 'sin']):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = []

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.activations.append(random.choice(activation_opts))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.activations.append(random.choice(activation_opts))

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, 1))

        # Random initialization
        for layer in self.layers:
            # Use Xavier initialization to prevent explosion
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Scale and offset for final output (as per text: "initialized with a random scale and offset")
        self.out_scale = np.random.uniform(0.1, 5.0) # Arbitrary range based on typical needs
        self.out_offset = np.random.uniform(-5.0, 5.0)

    def forward(self, x):
        # x: [input_dim]
        h_prev = None
        h = x

        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            act_name = self.activations[i]
            if act_name == 'relu':
                h = F.relu(h)
            elif act_name == 'tanh':
                h = torch.tanh(h)
            elif act_name == 'sigmoid':
                h = torch.sigmoid(h)
            elif act_name == 'sin':
                h = torch.sin(h)

            # Residual connection "aggregates the outputs of the first and second layers"
            # Text says: "residual connection that aggregates the outputs of the first and second layers"
            # Assuming this means adding the output of layer 1 to layer 2 input or output.
            # Given standard residual connections, let's implement a simple residual if shapes match
            # But here shapes match (hidden_dim).
            # If we are at layer 1 (index 1), we can add layer 0 output.
            if i == 1 and h_prev is not None:
                h = h + h_prev

            h_prev = h

        out = self.layers[-1](h)
        return out * self.out_scale + self.out_offset

class BatchedNNPrior:
    """
    Manages NNs for a batch of environments.
    """
    def __init__(self, batch_size, output_dim, input_state_dim, action_dim, hidden_dim=16, num_layers=3):
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.input_state_dim = input_state_dim
        self.action_dim = action_dim

        # Input to each NN is full previous state + action
        self.input_dim = input_state_dim + action_dim

        # Store models: List of List of RandomMLP [batch_size][output_dim]
        self.models = []
        for b in range(batch_size):
            env_models = []
            for s in range(output_dim):
                env_models.append(RandomMLP(self.input_dim, hidden_dim, num_layers))
            self.models.append(env_models)

    def step(self, state, action):
        # state: [B, input_state_dim]
        # action: [B, action_dim]

        next_states = []

        for b in range(self.batch_size):
            # Concatenate state and action
            inp = torch.cat([state[b], action[b]], dim=0) # [input_dim]

            ns = []
            for i in range(self.output_dim):
                out = self.models[b][i](inp)
                ns.append(out)

            next_states.append(torch.cat(ns))

        return torch.stack(next_states) # [B, output_dim]

class BatchedMomentumPrior:
    def __init__(self, batch_size, mom_dim, action_dim):
        self.batch_size = batch_size
        self.mom_dim = mom_dim # Number of position/velocity pairs? Or just dimension of space.
        # Text says: "initial position sampled from [0, 2pi], initial velocity [-3, 3]"
        # "angular dynamics... represented internally in radians"

        # Let's assume mom_dim is the number of independent physical bodies/joints
        # Each has pos and vel.
        # So state vector size for this prior is 2 * mom_dim.

        self.pos = torch.rand(batch_size, mom_dim) * 2 * np.pi
        self.vel = torch.rand(batch_size, mom_dim) * 6 - 3

        # Random gravity per environment? Or constant?
        # "Angular dynamics can incorporate gravity"
        self.gravity = torch.rand(batch_size, mom_dim) * 10 # Random gravity 0-10
        self.dt = 0.05 # simulation step

    def get_state(self):
        # Output can be sine, cosine, or radian values.
        # Let's return concatenated [pos, vel]. Or [sin(pos), cos(pos), vel].
        # Paper: "output can be sine, cosine, or radian values."
        # Let's stick to simple [pos, vel] for now or maybe [sin(p), cos(p), v].
        # For simplicity and given "compressed spatial and temporal representation",
        # let's return [sin(pos), cos(pos), vel] to handle periodicity nicely.

        s = torch.sin(self.pos)
        c = torch.cos(self.pos)
        v = self.vel
        return torch.cat([s, c, v], dim=1)

    def step(self, action):
        # action: [B, action_dim]
        # Update vel: v = v + a * dt - g * dt
        # We need to map action to momentum dimension.
        # If action_dim != mom_dim, we project or slice.
        # Let's assume action_dim >= mom_dim and we use first mom_dim actions.
        # Or we use a random projection.

        # Simple case: Assume action influences velocity directly.
        # We'll take first `mom_dim` actions.
        act = action[:, :self.pos.shape[1]]
        if act.shape[1] < self.pos.shape[1]:
            # Pad with zeros if not enough actions
            pad = torch.zeros(self.batch_size, self.pos.shape[1] - act.shape[1])
            act = torch.cat([act, pad], dim=1)

        # Update
        # Use torch.sin instead of np.sin to avoid mixing types and warnings
        self.vel = self.vel + act * self.dt - self.gravity * torch.sin(self.pos) * self.dt # Gravity acts on pendulum
        # Wait, gravity on linear is constant, on pendulum is sin(pos).
        # Text says "Angular dynamics can incorporate gravity".
        # Let's do pendulum dynamics as they are more interesting.

        self.pos = self.pos + self.vel * self.dt

        return self.get_state()

class BatchedRewardPrior:
    """
    Generates rewards based on state transitions and actions.
    """
    def __init__(self, batch_size, state_dim, action_dim):
        self.batch_size = batch_size
        # Reward function is also an MLP taking (s_next, a, s)
        self.input_dim = state_dim * 2 + action_dim
        self.models = []
        for b in range(batch_size):
            self.models.append(RandomMLP(self.input_dim, hidden_dim=16, num_layers=3))

        # "Reward is replaced by a constant reward of 1 with a probability of 0.5"
        self.is_constant = (torch.rand(batch_size) < 0.5)

    def step(self, state, action, next_state):
        rewards = []
        for b in range(self.batch_size):
            if self.is_constant[b]:
                rewards.append(torch.tensor([1.0]))
            else:
                inp = torch.cat([next_state[b], action[b], state[b]], dim=0)
                r = self.models[b](inp)
                rewards.append(r)
        return torch.stack(rewards)

class PriorSampler:
    def __init__(self, batch_size=8, seq_len=1001,
                 nn_state_dim=4, mom_bodies=1, action_dim=2):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.nn_state_dim = nn_state_dim
        self.mom_bodies = mom_bodies
        self.action_dim = action_dim

        # State dim calculation
        # Momentum state: [sin, cos, vel] per body -> 3 * mom_bodies
        self.mom_state_dim = 3 * mom_bodies
        self.total_state_dim = nn_state_dim + self.mom_state_dim

    def sample_batch(self):
        # Initialize priors
        nn_prior = BatchedNNPrior(self.batch_size, self.nn_state_dim, self.total_state_dim, self.action_dim)
        mom_prior = BatchedMomentumPrior(self.batch_size, self.mom_bodies, self.action_dim)
        reward_prior = BatchedRewardPrior(self.batch_size, self.total_state_dim, self.action_dim)

        # Initial states
        # NN prior init: "sampled from U(0,1), then scaled and offset"
        # The RandomMLP doesn't hold state, it's a transition function.
        # But we need an initial state vector s_0.
        s_nn = torch.rand(self.batch_size, self.nn_state_dim) * 2 - 1 # Simple random init
        s_mom = mom_prior.get_state()

        state = torch.cat([s_nn, s_mom], dim=1)

        # Storage
        states = [state]
        actions = []
        rewards = []
        next_states = []

        # Generate sequence
        for t in range(self.seq_len):
            # Sample random action
            action = torch.rand(self.batch_size, self.action_dim) * 2 - 1 # [-1, 1]
            actions.append(action)

            # Step Momentum
            s_mom_next = mom_prior.step(action)

            # Step NN
            # Input to NN prior is "entire previous state" (which includes momentum state)
            s_nn_next = nn_prior.step(state, action)

            # Combine
            state_next = torch.cat([s_nn_next, s_mom_next], dim=1)

            # Clip to prevent explosion (which causes NaNs in normalization)
            state_next = torch.clamp(state_next, -1e6, 1e6)

            next_states.append(state_next)

            # Reward
            reward = reward_prior.step(state, action, state_next)
            reward = torch.clamp(reward, -1e6, 1e6)
            rewards.append(reward)

            state = state_next
            states.append(state) # Keep for next step

        # Stack
        # X: [s_t, a_t]
        # Y: [s_{t+1}, r_{t+1}]

        X_list = []
        Y_list = []

        for t in range(self.seq_len):
            s_t = states[t]
            a_t = actions[t]
            s_next = states[t+1]
            r_next = rewards[t]

            x = torch.cat([s_t, a_t], dim=1)
            y = torch.cat([s_next, r_next], dim=1)

            X_list.append(x)
            Y_list.append(y)

        X = torch.stack(X_list, dim=1) # [B, T, dim_x]
        Y = torch.stack(Y_list, dim=1) # [B, T, dim_y]

        return X, Y
