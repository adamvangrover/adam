import torch
import numpy as np

class OSWMInference:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

        self.context_X = []
        self.context_Y = []

        self.mean_X = None
        self.std_X = None
        self.mean_Y = None
        self.std_Y = None

        self.simulation_history_X = []
        self.simulation_history_Y = []

    def set_context(self, transitions):
        """
        transitions: list of tuples (state, action, next_state, reward)
        """
        states = []
        actions = []
        next_states = []
        rewards = []

        for s, a, ns, r in transitions:
            states.append(s)
            actions.append(a)
            next_states.append(ns)
            rewards.append(r)

        # Convert to tensor [1, T, D]
        # Assuming inputs are numpy arrays or lists
        s_t = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(0).to(self.device)
        a_t = torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(0).to(self.device)
        ns_t = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(0).to(self.device)
        r_t = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(0).to(self.device)

        self.context_X = torch.cat([s_t, a_t], dim=2)
        self.context_Y = torch.cat([ns_t, r_t], dim=2)

        # Compute Stats
        self.mean_X = self.context_X.mean(dim=1, keepdim=True)
        self.std_X = self.context_X.std(dim=1, keepdim=True) + 1e-6

        self.mean_Y = self.context_Y.mean(dim=1, keepdim=True)
        self.std_Y = self.context_Y.std(dim=1, keepdim=True) + 1e-6

        # Reset simulation history
        self.simulation_history_X = []
        self.simulation_history_Y = []

    def step(self, state, action):
        """
        Predict next state and reward given current state and action.
        """
        # Prepare input
        s_t = torch.tensor(np.array(state), dtype=torch.float32).view(1, 1, -1).to(self.device)
        a_t = torch.tensor(np.array(action), dtype=torch.float32).view(1, 1, -1).to(self.device)

        current_X = torch.cat([s_t, a_t], dim=2)

        # Add to simulation history
        if len(self.simulation_history_X) == 0:
             hist_X = current_X
        else:
             hist_X = torch.cat(self.simulation_history_X + [current_X], dim=1)

        # Combine Context + History
        full_X = torch.cat([self.context_X, hist_X], dim=1)

        # Context Y is used, plus any past simulation Ys
        if len(self.simulation_history_Y) > 0:
            hist_Y = torch.cat(self.simulation_history_Y, dim=1)
            full_Y = torch.cat([self.context_Y, hist_Y], dim=1)
        else:
            full_Y = self.context_Y

        # Normalize
        # Use context stats
        norm_X = (full_X - self.mean_X) / self.std_X
        norm_Y = (full_Y - self.mean_Y) / self.std_Y

        # Model Forward
        # We need to pad Y to match X length?
        # Model expects X and Target Y.
        # But we don't have Target Y for the current step (that's what we want to predict).
        # My Model implementation takes `target_state` and `target_reward` and interleaves.
        # If I pass `norm_Y` which is length T-1 (relative to X length T),
        # I should probably just pad it with zeros or reuse last value (doesn't matter as masked).
        # But wait, `OSWMTransformer` interleaves `emb_x` and `emb_y`.
        # `seq = stack([x, y])`. This assumes lengths match.

        # So I must provide a "dummy" Y for the current step.
        # It will be masked anyway (causal mask prevents X_t from seeing Y_t).
        # So I can append zeros to norm_Y.

        dummy_Y = torch.zeros_like(current_X[:, :, :self.context_Y.shape[2]])
        input_Y = torch.cat([norm_Y, dummy_Y], dim=1)

        # Split input_Y back to state/reward for model input
        state_dim = self.model.config.state_dim
        target_s = input_Y[:, :, :state_dim]
        target_r = input_Y[:, :, state_dim:]

        state_in = norm_X[:, :, :state_dim]
        action_in = norm_X[:, :, state_dim:]

        with torch.no_grad():
            pred_s, pred_r = self.model(state_in, action_in, target_s, target_r)

        # Get last prediction
        last_s = pred_s[:, -1:, :]
        last_r = pred_r[:, -1:, :]

        # Denormalize
        # Y stats
        # mean_Y is [1, 1, D]
        mean_s = self.mean_Y[:, :, :state_dim]
        std_s = self.std_Y[:, :, :state_dim]
        mean_r = self.mean_Y[:, :, state_dim:]
        std_r = self.std_Y[:, :, state_dim:]

        final_s = last_s * std_s + mean_s
        final_r = last_r * std_r + mean_r

        # Store prediction in history for next step
        # We store the *normalized* prediction? Or denormalized?
        # If we use the prediction as truth for next step context, we should use it.
        # But for "Context Y", we essentially assume the prediction is what happened.
        # Actually, if we are simulating, we don't observe the real Y.
        # So we must use the predicted Y as the "Target" for the next step's context?
        # Yes, autoregressive generation.
        # So I should store `last_s` and `last_r` (normalized) or re-normalize.
        # Since I computed `input_Y` with a dummy, now I have the real "dummy" replacement.

        # But wait, `full_Y` needs to be in the same space as `context_Y`.
        # `context_Y` is raw data.
        # So I should store `final_s` and `final_r` (raw).

        self.simulation_history_X.append(current_X)
        self.simulation_history_Y.append(torch.cat([final_s, final_r], dim=2))

        return final_s.squeeze().cpu().numpy(), final_r.squeeze().cpu().numpy()
