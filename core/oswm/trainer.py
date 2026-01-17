import torch
import torch.nn as nn
import numpy as np
from .priors import PriorSampler
from .model import OSWMConfig, OSWMTransformer

class OSWMTrainer:
    def __init__(self, config: OSWMConfig = None, device='cpu'):
        self.config = config or OSWMConfig()
        self.device = device
        self.model = OSWMTransformer(self.config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.0)

        # Sampler config matching paper defaults
        self.sampler = PriorSampler(
            batch_size=8,
            seq_len=self.config.max_seq_len // 2, # Wait, seq_len in prior is steps. Model max_seq_len is tokens (2*steps)
            nn_state_dim=4,
            mom_bodies=1,
            action_dim=2
        )

    def normalize_batch(self, batch_data):
        """
        Normalize per sequence (batch element).
        batch_data: [B, T, D]
        Returns: normalized_data, mean, std
        """
        # Calculate mean/std per batch element over time dimension
        mean = batch_data.mean(dim=1, keepdim=True) # [B, 1, D]
        std = batch_data.std(dim=1, keepdim=True) + 1e-6 # [B, 1, D]

        norm_data = (batch_data - mean) / std
        return norm_data, mean, std

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Sample Batch
        # X: [s, a], Y: [s', r]
        raw_X, raw_Y = self.sampler.sample_batch()
        raw_X = raw_X.to(self.device)
        raw_Y = raw_Y.to(self.device)

        # 2. Normalize
        X, mean_X, std_X = self.normalize_batch(raw_X)
        Y, mean_Y, std_Y = self.normalize_batch(raw_Y)

        B, T, _ = X.shape

        # 3. Split into Context and Query (Implicitly via Masking in loss)
        # Algorithm 1: Sample eval from U(k, T-1).
        # k (min context) = 500 (Appendix C Table 5)
        # But our seq len is 1001.
        k = 500
        if T > k:
            eval_pos = np.random.randint(k, T)
        else:
            eval_pos = k // 2 # Fallback

        # 4. Forward Pass
        # We pass full sequences. The causal mask ensures we don't cheat.
        # s dim is part of X (first state_dim elems). a is rest.
        state_dim = self.config.state_dim

        # X is [s, a]
        state = X[:, :, :state_dim]
        action = X[:, :, state_dim:]

        # Y is [s', r]
        target_s = Y[:, :, :state_dim]
        target_r = Y[:, :, state_dim:]

        pred_s, pred_r = self.model(state, action, target_s, target_r)

        # 5. Loss Calculation
        # Only on positions > eval_pos
        # The prediction at index t (which saw X_t and Y_{t-1}) is predicting Y_t.
        # So pred_s[:, t] corresponds to target_s[:, t].

        # Slice from eval_pos+1 to end?
        # Text says: "predict future targets Y_{eval+1:T} based on remaining inputs X_{eval+1:T}"
        # This implies we predict starting from eval+1.

        loss_s = nn.functional.mse_loss(pred_s[:, eval_pos:, :], target_s[:, eval_pos:, :])
        loss_r = nn.functional.mse_loss(pred_r[:, eval_pos:, :], target_r[:, eval_pos:, :])

        loss = loss_s + loss_r

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), eval_pos
