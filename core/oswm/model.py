import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OSWMConfig:
    def __init__(self,
                 state_dim=7,
                 action_dim=2,
                 reward_dim=1,
                 d_model=512,
                 nhead=4,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.0,
                 max_seq_len=2048): # 1000 context steps * 2 tokens/step
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation=F.gelu, final_activation=None):
        super().__init__()
        self.activation = activation
        self.final_activation = final_activation
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.final_activation:
            x = self.final_activation(x)
        return x

class EncoderX(nn.Module):
    """
    Encodes [s, a].
    "Encoder type Cat... Separately encode action and state"
    """
    def __init__(self, config):
        super().__init__()
        # Split d_model between state and action encodings
        # Let's say half-half or proportional?
        # Table 7 just says "Encoder width 512".
        # I'll assume we produce two vectors that concat to 512.
        self.d_s = config.d_model // 2
        self.d_a = config.d_model - self.d_s

        # Encoder depth 3 (Table 7)
        self.state_enc = MLP(config.state_dim, self.d_s, 512, 3, activation=F.gelu)
        self.action_enc = MLP(config.action_dim, self.d_a, 512, 3, activation=F.gelu)

        # Project to d_model if needed (here sum is exactly d_model)

    def forward(self, state, action):
        e_s = self.state_enc(state)
        e_a = self.action_enc(action)
        return torch.cat([e_s, e_a], dim=-1)

class EncoderY(nn.Module):
    """
    Encodes [s', r].
    """
    def __init__(self, config):
        super().__init__()
        self.d_s = config.d_model // 2
        self.d_r = config.d_model - self.d_s

        self.state_enc = MLP(config.state_dim, self.d_s, 512, 3, activation=F.gelu)
        self.reward_enc = MLP(config.reward_dim, self.d_r, 512, 3, activation=F.gelu)

    def forward(self, next_state, reward):
        e_s = self.state_enc(next_state)
        e_r = self.reward_enc(reward)
        return torch.cat([e_s, e_r], dim=-1)

class DecoderY(nn.Module):
    """
    Decodes latent -> [s', r].
    "Separately decode next state and reward"
    Decoder depth 2 (Table 7).
    Decoder activation sigmoid.
    """
    def __init__(self, config):
        super().__init__()
        # Input is d_model.
        # We can share the latent or split it.
        # "Separately decode" usually means two separate heads on the same latent.
        # Table 7: Decoder width 64. Using d_model (512) for stability as 64 caused NaNs.
        decoder_width = config.d_model

        self.state_dec = MLP(config.d_model, config.state_dim, decoder_width, 2, activation=torch.sigmoid)
        self.reward_dec = MLP(config.d_model, config.reward_dim, decoder_width, 2, activation=torch.sigmoid)

    def forward(self, latent):
        s_next = self.state_dec(latent)
        r_next = self.reward_dec(latent)
        return s_next, r_next

class OSWMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder_x = EncoderX(config)
        self.encoder_y = EncoderY(config)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True # Usually better
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.decoder_y = DecoderY(config)

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model))

    def forward(self, state, action, target_state=None, target_reward=None):
        """
        Args:
            state: [B, T, state_dim]
            action: [B, T, action_dim]
            target_state: [B, T, state_dim] (Optional, for context)
            target_reward: [B, T, reward_dim] (Optional, for context)

        Returns:
            pred_next_state: [B, T, state_dim]
            pred_reward: [B, T, reward_dim]
        """
        B, T, _ = state.shape

        # Encode inputs (X)
        emb_x = self.encoder_x(state, action) # [B, T, D]

        if target_state is not None:
            # Training or Context mode
            # Encode targets (Y)
            # We shift targets for context: Y_t should be available to predict Y_{t+1}?
            # No, standard In-Context:
            # Seq: X1, Y1, X2, Y2, ...
            # To predict Y1, we see X1.
            # To predict Y2, we see X1, Y1, X2.

            # Embed Y
            emb_y = self.encoder_y(target_state, target_reward) # [B, T, D]

            # Interleave
            # Stack [B, T, 2, D] -> Flatten to [B, 2T, D]
            seq = torch.stack([emb_x, emb_y], dim=2).view(B, 2*T, self.config.d_model)

            # Add positional embeddings
            seq = seq + self.pos_embedding[:, :2*T, :]

            # Causal Mask
            # Allow attending to current X and all past.
            # For X_t (idx 2t): attend to 0..2t.
            # For Y_t (idx 2t+1): attend to 0..2t+1.
            mask = nn.Transformer.generate_square_subsequent_mask(2*T).to(seq.device)

            out = self.transformer(seq, mask=mask, is_causal=True)

            # Extract outputs corresponding to X tokens (to predict Y)
            # X_t is at index 2t (0, 2, 4...)
            # Output at X_t predicts Y_t.
            out_x = out[:, 0::2, :] # [B, T, D]

            pred_s, pred_r = self.decoder_y(out_x)

            return pred_s, pred_r

        else:
            # Inference mode step-by-step?
            # Actually, `target_state` being None implies we only have X.
            # But the model needs context Ys to work.
            # Usually we pass the full context sequence.
            # If target_state is None, maybe we just predict based on X?
            # But we need history Ys.
            # Let's assume this forward handles the "Batch Training" case.
            # For inference, we'll likely pass the context Ys but maybe the last Y is missing?
            # If T inputs X, and T-1 inputs Y.
            # We can pad Y.
            pass

        return None, None