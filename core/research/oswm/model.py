import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class WorldModelTransformer(nn.Module):
    """
    A Transformer model that predicts the next state in a sequence.
    Context: Past N states.
    Output: Next state.
    """
    def __init__(self, input_dim=1, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super(WorldModelTransformer, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # Feedforward typically 4x d_model
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, input_dim)
        self.d_model = d_model

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """
        src: (SeqLen, Batch, Feature)
        """
        device = src.device
        seq_len = src.size(0)
        mask = self._generate_square_subsequent_mask(seq_len).to(device)

        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=mask)
        output = self.decoder(output)
        return output
