import pytest
import torch
import torch.nn as nn
from core.research.gnn.model import GAT, GraphSAGE
from core.research.federated_learning.fl_client import FinGraphFLClient
from core.research.federated_learning.privacy import MSGuard
from core.research.oswm.priors import NeuralNetworkPrior, MomentumPrior

def test_gat_forward():
    # Test GAT model forward pass
    model = GAT(nfeat=10, nhid=8, nclass=1, nheads=2)
    x = torch.randn(5, 10)
    adj = torch.eye(5)
    out = model(x, adj)
    assert out.shape == (5, 1)
    assert out.min() >= 0 and out.max() <= 1

def test_graphsage_forward():
    model = GraphSAGE(nfeat=10, nhid=8, nclass=1)
    x = torch.randn(5, 10)
    adj = torch.eye(5)
    out = model(x, adj)
    assert out.shape == (5, 1)

def test_fingraphfl_client():
    client = FinGraphFLClient(client_id="test", input_dim=5, data_size=20)
    loss = client.train_epoch(epochs=1)
    assert loss > 0
    weights = client.get_weights()
    # Check if weights keys exist
    # GAT structure: attentions.0.W, etc.
    keys = list(weights.keys())
    # GAT uses add_module('attention_{}') so keys are attention_0...
    assert any("attention_" in k for k in keys)

def test_msguard():
    # Create 3 updates: 2 similar, 1 malicious (flipped sign)
    upd1 = {"w": torch.tensor([1.0, 1.0])}
    upd2 = {"w": torch.tensor([1.1, 0.9])}
    upd3 = {"w": torch.tensor([-1.0, -1.0])} # Malicious

    updates = [upd1, upd2, upd3]
    valid = MSGuard.filter_updates(updates)

    # MSGuard logic implemented filters negative cosine similarity to mean.
    # Mean is approx positive.
    # Sim(upd3, mean) should be negative.

    # Ideally upd3 is filtered.
    assert len(valid) == 2
    # Check by identity to avoid tensor ambiguity error
    assert any(u is upd1 for u in valid)
    assert any(u is upd2 for u in valid)
    assert not any(u is upd3 for u in valid)

def test_priors():
    nn_prior = NeuralNetworkPrior(input_dim=1, output_dim=1)
    data = nn_prior.sample_batch(batch_size=2, seq_len=10)
    # Expected: (Seq, Batch, Feat)
    assert data.shape == (10, 2, 1)

    mom_prior = MomentumPrior(dim=1)
    data2 = mom_prior.sample_batch(batch_size=2, seq_len=10)
    assert data2.shape == (10, 2, 1)
