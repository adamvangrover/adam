import torch
from core.research.oswm.priors import NoisyWavePrior

def test_noisy_wave_prior_bounds():
    prior = NoisyWavePrior(dim=1)

    batch_size = 10
    seq_len = 100

    # Generate data
    data = prior.sample_batch(batch_size, seq_len)

    # Check shape: (Seq, Batch, Feature)
    assert data.shape == (seq_len, batch_size, 1), f"Expected shape {(seq_len, batch_size, 1)}, got {data.shape}"

    # Check bounds: All values must be strictly between 0 and 1
    assert torch.all(data >= 0.0), f"Found values < 0.0: min is {data.min()}"
    assert torch.all(data <= 1.0), f"Found values > 1.0: max is {data.max()}"

    print("NoisyWavePrior test passed!")
