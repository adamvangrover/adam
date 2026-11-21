import torch
from typing import Optional, Tuple

class KVCache:
    """
    Optimized Key-Value Cache for Auto-Regressive Inference.

    Features:
    - Pre-allocation of GPU memory (avoid malloc overhead).
    - Rolling buffer for efficient sequence extension.
    - Support for 'Rolling Back' state (crucial for Speculative Decoding).
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, dtype=torch.float16, device="cuda"):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.current_seq_len = 0
        self.device = device

        # Pre-allocate memory
        # Shape: (batch, n_heads, seq_len, head_dim)
        # Using CPU for prototype compatibility, in real lab use 'device'
        device = "cpu" if not torch.cuda.is_available() else device

        self.k_cache = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device=device)
        self.v_cache = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device=device)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int):
        """
        Updates the cache with new tokens.
        k_new, v_new: (batch, n_heads, new_tokens, head_dim)
        """
        batch_size, n_heads, n_new, head_dim = k_new.shape

        # Boundary check
        if start_pos + n_new > self.max_seq_len:
            raise ValueError("Cache overflow")

        self.k_cache[:, :, start_pos : start_pos + n_new, :] = k_new
        self.v_cache[:, :, start_pos : start_pos + n_new, :] = v_new

        self.current_seq_len = start_pos + n_new

    def get_view(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a view of the active cache up to seq_len.
        """
        return (
            self.k_cache[:, :, :seq_len, :],
            self.v_cache[:, :, :seq_len, :]
        )

    def rollback(self, length: int):
        """
        Rolls back the sequence length pointer.
        Crucial for Speculative Decoding when draft tokens are rejected.
        Does not zero out memory (lazy deletion) for speed.
        """
        if length > self.current_seq_len:
             raise ValueError("Cannot rollback forward")
        self.current_seq_len = length

if __name__ == "__main__":
    print("Initializing KVCache...")
    cache = KVCache(1, 1024, 8, 64)
    print(f"Cache initialized. Shape: {cache.k_cache.shape}")
