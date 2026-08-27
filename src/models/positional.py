import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).
    Added to token embeddings before feeding into the transformer.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model) for broadcasting over batch
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — token embeddings
        Returns:
            (batch, seq_len, d_model) — embeddings + positional encoding
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) — Su et al. 2021.
    Applied to Q and K *after* linear projections, *before* the dot product.

    This module precomputes the cos/sin tables; the actual rotation is applied
    via the `apply_rotary_pos_emb` function called inside attention.
    """

    def __init__(self, d_k: int, max_len: int = 5000):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires even head dimension"

        # Frequency bases: theta_i = 1 / 10000^(2i/d)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Precompute for max_len positions
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        positions = torch.arange(max_len, dtype=torch.float32)
        # (max_len, d_k/2)
        freqs = torch.outer(positions, self.inv_freq)
        # (max_len, d_k) — interleaved cos and sin
        cos_cached = freqs.cos()
        sin_cached = freqs.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin tables for the given sequence length.
        Both have shape (seq_len, d_k/2).
        """
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to a single tensor (Q or K independently)."""
    # cos, sin: (seq_len, d_k/2) -> (1, 1, seq_len, d_k/2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1, x2 = x[..., ::2], x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cos: torch.Tensor | None = None,
    k_sin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to Q and K tensors.

    Args:
        q: (batch, num_heads, seq_q, d_k)
        k: (batch, num_heads, seq_k, d_k)
        cos: (seq_q, d_k/2) — positions for Q
        sin: (seq_q, d_k/2) — positions for Q
        k_cos: (seq_k, d_k/2) — positions for K (if different from Q). If None, uses cos.
        k_sin: (seq_k, d_k/2) — positions for K (if different from Q). If None, uses sin.
    Returns:
        rotated q, k with same shapes
    """
    q_rot = _rotate_half(q, cos, sin)

    if k_cos is not None and k_sin is not None:
        k_rot = _rotate_half(k, k_cos, k_sin)
    else:
        k_rot = _rotate_half(k, cos, sin)

    return q_rot, k_rot
