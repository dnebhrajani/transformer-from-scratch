import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        query: (batch, heads, seq_q, d_k)
        key:   (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask:  broadcastable to (batch, heads, seq_q, seq_k)
               True/1 = position to MASK (ignore), consistent with PyTorch convention
        dropout: optional dropout applied to attention weights
    Returns:
        output: (batch, heads, seq_q, d_v)
        attn_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # if mask is not None:
    #     scores = scores.masked_fill(mask.bool(), float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention with independent Q, K, V projections per head."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  (batch, 1, seq_q, seq_k) or (batch, 1, 1, seq_k)
                   True = masked position
        Returns:
            (batch, seq_q, d_model)
        """
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape to (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # Concatenate heads: (batch, seq_q, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.W_o(attn_output)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention: multiple query heads share fewer KV heads.
    Reduces KV memory/compute while retaining query expressiveness.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)  # full Q projection
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  (batch, 1, seq_q, seq_k) or (batch, 1, 1, seq_k)
        Returns:
            (batch, seq_q, d_model)
        """
        batch_size, seq_q, _ = query.shape
        seq_k = key.size(1)

        Q = self.W_q(query)  # (batch, seq_q, d_model)
        K = self.W_k(key)  # (batch, seq_k, num_kv_heads * d_k)
        V = self.W_v(value)  # (batch, seq_k, num_kv_heads * d_k)

        # Reshape Q: (batch, num_heads, seq_q, d_k)
        Q = Q.view(batch_size, seq_q, self.num_heads, self.d_k).transpose(1, 2)

        # Reshape K, V: (batch, num_kv_heads, seq_k, d_k)
        K = K.view(batch_size, seq_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_k, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Expand KV heads to match Q heads by repeating
        # (batch, num_kv_heads, seq_k, d_k) -> (batch, num_heads, seq_k, d_k)
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # (batch, seq_q, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)
        )

        return self.W_o(attn_output)
