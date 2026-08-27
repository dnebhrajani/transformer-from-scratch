import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .norm import LayerNorm


class LocalEncoder(nn.Module):
    """
    BLT Local Encoder: compresses a sequence of raw bytes into patch representations.

    Takes byte embeddings and groups them into non-overlapping patches of size `patch_size`.
    A small causal transformer processes each patch, and the final hidden state of each
    patch becomes the patch representation fed to the global transformer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        patch_size: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Byte embedding: 256 possible byte values + PAD (0)
        self.byte_embedding = nn.Embedding(257, d_model, padding_idx=0)

        # Positional embedding within a patch (learned, since patches are short)
        self.pos_embedding = nn.Embedding(patch_size, d_model)

        # Small transformer layers for local context within patches
        self.layers = nn.ModuleList([
            LocalEncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(d_model)

    def forward(
        self, byte_ids: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            byte_ids: (batch, seq_len) — raw byte values (1-256, 0=PAD)
            padding_mask: (batch, seq_len) — True where padded
        Returns:
            patches: (batch, num_patches, d_model)
            patch_padding_mask: (batch, num_patches) — True where entire patch is padding
        """
        batch_size, seq_len = byte_ids.shape

        # Pad sequence length to be divisible by patch_size
        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            byte_ids = F.pad(byte_ids, (0, pad_len), value=0)
            if padding_mask is not None:
                padding_mask = F.pad(padding_mask, (0, pad_len), value=True)

        padded_len = byte_ids.size(1)
        num_patches = padded_len // self.patch_size

        # Embed bytes
        x = self.byte_embedding(byte_ids)  # (batch, padded_len, d_model)

        # Add within-patch positional embeddings
        positions = torch.arange(self.patch_size, device=byte_ids.device)
        pos_emb = self.pos_embedding(positions)  # (patch_size, d_model)
        # Tile across all patches
        pos_emb = pos_emb.unsqueeze(0).repeat(1, num_patches, 1)  # (1, padded_len, d_model)
        x = x + pos_emb

        # Reshape into patches: (batch * num_patches, patch_size, d_model)
        x = x.view(batch_size * num_patches, self.patch_size, self.d_model)

        # Build causal mask for local attention within each patch
        causal_mask = torch.triu(
            torch.ones(self.patch_size, self.patch_size, device=x.device, dtype=torch.bool),
            diagonal=1,
        )  # (patch_size, patch_size), True = masked
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, ps, ps)

        # Local padding mask per patch
        local_pad_mask = None
        if padding_mask is not None:
            # (batch, num_patches, patch_size)
            local_pad_mask = padding_mask.view(batch_size, num_patches, self.patch_size)
            # (batch * num_patches, patch_size)
            local_pad_mask = local_pad_mask.view(batch_size * num_patches, self.patch_size)
            # Expand for attention: (batch*num_patches, 1, 1, patch_size)
            pad_attn_mask = local_pad_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = causal_mask | pad_attn_mask
        else:
            combined_mask = causal_mask

        for layer in self.layers:
            x = layer(x, combined_mask)

        x = self.norm(x)

        # Take last non-padded position as patch representation
        # For simplicity, take the last position in the patch
        patches = x[:, -1, :]  # (batch * num_patches, d_model)
        patches = patches.view(batch_size, num_patches, self.d_model)

        # Patch-level padding mask: a patch is "padded" if ALL its bytes are padding
        if padding_mask is not None:
            patch_padding_mask = padding_mask.view(batch_size, num_patches, self.patch_size)
            patch_padding_mask = patch_padding_mask.all(dim=-1)  # (batch, num_patches)
        else:
            patch_padding_mask = torch.zeros(
                batch_size, num_patches, dtype=torch.bool, device=byte_ids.device
            )

        return patches, patch_padding_mask


class LocalEncoderLayer(nn.Module):
    """Single transformer layer for the local encoder."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = residual + self.dropout(x)

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class LocalDecoder(nn.Module):
    """
    BLT Local Decoder: expands patch representations back into byte-level predictions.

    Given a patch vector, autoregressively generates `patch_size` byte logits.
    Uses a small causal transformer that takes the patch representation as a "prefix"
    and generates bytes one at a time (during training, teacher-forced).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        patch_size: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Byte embedding for teacher-forced decoding
        self.byte_embedding = nn.Embedding(257, d_model, padding_idx=0)

        # Positional embedding within patch
        self.pos_embedding = nn.Embedding(patch_size, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            LocalDecoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(d_model)

        # Project patch representation to serve as the "start" token for each patch
        self.patch_proj = nn.Linear(d_model, d_model)

        # Output projection to byte logits (256 classes)
        self.output_proj = nn.Linear(d_model, 256)

    def forward(
        self,
        patch_representations: torch.Tensor,
        target_bytes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_representations: (batch, num_patches, d_model) from global decoder
            target_bytes: (batch, num_patches * patch_size) — teacher forcing targets
                          Byte values 1-256 (0 = PAD). If None, uses autoregressive generation.
        Returns:
            logits: (batch, num_patches * patch_size, 256)
        """
        batch_size, num_patches, _ = patch_representations.shape
        device = patch_representations.device

        # Project patch reps to serve as context
        patch_context = self.patch_proj(patch_representations)  # (batch, num_patches, d_model)

        if target_bytes is not None:
            # Teacher forcing: shift targets right within each patch
            # Reshape targets into patches
            target_patches = target_bytes.view(batch_size, num_patches, self.patch_size)

            # Create input sequence for each patch: [patch_context, byte_0, ..., byte_{n-2}]
            # The patch context acts as the "start" signal
            byte_emb = self.byte_embedding(target_patches)  # (batch, num_patches, patch_size, d_model)

            # Prepend patch context, drop last byte (shifted right)
            # patch_context: (batch, num_patches, 1, d_model)
            ctx = patch_context.unsqueeze(2)
            # Input is [ctx, byte_0, ..., byte_{patch_size-2}]
            decoder_input = torch.cat([ctx, byte_emb[:, :, :-1, :]], dim=2)
            # (batch, num_patches, patch_size, d_model)

            # Add positional embeddings
            positions = torch.arange(self.patch_size, device=device)
            decoder_input = decoder_input + self.pos_embedding(positions)

            # Reshape for processing: (batch * num_patches, patch_size, d_model)
            decoder_input = decoder_input.view(
                batch_size * num_patches, self.patch_size, self.d_model
            )

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(self.patch_size, self.patch_size, device=device, dtype=torch.bool),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

            x = decoder_input
            for layer in self.layers:
                x = layer(x, causal_mask)

            x = self.norm(x)
            logits = self.output_proj(x)  # (batch * num_patches, patch_size, 256)
            logits = logits.view(batch_size, num_patches * self.patch_size, 256)

        else:
            # Autoregressive generation (used at inference)
            all_logits = []
            for p in range(num_patches):
                ctx = patch_context[:, p : p + 1, :]  # (batch, 1, d_model)
                generated = []

                for t in range(self.patch_size):
                    # Build full sequence: [ctx, prev_byte_0, ..., prev_byte_{t-1}]
                    if t == 0:
                        full_seq = ctx
                    else:
                        byte_tokens = torch.stack(generated, dim=1)  # (batch, t)
                        byte_embs = self.byte_embedding(byte_tokens)  # (batch, t, d_model)
                        full_seq = torch.cat([ctx, byte_embs], dim=1)  # (batch, t+1, d_model)

                    # Add positional embedding
                    pos_ids = torch.arange(full_seq.size(1), device=device)
                    full_seq = full_seq + self.pos_embedding(pos_ids)

                    # Causal mask for current length
                    cur_len = full_seq.size(1)
                    causal_mask = torch.triu(
                        torch.ones(cur_len, cur_len, device=device, dtype=torch.bool),
                        diagonal=1,
                    ).unsqueeze(0).unsqueeze(0)

                    x = full_seq
                    for layer in self.layers:
                        x = layer(x, causal_mask)
                    x = self.norm(x)

                    # Take last position logits
                    step_logits = self.output_proj(x[:, -1, :])  # (batch, 256)
                    all_logits.append(step_logits)

                    # Greedy decode next byte (1-indexed for embedding)
                    next_byte = step_logits.argmax(dim=-1) + 1
                    generated.append(next_byte)

            logits = torch.stack(all_logits, dim=1)  # (batch, num_patches * patch_size, 256)

        return logits


class LocalDecoderLayer(nn.Module):
    """Single transformer layer for the local decoder."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x
