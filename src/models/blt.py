import math
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
            # --- NEW PADDING FIX ---
            # Ensure the sequence length is a perfect multiple of patch_size
            remainder = target_bytes.size(1) % self.patch_size
            if remainder != 0:
                pad_len = self.patch_size - remainder
                target_bytes = F.pad(target_bytes, (0, pad_len), value=0)
                num_patches = target_bytes.size(1) // self.patch_size

            target_patches = target_bytes.view(batch_size, num_patches, self.patch_size)
            # -----------------------
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


# ============================================================================
# Dynamic (entropy-based) components for C5
# ============================================================================


class EntropyPatcher:
    """Deterministic entropy-based byte sequence patcher for BLT (C5).

    Segments a byte sequence into variable-length patches based on local
    Shannon entropy.  High-entropy regions produce shorter patches;
    low-entropy regions produce longer patches.
    """

    def __init__(
        self,
        min_patch_size: int = 2,
        max_patch_size: int = 8,
        entropy_threshold: float = 3.0,
        entropy_window: int = 8,
    ):
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.entropy_threshold = entropy_threshold
        self.entropy_window = entropy_window

    @staticmethod
    def _compute_entropy(values: list[int]) -> float:
        """Unigram Shannon entropy  H = -\u03a3 p(c) log2 p(c)."""
        n = len(values)
        if n == 0:
            return 0.0
        counts: dict[int, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        H = 0.0
        for c in counts.values():
            p = c / n
            H -= p * math.log2(p)
        return H

    # def patch_sequence(self, byte_ids: list[int]) -> list[list[int]]:
    #     """Segment a sequence of embedding IDs into variable-length patches.

    #     Args:
    #         byte_ids: list of byte embedding IDs in [1, 256].
    #     Returns:
    #         List of patches, each a list of embedding IDs.
    #     """
    #     if not byte_ids:
    #         return []

    #     patches: list[list[int]] = []
    #     i = 0
    #     n = len(byte_ids)

    #     while i < n:
    #         patch_start = i
    #         # Always include at least min_patch_size bytes (or remainder)
    #         patch_end = min(i + self.min_patch_size, n)

    #         # After min_patch_size, check entropy to decide whether to close
    #         while (
    #             patch_end < n
    #             and (patch_end - patch_start) < self.max_patch_size
    #         ):
    #             w_start = max(patch_start, patch_end - self.entropy_window)
    #             window = byte_ids[w_start:patch_end]
    #             actual = [bid - 1 for bid in window]  # embedding ID -> byte value
    #             H = self._compute_entropy(actual)
    #             if H >= self.entropy_threshold:
    #                 break
    #             patch_end += 1

    #         patches.append(byte_ids[patch_start:patch_end])
    #         i = patch_end

    #     return patches

    def patch_sequence(self, byte_ids: list[int]) -> list[list[int]]:
        """Segment bytes into variable-length entropy-based patches."""
        if not byte_ids:
            return []

        patches: list[list[int]] = []
        i = 0
        n = len(byte_ids)

        while i < n:
            patch_start = i
            remaining = n - i

            # Final remainder can be shorter than min_patch_size.
            if remaining <= self.min_patch_size:
                patches.append(byte_ids[i:])
                break

            # Look at the local region ahead to estimate its entropy.
            window_end = min(i + self.entropy_window, n)
            actual = [bid - 1 for bid in byte_ids[i:window_end]]
            H = self._compute_entropy(actual)

            if H >= self.entropy_threshold:
                # High entropy -> short patch.
                patch_end = min(i + self.min_patch_size, n)
            else:
                # Low entropy -> grow until entropy becomes high
                # or the maximum patch size is reached.
                patch_end = min(i + self.min_patch_size, n)

                while patch_end < n and (patch_end - i) < self.max_patch_size:
                    window_end = min(i + self.entropy_window, n)
                    actual = [bid - 1 for bid in byte_ids[i:window_end]]
                    H = self._compute_entropy(actual)

                    if H >= self.entropy_threshold:
                        break

                    patch_end += 1

            patches.append(byte_ids[i:patch_end])
            i = patch_end

        return patches

    def batch_patch(
        self,
        byte_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Entropy-patch a batch of sequences.

        Args:
            byte_ids: (B, S) embedding IDs (1-256, 0=PAD)
            lengths:  (B,)   true sequence lengths
        Returns:
            patch_ids:     (B, P_max, M)  padded patch byte IDs
            patch_lengths: (B, P_max)     per-patch true lengths
            patch_mask:    (B, P_max)     True = nonexistent patch
        """
        B = byte_ids.size(0)
        M = self.max_patch_size
        device = byte_ids.device

        all_patches: list[list[list[int]]] = []
        max_num_patches = 0

        for b in range(B):
            seq = byte_ids[b, : lengths[b].item()].tolist()
            patches = self.patch_sequence(seq)
            all_patches.append(patches)
            max_num_patches = max(max_num_patches, len(patches))

        max_num_patches = max(max_num_patches, 1)  # at least 1 slot

        patch_ids = torch.zeros(B, max_num_patches, M, dtype=torch.long, device=device)
        patch_lengths = torch.zeros(B, max_num_patches, dtype=torch.long, device=device)
        patch_mask = torch.ones(B, max_num_patches, dtype=torch.bool, device=device)

        for b in range(B):
            for p, patch in enumerate(all_patches[b]):
                L = len(patch)
                patch_ids[b, p, :L] = torch.tensor(patch, dtype=torch.long, device=device)
                patch_lengths[b, p] = L
                patch_mask[b, p] = False  # this patch exists

        return patch_ids, patch_lengths, patch_mask


class DynamicLocalEncoder(nn.Module):
    """BLT local encoder for variable-length entropy-based patches.

    Processes pre-patched byte sequences through a small causal transformer
    and pools the last *valid* hidden state of each patch.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_patch_size: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_patch_size = max_patch_size
        self.d_model = d_model

        self.byte_embedding = nn.Embedding(257, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_patch_size, d_model)

        self.layers = nn.ModuleList([
            LocalEncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        patch_ids: torch.Tensor,
        patch_lengths: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_ids:     (B, P, M)  byte embedding IDs per patch (0=PAD)
            patch_lengths: (B, P)     true length of each patch
            patch_mask:    (B, P)     True = nonexistent patch
        Returns:
            patch_reps: (B, P, D)  one representation per patch
            patch_mask: (B, P)     unchanged
        """
        B, P, M = patch_ids.shape
        device = patch_ids.device

        x = self.byte_embedding(patch_ids)  # (B, P, M, D)
        positions = torch.arange(M, device=device)
        x = x + self.pos_embedding(positions)  # broadcast over B, P

        x = x.view(B * P, M, self.d_model)

        # Attention mask: causal + within-patch padding
        causal = torch.triu(
            torch.ones(M, M, device=device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, M, M)

        flat_lengths = patch_lengths.view(-1)  # (B*P,)
        pos_idx = torch.arange(M, device=device).unsqueeze(0)  # (1, M)
        local_pad = pos_idx >= flat_lengths.unsqueeze(1)  # (B*P, M)
        pad_attn = local_pad.unsqueeze(1).unsqueeze(2)  # (B*P, 1, 1, M)

        mask = causal | pad_attn

        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        # Pool last valid position per patch
        gather_idx = (flat_lengths - 1).clamp(min=0)  # (B*P,)
        gather_idx = gather_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.d_model)
        pooled = x.gather(1, gather_idx).squeeze(1)  # (B*P, D)
        pooled = pooled.view(B, P, self.d_model)

        return pooled, patch_mask


class DynamicLocalDecoder(nn.Module):
    """BLT local decoder for variable-length patches with EOP prediction.

    Expands patch representations back into byte-level predictions.
    Produces both byte logits (256 classes) and end-of-patch logits.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_patch_size: int = 8,
        min_patch_size: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size
        self.d_model = d_model

        self.byte_embedding = nn.Embedding(257, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_patch_size, d_model)

        self.layers = nn.ModuleList([
            LocalDecoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)

        self.patch_proj = nn.Linear(d_model, d_model)
        self.byte_proj = nn.Linear(d_model, 256)
        self.eop_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        patch_reps: torch.Tensor,
        target_patch_ids: torch.Tensor,
        target_patch_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced training.

        Args:
            patch_reps:           (B, P, D)  from global decoder
            target_patch_ids:     (B, P, M)  target byte IDs (1-256, 0=PAD)
            target_patch_lengths: (B, P)     true lengths per target patch
        Returns:
            byte_logits: (B, P, M, 256)
            eop_logits:  (B, P, M)
        """
        B, P, _ = patch_reps.shape
        M = self.max_patch_size
        device = patch_reps.device

        ctx = self.patch_proj(patch_reps).unsqueeze(2)  # (B, P, 1, D)
        byte_emb = self.byte_embedding(target_patch_ids)  # (B, P, M, D)

        # Shift right: [ctx, byte_0, ..., byte_{M-2}]
        decoder_input = torch.cat([ctx, byte_emb[:, :, :-1, :]], dim=2)
        positions = torch.arange(M, device=device)
        decoder_input = decoder_input + self.pos_embedding(positions)

        decoder_input = decoder_input.view(B * P, M, self.d_model)

        # Causal + padding mask
        causal = torch.triu(
            torch.ones(M, M, device=device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        flat_lengths = target_patch_lengths.view(-1)
        pos_idx = torch.arange(M, device=device).unsqueeze(0)
        local_pad = pos_idx >= flat_lengths.unsqueeze(1)
        pad_attn = local_pad.unsqueeze(1).unsqueeze(2)
        mask = causal | pad_attn

        x = decoder_input
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        byte_logits = self.byte_proj(x).view(B, P, M, 256)
        eop_logits = self.eop_proj(x).squeeze(-1).view(B, P, M)

        return byte_logits, eop_logits

    @torch.no_grad()
    def generate_patch(
        self, patch_rep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate one variable-length patch.

        Args:
            patch_rep: (B, 1, D)
        Returns:
            generated_ids:     (B, gen_len) embedding IDs in [1, 256]
            generated_lengths: (B,)         actual byte counts
        """
        B = patch_rep.size(0)
        device = patch_rep.device
        M = self.max_patch_size

        ctx = self.patch_proj(patch_rep)  # (B, 1, D)

        generated: list[torch.Tensor] = []
        generated_lengths = torch.full((B,), M, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(M):
            # Build input sequence
            if t == 0:
                full_seq = ctx  # (B, 1, D)
            else:
                byte_tokens = torch.stack(generated, dim=1)  # (B, t)
                byte_embs = self.byte_embedding(byte_tokens)  # (B, t, D)
                full_seq = torch.cat([ctx, byte_embs], dim=1)  # (B, t+1, D)

            cur_len = full_seq.size(1)
            pos_ids = torch.arange(cur_len, device=device)
            full_seq = full_seq + self.pos_embedding(pos_ids)

            causal = torch.triu(
                torch.ones(cur_len, cur_len, device=device, dtype=torch.bool),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

            x = full_seq
            for layer in self.layers:
                x = layer(x, causal)
            x = self.norm(x)

            step_byte_logits = self.byte_proj(x[:, -1, :])  # (B, 256)
            step_eop_logits = self.eop_proj(x[:, -1, :]).squeeze(-1)  # (B,)

            # Greedy: class 0..255 -> embedding ID 1..256
            next_byte = step_byte_logits.argmax(dim=-1) + 1  # (B,)
            next_byte[finished] = 0  # PAD for finished items
            generated.append(next_byte)

            # EOP decision
            eop_prob = torch.sigmoid(step_eop_logits)
            should_stop = eop_prob > 0.5
            if t + 1 < self.min_patch_size:
                should_stop.fill_(False)

            newly_finished = should_stop & ~finished
            generated_lengths[newly_finished] = t + 1
            finished = finished | should_stop

            if finished.all():
                break

        if not generated:
            return (
                torch.zeros(B, 1, dtype=torch.long, device=device),
                torch.ones(B, dtype=torch.long, device=device),
            )

        # Clamp lengths to actual number of generated steps
        generated_lengths = generated_lengths.clamp(max=len(generated))
        return torch.stack(generated, dim=1), generated_lengths
