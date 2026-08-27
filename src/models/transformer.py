import math
import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GroupedQueryAttention, scaled_dot_product_attention
from .positional import SinusoidalPositionalEncoding, RotaryPositionalEncoding, apply_rotary_pos_emb, _rotate_half
from .norm import LayerNorm, RMSNorm
from .blt import LocalEncoder, LocalDecoder


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-norm encoder layer: Norm -> Attention -> Add -> Norm -> FFN -> Add"""

    def __init__(self, d_model, num_heads, d_ff, dropout, norm_type="layernorm", attn_type="mha", num_kv_heads=None):
        super().__init__()
        self.norm1 = _make_norm(norm_type, d_model)
        self.norm2 = _make_norm(norm_type, d_model)

        if attn_type == "gqa":
            self.self_attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
        else:
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, src_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class DecoderLayer(nn.Module):
    """Pre-norm decoder layer: masked self-attn + cross-attn + FFN"""

    def __init__(self, d_model, num_heads, d_ff, dropout, norm_type="layernorm", attn_type="mha", num_kv_heads=None):
        super().__init__()
        self.norm1 = _make_norm(norm_type, d_model)
        self.norm2 = _make_norm(norm_type, d_model)
        self.norm3 = _make_norm(norm_type, d_model)

        if attn_type == "gqa":
            self.self_attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
            self.cross_attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
        else:
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = residual + self.dropout(x)

        # Cross-attention
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = residual + self.dropout(x)

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x

        return x


class RoPEEncoderLayer(nn.Module):
    """Encoder layer with RoPE applied inside attention."""

    def __init__(self, d_model, num_heads, d_ff, dropout, norm_type="layernorm"):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

        self.norm1 = _make_norm(norm_type, d_model)
        self.norm2 = _make_norm(norm_type, d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Pre-norm self-attention with RoPE
        residual = x
        x = self.norm1(x)

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to Q and K
        Q, K = apply_rotary_pos_emb(Q, K, rope_cos, rope_sin)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, src_mask, self.attn_dropout)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.W_o(attn_out)

        x = residual + self.dropout(attn_out)

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class RoPEDecoderLayer(nn.Module):
    """Decoder layer with RoPE applied inside self-attention and cross-attention."""

    def __init__(self, d_model, num_heads, d_ff, dropout, norm_type="layernorm"):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

        self.norm1 = _make_norm(norm_type, d_model)
        self.norm2 = _make_norm(norm_type, d_model)
        self.norm3 = _make_norm(norm_type, d_model)

        # Self-attention projections
        self.self_W_q = nn.Linear(d_model, d_model)
        self.self_W_k = nn.Linear(d_model, d_model)
        self.self_W_v = nn.Linear(d_model, d_model)
        self.self_W_o = nn.Linear(d_model, d_model)

        # Cross-attention projections
        self.cross_W_q = nn.Linear(d_model, d_model)
        self.cross_W_k = nn.Linear(d_model, d_model)
        self.cross_W_v = nn.Linear(d_model, d_model)
        self.cross_W_o = nn.Linear(d_model, d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_rope_cos: torch.Tensor,
        tgt_rope_sin: torch.Tensor,
        src_rope_cos: torch.Tensor,
        src_rope_sin: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        tgt_len = x.size(1)
        src_len = enc_output.size(1)

        # Masked self-attention with RoPE
        residual = x
        x_norm = self.norm1(x)

        Q = self.self_W_q(x_norm).view(batch_size, tgt_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.self_W_k(x_norm).view(batch_size, tgt_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.self_W_v(x_norm).view(batch_size, tgt_len, self.num_heads, self.d_k).transpose(1, 2)

        Q, K = apply_rotary_pos_emb(Q, K, tgt_rope_cos, tgt_rope_sin)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, tgt_mask, self.attn_dropout)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        x = residual + self.dropout(self.self_W_o(attn_out))

        # Cross-attention with RoPE (Q and K have different sequence lengths)
        residual = x
        x_norm = self.norm2(x)

        Q = self.cross_W_q(x_norm).view(batch_size, tgt_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.cross_W_k(enc_output).view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.cross_W_v(enc_output).view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)

        # RoPE on Q with target positions, K with source positions (different lengths)
        Q, K = apply_rotary_pos_emb(Q, K, tgt_rope_cos, tgt_rope_sin, src_rope_cos, src_rope_sin)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, memory_mask, self.attn_dropout)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        x = residual + self.dropout(self.cross_W_o(attn_out))

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerSeq2Seq(nn.Module):
    """
    Full Encoder-Decoder Transformer for configurations C1-C4.
    Configurable positional encoding, attention, and normalization.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        d_ff = config["d_ff"]
        dropout = config["dropout"]
        src_vocab_size = config["src_vocab_size"]
        tgt_vocab_size = config["tgt_vocab_size"]
        max_len = config.get("max_len", 1024)
        norm_type = config.get("norm_type", "layernorm")
        attn_type = config.get("attn_type", "mha")
        pos_type = config.get("pos_type", "sinusoidal")
        num_kv_heads = config.get("num_kv_heads", num_heads // 4)

        self.d_model = d_model
        self.pos_type = pos_type

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        if pos_type == "sinusoidal":
            self.src_pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
            self.tgt_pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
            self.rope = None
        elif pos_type == "rope":
            self.src_pos_enc = None
            self.tgt_pos_enc = None
            self.rope = RotaryPositionalEncoding(d_model // num_heads, max_len)
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

        # Encoder
        if pos_type == "rope":
            self.encoder_layers = nn.ModuleList([
                RoPEEncoderLayer(d_model, num_heads, d_ff, dropout, norm_type)
                for _ in range(num_layers)
            ])
        else:
            self.encoder_layers = nn.ModuleList([
                EncoderLayer(d_model, num_heads, d_ff, dropout, norm_type, attn_type, num_kv_heads)
                for _ in range(num_layers)
            ])

        # Decoder
        if pos_type == "rope":
            self.decoder_layers = nn.ModuleList([
                RoPEDecoderLayer(d_model, num_heads, d_ff, dropout, norm_type)
                for _ in range(num_layers)
            ])
        else:
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(d_model, num_heads, d_ff, dropout, norm_type, attn_type, num_kv_heads)
                for _ in range(num_layers)
            ])

        self.encoder_norm = _make_norm(norm_type, d_model)
        self.decoder_norm = _make_norm(norm_type, d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Weight initialization
        self._init_weights(num_layers)

    def _init_weights(self, num_layers: int):
        """Xavier init + residual branch scaling."""
        scale = 1.0 / math.sqrt(2.0 * num_layers)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # Scale down residual projections (output projections of attn and FFN)
            if "W_o" in name or "net.2" in name:
                with torch.no_grad():
                    p.mul_(scale)

    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)

        if self.pos_type == "sinusoidal":
            x = self.src_pos_enc(x)
            for layer in self.encoder_layers:
                x = layer(x, src_mask)
        elif self.pos_type == "rope":
            rope_cos, rope_sin = self.rope(src.size(1))
            for layer in self.encoder_layers:
                x = layer(x, rope_cos, rope_sin, src_mask)

        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        if self.pos_type == "sinusoidal":
            x = self.tgt_pos_enc(x)
            for layer in self.decoder_layers:
                x = layer(x, enc_output, tgt_mask, memory_mask)
        elif self.pos_type == "rope":
            tgt_cos, tgt_sin = self.rope(tgt.size(1))
            src_cos, src_sin = self.rope(enc_output.size(1))
            for layer in self.decoder_layers:
                x = layer(x, enc_output, tgt_cos, tgt_sin, src_cos, src_sin, tgt_mask, memory_mask)

        x = self.decoder_norm(x)
        return self.output_proj(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) — source token IDs
            tgt: (batch, tgt_len) — target token IDs (shifted right for teacher forcing)
            src_mask: (batch, 1, 1, src_len) — padding mask for encoder
            tgt_mask: (batch, 1, tgt_len, tgt_len) — causal + padding mask for decoder
            memory_mask: (batch, 1, 1, src_len) — padding mask for cross-attention
        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        enc_output = self.encode(src, src_mask)
        logits = self.decode(tgt, enc_output, tgt_mask, memory_mask)
        return logits


class BLTTransformerSeq2Seq(nn.Module):
    """
    Configuration C5: Byte Latent Transformer.
    No tokenizer — raw bytes processed through local encoder -> global transformer -> local decoder.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        d_ff = config["d_ff"]
        dropout = config["dropout"]
        patch_size = config.get("patch_size", 4)
        local_layers = config.get("local_layers", 1)
        local_heads = config.get("local_heads", 4)
        max_len = config.get("max_len", 1024)
        norm_type = config.get("norm_type", "layernorm")

        self.patch_size = patch_size

        # Local encoder: bytes -> patches (for source)
        self.src_local_encoder = LocalEncoder(
            d_model, local_heads, patch_size, local_layers, dropout
        )
        # Local encoder for target (during training for teacher forcing context)
        self.tgt_local_encoder = LocalEncoder(
            d_model, local_heads, patch_size, local_layers, dropout
        )

        # Global positional encoding (on patches)
        self.src_pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Global encoder (processes source patches)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, norm_type)
            for _ in range(num_layers)
        ])
        self.encoder_norm = _make_norm(norm_type, d_model)

        # Global decoder (cross-attends target patches to source patches)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, norm_type)
            for _ in range(num_layers)
        ])
        self.decoder_norm = _make_norm(norm_type, d_model)

        # Local decoder: patches -> bytes (for target)
        self.local_decoder = LocalDecoder(
            d_model, local_heads, patch_size, local_layers, dropout
        )

        self._init_weights(num_layers)

    def _init_weights(self, num_layers: int):
        scale = 1.0 / math.sqrt(2.0 * num_layers)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if "W_o" in name or "net.2" in name:
                with torch.no_grad():
                    p.mul_(scale)

    def forward(
        self,
        src_bytes: torch.Tensor,
        tgt_bytes: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src_bytes: (batch, src_byte_len) — source byte IDs (1-256, 0=PAD)
            tgt_bytes: (batch, tgt_byte_len) — target byte IDs for teacher forcing
            src_padding_mask: (batch, src_byte_len) — True where padded
            tgt_padding_mask: (batch, tgt_byte_len) — True where padded
        Returns:
            logits: (batch, tgt_byte_len, 256)
        """
        # Local encode source bytes into patches
        src_patches, src_patch_mask = self.src_local_encoder(src_bytes, src_padding_mask)

        # Local encode target bytes into patches (for global decoder input)
        tgt_patches, tgt_patch_mask = self.tgt_local_encoder(tgt_bytes, tgt_padding_mask)

        # Add positional encoding to patches
        src_patches = self.src_pos_enc(src_patches)
        tgt_patches = self.tgt_pos_enc(tgt_patches)

        # Build masks for global attention
        # src_patch_mask: (batch, num_src_patches) -> (batch, 1, 1, num_src_patches)
        src_attn_mask = src_patch_mask.unsqueeze(1).unsqueeze(2) if src_patch_mask is not None else None

        # Target causal + padding mask
        num_tgt_patches = tgt_patches.size(1)
        causal = torch.triu(
            torch.ones(num_tgt_patches, num_tgt_patches, device=tgt_patches.device, dtype=torch.bool),
            diagonal=1,
        )  # (T, T)
        tgt_attn_mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        if tgt_patch_mask is not None:
            pad_mask = tgt_patch_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            tgt_attn_mask = tgt_attn_mask | pad_mask

        # Global encoder
        enc_out = src_patches
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_attn_mask)
        enc_out = self.encoder_norm(enc_out)

        # Global decoder
        dec_out = tgt_patches
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_attn_mask, src_attn_mask)
        dec_out = self.decoder_norm(dec_out)

        # Local decoder: expand patches back to byte logits
        logits = self.local_decoder(dec_out, tgt_bytes)

        return logits

    def generate(
        self,
        src_bytes: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
        max_patches: int = 256,
    ) -> torch.Tensor:
        """
        Non-autoregressive generation for inference (greedy).

        Since source and target have equal byte counts in this task, we use
        the encoder output as decoder queries (non-autoregressive at patch level),
        then autoregressively decode bytes within each patch via the local decoder.
        """
        batch_size = src_bytes.size(0)
        device = src_bytes.device

        # Encode source -> source patches
        src_patches, src_patch_mask = self.src_local_encoder(src_bytes, src_padding_mask)
        src_patches = self.src_pos_enc(src_patches)
        num_patches = src_patches.size(1)

        src_attn_mask = None
        if src_patch_mask is not None:
            src_attn_mask = src_patch_mask.unsqueeze(1).unsqueeze(2)

        # Global encoder
        enc_out = src_patches
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_attn_mask)
        enc_out = self.encoder_norm(enc_out)

        # Global decoder: use learned positional queries as decoder input
        # (non-autoregressive — no causal mask, since we decode all patches at once)
        dec_queries = self.tgt_pos_enc(
            torch.zeros(batch_size, num_patches, self.config["d_model"], device=device)
        )

        # No causal mask for non-autoregressive decoding at patch level
        for layer in self.decoder_layers:
            dec_queries = layer(dec_queries, enc_out, tgt_mask=None, memory_mask=src_attn_mask)
        dec_out = self.decoder_norm(dec_queries)

        # Local decoder: expand patches to bytes autoregressively
        logits = self.local_decoder(dec_out, target_bytes=None)

        return logits


def _make_norm(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == "layernorm":
        return LayerNorm(d_model)
    elif norm_type == "rmsnorm":
        return RMSNorm(d_model)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def create_masks(
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_pad_idx: int = 0,
    tgt_pad_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create all necessary masks for the transformer.

    Args:
        src: (batch, src_len) source token IDs
        tgt: (batch, tgt_len) target token IDs
        src_pad_idx: padding token ID for source
        tgt_pad_idx: padding token ID for target
    Returns:
        src_mask: (batch, 1, 1, src_len) — True where source is PAD
        tgt_mask: (batch, 1, tgt_len, tgt_len) — causal + padding mask
        memory_mask: (batch, 1, 1, src_len) — same as src_mask, for cross-attention
    """
    batch_size, src_len = src.shape
    _, tgt_len = tgt.shape

    # Source padding mask: (batch, 1, 1, src_len)
    src_mask = (src == src_pad_idx).unsqueeze(1).unsqueeze(2)

    # Target: causal mask + padding mask
    # Causal: (1, 1, tgt_len, tgt_len)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)

    # Padding: (batch, 1, 1, tgt_len)
    tgt_pad_mask = (tgt == tgt_pad_idx).unsqueeze(1).unsqueeze(2)

    # Combined: (batch, 1, tgt_len, tgt_len) — OR of causal and padding
    tgt_mask = causal_mask | tgt_pad_mask

    # Memory mask (for cross-attention keys): same as src_mask
    memory_mask = src_mask

    return src_mask, tgt_mask, memory_mask
