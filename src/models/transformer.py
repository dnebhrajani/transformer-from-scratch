import math
import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GroupedQueryAttention, scaled_dot_product_attention
from .positional import SinusoidalPositionalEncoding, RotaryPositionalEncoding, apply_rotary_pos_emb
from .norm import LayerNorm, RMSNorm
from .blt import LocalEncoder, LocalDecoder, EntropyPatcher, DynamicLocalEncoder, DynamicLocalDecoder


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
    Configuration C5: Byte Latent Transformer with entropy-based dynamic patching.

    Raw bytes -> entropy patcher -> local encoder -> global transformer -> local decoder -> bytes.
    Patches have variable length in [min_patch_size, max_patch_size].
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        d_ff = config["d_ff"]
        dropout = config["dropout"]
        max_len = config.get("max_len", 1024)
        norm_type = config.get("norm_type", "layernorm")
        local_layers = config.get("local_layers", 1)
        local_heads = config.get("local_heads", 4)
        max_patch_size = config.get("max_patch_size", 8)
        min_patch_size = config.get("min_patch_size", 2)
        entropy_threshold = config.get("entropy_threshold", 3.0)
        entropy_window = config.get("entropy_window", 8)

        # Entropy-based patcher (no learnable parameters)
        self.patcher = EntropyPatcher(
            min_patch_size=min_patch_size,
            max_patch_size=max_patch_size,
            entropy_threshold=entropy_threshold,
            entropy_window=entropy_window,
        )

        # Local encoder: variable-length patches -> patch representations
        self.src_local_encoder = DynamicLocalEncoder(
            d_model, local_heads, max_patch_size, local_layers, dropout
        )
        self.tgt_local_encoder = DynamicLocalEncoder(
            d_model, local_heads, max_patch_size, local_layers, dropout
        )

        # Global positional encoding (on patches)
        self.src_pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Global encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, norm_type)
            for _ in range(num_layers)
        ])
        self.encoder_norm = _make_norm(norm_type, d_model)

        # Global decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, norm_type)
            for _ in range(num_layers)
        ])
        self.decoder_norm = _make_norm(norm_type, d_model)

        # Local decoder: patch representations -> variable-length byte sequences
        self.local_decoder = DynamicLocalDecoder(
            d_model, local_heads, max_patch_size, min_patch_size,
            local_layers, dropout
        )

        self._init_weights(num_layers)

    def _init_weights(self, num_layers: int):
        local_layers = self.config.get("local_layers", 1)
        global_scale = 1.0 / math.sqrt(2.0 * num_layers)
        local_scale = 1.0 / math.sqrt(2.0 * max(local_layers, 1))

        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if "W_o" in name or "net.2" in name:
                is_local = "local_encoder" in name or "local_decoder" in name
                s = local_scale if is_local else global_scale
                with torch.no_grad():
                    p.mul_(s)

        # Re-zero padding embeddings that Xavier overwrote
        with torch.no_grad():
            self.src_local_encoder.byte_embedding.weight[0].zero_()
            self.tgt_local_encoder.byte_embedding.weight[0].zero_()
            self.local_decoder.byte_embedding.weight[0].zero_()

    def _encode_source(
        self, src_ids: torch.Tensor, src_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Entropy-patch, local-encode, and globally encode the source."""
        src_patch_ids, src_patch_lengths, src_patch_mask = self.patcher.batch_patch(
            src_ids, src_lens
        )
        src_reps, src_patch_mask = self.src_local_encoder(
            src_patch_ids, src_patch_lengths, src_patch_mask
        )
        src_reps = self.src_pos_enc(src_reps)

        src_attn_mask = src_patch_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,P_src)

        enc_out = src_reps
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_attn_mask)
        enc_out = self.encoder_norm(enc_out)

        return enc_out, src_attn_mask

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_lens: torch.Tensor,
        tgt_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Args:
            src_ids:  (B, S_src) source byte embedding IDs (1-256, 0=PAD)
            tgt_ids:  (B, S_tgt) target byte embedding IDs (1-256, 0=PAD)
            src_lens: (B,) true source lengths
            tgt_lens: (B,) true target lengths
        Returns:
            byte_logits:       (B, P_tgt, M, 256)
            eop_logits:        (B, P_tgt, M)
            tgt_patch_ids:     (B, P_tgt, M)
            tgt_patch_lengths: (B, P_tgt)
            tgt_patch_mask:    (B, P_tgt)
        """
        # Source encoding
        enc_out, src_attn_mask = self._encode_source(src_ids, src_lens)

        # Entropy-patch target
        tgt_patch_ids, tgt_patch_lengths, tgt_patch_mask = self.patcher.batch_patch(
            tgt_ids, tgt_lens
        )

        # Local-encode target patches
        tgt_reps, tgt_patch_mask = self.tgt_local_encoder(
            tgt_patch_ids, tgt_patch_lengths, tgt_patch_mask
        )
        tgt_reps = self.tgt_pos_enc(tgt_reps)

        # Global decoder masks
        P_tgt = tgt_reps.size(1)
        causal = torch.triu(
            torch.ones(P_tgt, P_tgt, device=tgt_reps.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        tgt_attn_mask = causal | tgt_patch_mask.unsqueeze(1).unsqueeze(2)

        # Global decoder
        dec_out = tgt_reps
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_attn_mask, src_attn_mask)
        dec_out = self.decoder_norm(dec_out)

        # Local decoder
        byte_logits, eop_logits = self.local_decoder(
            dec_out, tgt_patch_ids, tgt_patch_lengths
        )

        return byte_logits, eop_logits, tgt_patch_ids, tgt_patch_lengths, tgt_patch_mask

    def generate(
        self,
        src_ids: torch.Tensor,
        src_lens: torch.Tensor,
        max_tgt_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive patch-by-patch generation.

        Each generated patch can have a different length in
        [min_patch_size, max_patch_size], determined by the EOP head.

        Returns:
            result:         (B, T_max) predicted byte embedding IDs (1-256)
            result_lengths: (B,)       actual generated byte counts
        """
        B = src_ids.size(0)
        device = src_ids.device
        M = self.patcher.max_patch_size

        if max_tgt_len is None:
            max_tgt_len = self.config.get("max_tgt_len", 128)

        # Source encoding
        enc_out, src_attn_mask = self._encode_source(src_ids, src_lens)

        d_model = self.config["d_model"]
        all_patch_reps: list[torch.Tensor] = []
        all_gen_ids: list[torch.Tensor] = []
        all_gen_lens: list[torch.Tensor] = []
        total_bytes = torch.zeros(B, dtype=torch.long, device=device)

        for _ in range(max_tgt_len):  # upper bound on patches
            if (total_bytes >= max_tgt_len).all():
                break

            # Build global decoder input from previous patch reps + new zero slot
            new_slot = torch.zeros(B, 1, d_model, device=device)
            if all_patch_reps:
                dec_input = torch.cat(all_patch_reps + [new_slot], dim=1)
            else:
                dec_input = new_slot

            dec_input_pe = self.tgt_pos_enc(dec_input)

            cur_len = dec_input.size(1)
            causal_mask = torch.triu(
                torch.ones(cur_len, cur_len, device=device, dtype=torch.bool),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

            x = dec_input_pe
            for layer in self.decoder_layers:
                x = layer(x, enc_out, causal_mask, src_attn_mask)
            dec_out = self.decoder_norm(x)

            # Take the last position (the new patch slot)
            patch_out = dec_out[:, -1:, :]  # (B, 1, D)

            # Generate variable-length patch via local decoder
            gen_ids, gen_lens = self.local_decoder.generate_patch(patch_out)
            all_gen_ids.append(gen_ids)
            all_gen_lens.append(gen_lens)
            total_bytes = total_bytes + gen_lens

            # Local-encode generated bytes as context for the next patch
            gen_len = gen_ids.size(1)
            packed = torch.zeros(B, 1, M, dtype=torch.long, device=device)
            copy_len = min(gen_len, M)
            packed[:, 0, :copy_len] = gen_ids[:, :copy_len]
            packed_lengths = gen_lens.clamp(max=M).unsqueeze(1)  # (B, 1)
            packed_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

            patch_rep, _ = self.tgt_local_encoder(packed, packed_lengths, packed_mask)
            all_patch_reps.append(patch_rep)  # (B, 1, D)

        # Concatenate all generated bytes
        if not all_gen_ids:
            return (
                torch.zeros(B, 1, dtype=torch.long, device=device),
                torch.zeros(B, dtype=torch.long, device=device),
            )

        max_total = min(int(total_bytes.max().item()), max_tgt_len)
        max_total = max(max_total, 1)
        result = torch.zeros(B, max_total, dtype=torch.long, device=device)
        result_lengths = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            offset = 0
            for p_idx in range(len(all_gen_ids)):
                L = int(all_gen_lens[p_idx][b].item())
                remaining = min(L, max_tgt_len - offset)
                if remaining <= 0:
                    break
                gen = all_gen_ids[p_idx]
                copy_len = min(remaining, gen.size(1))
                result[b, offset : offset + copy_len] = gen[b, :copy_len]
                offset += copy_len
            result_lengths[b] = offset

        return result, result_lengths


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
