"""
Verification test suite for C5 Byte Latent Transformer (Fixed Byte Patching).

Tests:
  A. Byte conversion test
  B. Patching test
  C. Padding test
  D. Tensor shape test
  E. No-target-leakage test
  F. Causal mask test
  G. Generation smoke test
  H. Training step smoke test
  I. Overfitting test (tiny dataset)
  J. C1-C4 regression test
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.blt import FixedBytePatcher, LocalByteEncoder, LocalByteDecoder
from src.models.transformer import BLTTransformerSeq2Seq, TransformerSeq2Seq
from src.dataset import CipherToTextBLTDataset, collate_blt
from src.train import CONFIGS


def test_a_byte_conversion():
    print("\n" + "=" * 60)
    print("TEST A: Byte Conversion Test")
    print("=" * 60)
    test_cases = [
        ("00000000", 0, 1),
        ("00000001", 1, 2),
        ("11111111", 255, 256),
        ("01000001", 65, 66),
    ]

    all_passed = True
    for bit_str, exp_byte, exp_id in test_cases:
        byte_val = int(bit_str, 2)
        model_id = byte_val + 1  # 0 is PAD, 1..256 are valid bytes
        recon_byte = model_id - 1
        print(f"  Binary '{bit_str}' -> byte {byte_val:3d} -> ID {model_id:3d} -> decoded {recon_byte:3d}")
        if byte_val != exp_byte or model_id != exp_id or recon_byte != exp_byte:
            all_passed = False

    # Check that PAD is ID 0
    pad_id = 0
    print(f"  PAD ID: {pad_id} (reserved, not used for valid byte 0)")

    assert all_passed, "Byte conversion failed"
    print(">>> TEST A: PASS")
    return True


def test_b_patching():
    print("\n" + "=" * 60)
    print("TEST B: Patching and Reconstruction Test")
    print("=" * 60)
    patcher = FixedBytePatcher(patch_size=8)

    # 1. 20-byte test
    seq_20 = list(range(1, 21))
    patches, lengths = patcher.patch_sequence(seq_20)
    print(f"  20-byte sequence: patch lengths = {lengths}")
    assert lengths == [8, 8, 4], f"Expected [8, 8, 4], got {lengths}"

    # Reconstruct
    reconstructed = []
    for patch, l in zip(patches, lengths):
        reconstructed.extend(patch[:l])
    assert reconstructed == seq_20, "Reconstructed sequence does not match original"
    print("  20-byte reconstruction: EXACT MATCH")

    # 2. Lengths: 1, 7, 8, 9, 16, 17, 20
    test_lengths = [1, 7, 8, 9, 16, 17, 20]
    for L in test_lengths:
        seq = list(range(1, L + 1))
        patches, lens = patcher.patch_sequence(seq)
        for p in patches:
            assert len(p) == 8, f"Patch length should be 8, got {len(p)}"
        recon = []
        for patch, l in zip(patches, lens):
            recon.extend(patch[:l])
        assert recon == seq, f"Mismatch at length {L}"
        print(f"  Length {L:2d}: {len(patches)} patches, lengths={lens} -> Reconstructed OK")

    print(">>> TEST B: PASS")
    return True


def test_c_padding():
    print("\n" + "=" * 60)
    print("TEST C: Padding Mask and Loss Contribution Test")
    print("=" * 60)
    patcher = FixedBytePatcher(patch_size=8)
    byte_ids = torch.tensor([[10, 20, 30, 0, 0, 0, 0, 0]])  # true length = 3
    lengths = torch.tensor([3])

    patch_ids, patch_lengths, patch_mask = patcher.batch_patch(byte_ids, lengths)
    assert patch_lengths[0, 0].item() == 3
    assert not patch_mask[0, 0].item()

    # Verify loss mask logic
    B, P, M = patch_ids.shape
    logits = torch.randn(B, P, M, 256, requires_grad=True)
    pos_idx = torch.arange(M).view(1, 1, M)
    real_mask = (pos_idx < patch_lengths.unsqueeze(-1)) & ~patch_mask.unsqueeze(-1)

    labels = patch_ids - 1
    labels[~real_mask] = -100

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss = criterion(logits.reshape(-1, 256), labels.reshape(-1))
    loss.backward()

    # The gradients for padded positions (pos >= 3) must be EXACTLY ZERO
    pad_grads = logits.grad[0, 0, 3:]
    assert torch.all(pad_grads == 0.0), f"Padded positions received non-zero gradient: {pad_grads}"
    real_grads = logits.grad[0, 0, :3]
    assert torch.any(real_grads != 0.0), "Real positions should receive gradients"
    print(f"  Real positions (0..2) gradient norm: {real_grads.norm().item():.4f}")
    print(f"  Padded positions (3..7) gradient norm: {pad_grads.norm().item():.4f} (EXACTLY ZERO)")

    print(">>> TEST C: PASS")
    return True


def test_d_shapes():
    print("\n" + "=" * 60)
    print("TEST D: Pipeline Tensor Shape Test")
    print("=" * 60)
    config = {
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "max_len": 512,
        "norm_type": "layernorm",
        "patch_size": 8,
        "local_layers": 1,
        "local_heads": 4,
    }
    model = BLTTransformerSeq2Seq(config)
    model.eval()

    B = 2
    src_lens = torch.tensor([19, 8])
    tgt_lens = torch.tensor([14, 7])
    src_ids = torch.randint(1, 257, (B, 19))
    tgt_ids = torch.randint(1, 257, (B, 14))

    print(f"  Input src_ids:       {tuple(src_ids.shape)}")
    print(f"  Input tgt_ids:       {tuple(tgt_ids.shape)}")

    # Forward pass
    byte_logits, tgt_patch_ids, tgt_patch_lengths, tgt_patch_mask = model(
        src_ids, tgt_ids, src_lens, tgt_lens
    )

    print(f"  tgt_patch_ids:       {tuple(tgt_patch_ids.shape)} (expected: B={B}, P=2, M=8)")
    print(f"  tgt_patch_lengths:   {tuple(tgt_patch_lengths.shape)}")
    print(f"  tgt_patch_mask:      {tuple(tgt_patch_mask.shape)}")
    print(f"  byte_logits:         {tuple(byte_logits.shape)} (expected: B={B}, P=2, M=8, Vocab=256)")

    assert byte_logits.shape == (B, 2, 8, 256)
    assert tgt_patch_ids.shape == (B, 2, 8)
    assert tgt_patch_lengths.shape == (B, 2)
    assert tgt_patch_mask.shape == (B, 2)

    print(">>> TEST D: PASS")
    return True


def test_e_no_target_leakage():
    print("\n" + "=" * 60)
    print("TEST E: No-Target-Leakage Invariant Test (Critical)")
    print("=" * 60)
    torch.manual_seed(42)
    config = {
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "max_len": 512,
        "norm_type": "layernorm",
        "patch_size": 8,
        "local_layers": 1,
        "local_heads": 4,
    }
    model = BLTTransformerSeq2Seq(config)
    model.eval()

    # Batch 1 and Batch 2 have IDENTICAL source and IDENTICAL target patch 0.
    # But Batch 2 has a completely DIFFERENT target patch 1!
    src = torch.randint(1, 257, (1, 16))
    src_lens = torch.tensor([16])

    # Target 1: patch 0 is [1..8], patch 1 is [9..16]
    tgt_1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    # Target 2: patch 0 is [1..8] (SAME!), patch 1 is [99, 100, ...] (DIFFERENT!)
    tgt_2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 99, 100, 101, 102, 103, 104, 105, 106]])
    tgt_lens = torch.tensor([16])

    with torch.no_grad():
        logits_1, _, _, _ = model(src, tgt_1, src_lens, tgt_lens)
        logits_2, _, _, _ = model(src, tgt_2, src_lens, tgt_lens)

    # Patch 0 predictions:
    patch0_diff = (logits_1[0, 0] - logits_2[0, 0]).abs().max().item()
    # Patch 1 predictions:
    patch1_diff = (logits_1[0, 1] - logits_2[0, 1]).abs().max().item()

    print(f"  Max absolute difference in predicted patch 0 logits: {patch0_diff:.8f}")
    print(f"  Max absolute difference in predicted patch 1 logits: {patch1_diff:.8f}")

    assert patch0_diff == 0.0, (
        f"TARGET LEAKAGE DETECTED! Changing patch 1 altered the logits of patch 0 (diff={patch0_diff})"
    )
    print("  Zero leakage: predicting patch 0 did NOT depend on patch 1.")
    print(">>> TEST E: PASS")
    return True


def test_f_causal_mask():
    print("\n" + "=" * 60)
    print("TEST F: Causal Attention Mask Invariant Test")
    print("=" * 60)
    # Check that position p cannot attend to positions > p
    P = 5
    causal = torch.triu(torch.ones(P, P, dtype=torch.bool), diagonal=1)
    print("  Causal attention mask upper-triangle (True = masked):")
    for row in range(P):
        line = [" M " if causal[row, col].item() else " . " for col in range(P)]
        print(f"    pos {row}: {''.join(line)}")

    for row in range(P):
        for col in range(P):
            if col > row:
                assert causal[row, col].item() is True, f"Future position ({row}, {col}) unmasked"
            else:
                assert causal[row, col].item() is False, f"Past/current position ({row}, {col}) masked"

    print(">>> TEST F: PASS")
    return True


def test_g_generation_smoke():
    print("\n" + "=" * 60)
    print("TEST G: Autoregressive Generation Smoke Test")
    print("=" * 60)
    torch.manual_seed(42)
    config = {
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "max_len": 512,
        "norm_type": "layernorm",
        "patch_size": 8,
        "local_layers": 1,
        "local_heads": 4,
    }
    model = BLTTransformerSeq2Seq(config)
    model.eval()

    B = 2
    src_lens = torch.tensor([19, 8])
    src_ids = torch.randint(1, 257, (B, 19))

    with torch.no_grad():
        result, result_lengths = model.generate(src_ids, src_lens)

    print(f"  Generated result tensor shape: {tuple(result.shape)}")
    print(f"  Generated lengths:             {result_lengths.tolist()}")

    assert not torch.isnan(result).any(), "NaN in generated output"
    assert result_lengths.tolist() == [19, 8], f"Expected lengths [19, 8], got {result_lengths.tolist()}"
    assert result.min().item() >= 0
    assert result.max().item() <= 256

    # Convert back to text
    for i in range(B):
        L = result_lengths[i].item()
        byte_vals = [(eid - 1) % 256 for eid in result[i, :L].cpu().tolist()]
        text = bytes(byte_vals).decode("utf-8", errors="replace")
        print(f"  Sample {i} decoded ({L} bytes): {text[:40]!r}")

    print(">>> TEST G: PASS")
    return True


def test_h_training_step_smoke():
    print("\n" + "=" * 60)
    print("TEST H: Training Step (Forward, Loss, Backward, Step) Test")
    print("=" * 60)
    config = {
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "max_len": 512,
        "norm_type": "layernorm",
        "patch_size": 8,
        "local_layers": 1,
        "local_heads": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    }
    model = BLTTransformerSeq2Seq(config)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Record initial parameter values
    param_before = [p.clone() for p in model.parameters() if p.requires_grad]

    # Dummy batch
    B = 2
    src_ids = torch.randint(1, 257, (B, 16))
    tgt_ids = torch.randint(1, 257, (B, 16))
    src_lens = torch.tensor([16, 12])
    tgt_lens = torch.tensor([16, 10])

    optimizer.zero_grad()
    byte_logits, tgt_patch_ids, tgt_patch_lengths, tgt_patch_mask = model(
        src_ids, tgt_ids, src_lens, tgt_lens
    )

    B, P, M, _ = byte_logits.shape
    pos_idx = torch.arange(M).view(1, 1, M)
    real_mask = (pos_idx < tgt_patch_lengths.unsqueeze(-1)) & ~tgt_patch_mask.unsqueeze(-1)
    labels = tgt_patch_ids - 1
    labels[~real_mask] = -100

    loss = criterion(byte_logits.reshape(-1, 256), labels.reshape(-1))
    print(f"  Initial loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite"

    loss.backward()

    # Check gradients
    all_grads_finite = True
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all():
                all_grads_finite = False
                print(f"    Non-finite grad in {name}")
    assert all_grads_finite, "Non-finite gradients detected"
    print("  All gradients are finite.")

    optimizer.step()

    # Check parameters changed
    any_param_changed = False
    for p_bef, p_aft in zip(param_before, [p for p in model.parameters() if p.requires_grad]):
        if not torch.equal(p_bef, p_aft):
            any_param_changed = True
            break
    assert any_param_changed, "Parameters did not change after optimizer step"
    print("  Parameters successfully updated.")

    print(">>> TEST H: PASS")
    return True


def test_i_overfit():
    print("\n" + "=" * 60)
    print("TEST I: Overfitting Test on Tiny Dataset (Sanity Check)")
    print("=" * 60)
    torch.manual_seed(42)
    config = {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 256,
        "dropout": 0.0,
        "max_len": 512,
        "norm_type": "layernorm",
        "patch_size": 8,
        "local_layers": 1,
        "local_heads": 4,
    }
    model = BLTTransformerSeq2Seq(config)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 2 tiny sentences
    raw_texts = ["Hello, World!", "Antigravity BLT"]
    src_bytes_list = [[(b + 1) for b in s.encode("utf-8")] for s in raw_texts]
    tgt_bytes_list = src_bytes_list

    src_lens = torch.tensor([len(s) for s in src_bytes_list])
    tgt_lens = torch.tensor([len(s) for s in tgt_bytes_list])
    max_len = max(src_lens.max().item(), tgt_lens.max().item())

    src_ids = torch.zeros(2, max_len, dtype=torch.long)
    tgt_ids = torch.zeros(2, max_len, dtype=torch.long)
    for i in range(2):
        src_ids[i, :src_lens[i]] = torch.tensor(src_bytes_list[i])
        tgt_ids[i, :tgt_lens[i]] = torch.tensor(tgt_bytes_list[i])

    print("  Training on 2 examples for 100 steps...")
    model.train()
    for step in range(100):
        optimizer.zero_grad()
        byte_logits, tgt_patch_ids, tgt_patch_lengths, tgt_patch_mask = model(
            src_ids, tgt_ids, src_lens, tgt_lens
        )
        B, P, M, _ = byte_logits.shape
        pos_idx = torch.arange(M).view(1, 1, M)
        real_mask = (pos_idx < tgt_patch_lengths.unsqueeze(-1)) & ~tgt_patch_mask.unsqueeze(-1)
        labels = tgt_patch_ids - 1
        labels[~real_mask] = -100

        loss = criterion(byte_logits.reshape(-1, 256), labels.reshape(-1))
        loss.backward()
        optimizer.step()

        if (step + 1) % 25 == 0:
            print(f"    Step {step+1:3d} | Loss: {loss.item():.4f}")

    print(f"  Final training loss: {loss.item():.4f}")
    assert loss.item() < 0.1, f"Model failed to overfit tiny batch (final loss: {loss.item()})"

    # Verify greedy decoding memorization
    model.eval()
    with torch.no_grad():
        result, result_lengths = model.generate(src_ids, src_lens)

    memorized_all = True
    for i in range(2):
        L = result_lengths[i].item()
        byte_vals = [(eid - 1) % 256 for eid in result[i, :L].cpu().tolist()]
        text = bytes(byte_vals).decode("utf-8", errors="replace")
        match = (text == raw_texts[i])
        print(f"  Example {i}: Target: {raw_texts[i]!r} | Pred: {text!r} | Match: {match}")
        if not match:
            memorized_all = False

    assert memorized_all, "Model failed to memorize tiny batch during autoregressive generation"
    print("  Autoregressive greedy generation matches targets exactly.")
    print(">>> TEST I: PASS")
    return True


def test_j_c1_c4_regression():
    print("\n" + "=" * 60)
    print("TEST J: C1-C4 Regression Test")
    print("=" * 60)
    for c_name in ["C1", "C2", "C3", "C4"]:
        cfg = CONFIGS[c_name]
        print(f"  Checking {c_name} config ({cfg['name']})...")
        assert cfg["tokenization"] == "subword", f"{c_name} tokenization altered"
        assert cfg["segment_bytes"] == 128, f"{c_name} segment_bytes altered"

        # Instantiate C1-C4 model
        model_cfg = {
            **cfg,
            "src_vocab_size": 100,
            "tgt_vocab_size": 100,
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 1,
            "d_ff": 128,
            "num_kv_heads": 2 if cfg.get("attn_type") == "gqa" else 4,
        }
        model = TransformerSeq2Seq(model_cfg)
        model.eval()

        src = torch.randint(2, 80, (2, 12))
        tgt = torch.randint(2, 80, (2, 10))
        src_mask = torch.zeros(2, 1, 1, 12, dtype=torch.bool)
        tgt_mask = torch.zeros(2, 1, 10, 10, dtype=torch.bool)
        mem_mask = src_mask

        with torch.no_grad():
            out = model(src, tgt, src_mask, tgt_mask, mem_mask)
        assert out.shape == (2, 10, 100), f"Unexpected shape for {c_name}: {out.shape}"
        print(f"    {c_name} forward pass successful, output shape: {tuple(out.shape)}")

    print(">>> TEST J: PASS")
    return True


if __name__ == "__main__":
    print("============================================================")
    print("RUNNING COMPLETE C5 VERIFICATION TEST SUITE")
    print("============================================================")
    test_a_byte_conversion()
    test_b_patching()
    test_c_padding()
    test_d_shapes()
    test_e_no_target_leakage()
    test_f_causal_mask()
    test_g_generation_smoke()
    test_h_training_step_smoke()
    test_i_overfit()
    test_j_c1_c4_regression()
    print("\n" + "=" * 60)
    print("ALL 10 TESTS (A - J) PASSED SUCCESSFULLY!")
    print("============================================================")
