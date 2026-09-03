import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from dataset import (
    load_raw_data,
    train_val_test_split,
    get_or_train_tokenizers,
    get_dataloaders,
    OUTPUT_DIR,
)
from tokenizer import BPETokenizer
from models.transformer import TransformerSeq2Seq, BLTTransformerSeq2Seq, create_masks
from utils import evaluate_all, greedy_decode_batch


# ---------- Configuration Presets ----------

BASE_CONFIG = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_len": 2048,
    "batch_size": 32,
    "max_src_len": 1024,
    "max_tgt_len": 128,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 500,
    "max_epochs": 100,
    "patience": 15,
    "label_smoothing": 0.1,
    "src_bpe_vocab_size": 4000,
    "tgt_bpe_vocab_size": 4000,
    "segment_bytes": 128,
}


CONFIGS = {
    "C1": {
        **BASE_CONFIG,
        "name": "C1_base",
        "pos_type": "sinusoidal",
        "attn_type": "mha",
        "norm_type": "layernorm",
        "tokenization": "subword",
    },
    "C2": {
        **BASE_CONFIG,
        "name": "C2_rope",
        "pos_type": "rope",
        "attn_type": "mha",
        "norm_type": "layernorm",
        "tokenization": "subword",
    },
    "C3": {
        **BASE_CONFIG,
        "name": "C3_gqa",
        "pos_type": "sinusoidal",
        "attn_type": "gqa",
        "num_kv_heads": 2,
        "norm_type": "layernorm",
        "tokenization": "subword",
    },
    "C4": {
        **BASE_CONFIG,
        "name": "C4_rmsnorm",
        "pos_type": "sinusoidal",
        "attn_type": "mha",
        "norm_type": "rmsnorm",
        "tokenization": "subword",
    },
    "C5": {
        **BASE_CONFIG,
        "name": "C5_blt",
        "pos_type": "sinusoidal",
        "attn_type": "mha",
        "norm_type": "layernorm",
        "tokenization": "blt",
        "patch_size": 8,
        "batch_size": 8,
        "local_layers": 1,
        "local_heads": 4,
        "segment_bytes": 0,  # C5 skips segmentation — full sequences
        "max_src_len": 2670,
        "max_tgt_len": 2670,
        "max_len": 4096,
    },
}



# ---------- Learning Rate Scheduler (Warmup + Cosine Decay) ----------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self) -> float:
        base_lr = self.optimizer.defaults["lr"]
        if self.current_step < self.warmup_steps:
            return base_lr * self.current_step / max(self.warmup_steps, 1)
        else:
            import math
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_last_lr(self) -> float:
        return self._get_lr()


# ---------- Training Loop ----------

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, config):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if config["tokenization"] == "blt":
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)
            src_lens = batch["src_lens"].to(device)
            tgt_lens = batch["tgt_lens"].to(device)

            byte_logits, tgt_patch_ids, tgt_patch_lengths, tgt_patch_mask = (
                model(src, tgt, src_lens, tgt_lens)
            )

            B, P, M, _ = byte_logits.shape

            # Real-position mask: within-patch non-padding AND existing patch
            pos_idx = torch.arange(M, device=device).view(1, 1, M)
            real_mask = (pos_idx < tgt_patch_lengths.unsqueeze(-1)) & ~tgt_patch_mask.unsqueeze(-1)

            # Byte CE loss (embedding IDs 1-256 -> class labels 0-255)
            byte_labels = tgt_patch_ids - 1
            byte_labels[~real_mask] = -100
            loss = criterion(
                byte_logits.reshape(-1, 256),
                byte_labels.reshape(-1),
            )
        else:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            # Decoder input: all tokens except last; labels: all tokens except first
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            src_mask, tgt_mask, memory_mask = create_masks(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask, memory_mask)

            # Flatten for cross-entropy
            vocab_size = logits.size(-1)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt_labels.reshape(-1),
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if config["tokenization"] == "blt":
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)
            src_lens = batch["src_lens"].to(device)
            tgt_lens = batch["tgt_lens"].to(device)

            byte_logits, tgt_patch_ids, tgt_patch_lengths, tgt_patch_mask = (
                model(src, tgt, src_lens, tgt_lens)
            )

            B, P, M, _ = byte_logits.shape
            pos_idx = torch.arange(M, device=device).view(1, 1, M)
            real_mask = (pos_idx < tgt_patch_lengths.unsqueeze(-1)) & ~tgt_patch_mask.unsqueeze(-1)

            byte_labels = tgt_patch_ids - 1
            byte_labels[~real_mask] = -100
            loss = criterion(
                byte_logits.reshape(-1, 256),
                byte_labels.reshape(-1),
            )
        else:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            src_mask, tgt_mask, memory_mask = create_masks(src, tgt_input)
            logits = model(src, tgt_input, src_mask, tgt_mask, memory_mask)

            vocab_size = logits.size(-1)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt_labels.reshape(-1),
            )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def run_evaluation(model, dataloader, tgt_tokenizer, device, config, original_plains=None) -> dict:
    """Run full evaluation with greedy decoding on a dataset split.

    When original_plains is provided and batches carry line_indices,
    segment-level predictions are concatenated back into whole original
    lines before computing metrics.  This gives whole-line sequence
    accuracy instead of per-segment accuracy.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_line_indices = []

    for batch in dataloader:
        if config["tokenization"] == "blt":
            src = batch["src_ids"].to(device)
            src_lens = batch["src_lens"].to(device)

            with torch.no_grad():
                pred_ids, pred_lengths = model.generate(src, src_lens)

            for i in range(pred_ids.size(0)):
                L = pred_lengths[i].item()
                embedding_ids = pred_ids[i, :L].cpu().tolist()
                # Embedding IDs 1-256 -> actual bytes 0-255
                actual_bytes = [(eid - 1) % 256 for eid in embedding_ids]
                text = bytes(actual_bytes).decode("utf-8", errors="replace")
                all_predictions.append(text)
        else:
            src = batch["src"].to(device)
            src_pad_mask = (src == 0).unsqueeze(1).unsqueeze(2)

            bos_id = tgt_tokenizer.bos_id
            eos_id = tgt_tokenizer.eos_id

            decoded = greedy_decode_batch(
                model, src, src_pad_mask, config["max_tgt_len"], bos_id, eos_id, device
            )

            for i in range(decoded.size(0)):
                token_ids = decoded[i].cpu().tolist()
                text = tgt_tokenizer.decode(token_ids, skip_special_tokens=True)
                all_predictions.append(text)

        all_targets.extend(batch["plain_texts"])
        if "line_indices" in batch:
            all_line_indices.extend(batch["line_indices"])

    # Reconstruct whole original lines from segment predictions
    if original_plains is not None and all_line_indices:
        line_preds: dict[int, list[tuple[int, str]]] = {}
        for global_idx, (pred, line_idx) in enumerate(zip(all_predictions, all_line_indices)):
            if line_idx not in line_preds:
                line_preds[line_idx] = []
            line_preds[line_idx].append((global_idx, pred))

        whole_predictions = []
        whole_targets = []
        for line_idx in sorted(line_preds.keys()):
            segs = line_preds[line_idx]
            segs.sort(key=lambda x: x[0])
            whole_pred = "".join(s[1] for s in segs)
            whole_predictions.append(whole_pred)
            whole_targets.append(original_plains[line_idx])

        all_predictions = whole_predictions
        all_targets = whole_targets

    # For BLT (C5), truncate targets to max_tgt_len since model can only
    # generate up to that many bytes.
    if config["tokenization"] == "blt":
        max_tgt_len = config.get("max_tgt_len", 128)
        truncated = []
        for t in all_targets:
            encoded = t.encode("utf-8")[:max_tgt_len]
            truncated.append(encoded.decode("utf-8", errors="ignore"))
        all_targets = truncated

    include_bleu_rouge = config["tokenization"] != "blt"
    return evaluate_all(all_predictions, all_targets, include_bleu_rouge)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Train Transformer Seq2Seq")
    parser.add_argument(
        "--config", type=str, default="C1", choices=["C1", "C2", "C3", "C4", "C5"],
        help="Configuration to run",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overfit_test", action="store_true",
                        help="Overfit a single batch to verify correctness")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    config = CONFIGS[args.config]

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # WandB
    if not args.no_wandb:
        wandb.init(project="anlp-transformer-from-scratch", name=config["name"], config=config)

    # Load data and split FIRST (before tokenizer training to avoid data leakage)
    ciphers, plains = load_raw_data()
    splits = train_val_test_split(ciphers, plains)

    # Tokenizers — train ONLY on the training split
    src_tokenizer = None
    tgt_tokenizer = None
    if config["tokenization"] == "subword":
        train_ciphers, train_plains = splits["train"]
        src_tokenizer, tgt_tokenizer = get_or_train_tokenizers(
            train_ciphers, train_plains,
            src_vocab_size=config["src_bpe_vocab_size"],
            tgt_vocab_size=config["tgt_bpe_vocab_size"],
        )
        config["src_vocab_size"] = src_tokenizer.get_vocab_size()
        config["tgt_vocab_size"] = tgt_tokenizer.get_vocab_size()
        print(f"Source vocab size: {config['src_vocab_size']}")
        print(f"Target vocab size: {config['tgt_vocab_size']}")

    # Dataloaders — pass pre-split data to avoid re-loading
    train_loader, val_loader, test_loader, data_meta = get_dataloaders(
        config, src_tokenizer, tgt_tokenizer, splits=splits
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model
    if config["tokenization"] == "blt":
        model = BLTTransformerSeq2Seq(config).to(device)
    else:
        model = TransformerSeq2Seq(config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    if not args.no_wandb:
        wandb.log({"num_parameters": num_params})

    # Loss & optimizer
    ls = config.get("label_smoothing", 0.0)
    if config["tokenization"] == "blt":
        criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=ls)
    else:
        pad_id = tgt_tokenizer.pad_id
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=ls)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    total_steps = len(train_loader) * config["max_epochs"]
    scheduler = WarmupCosineScheduler(optimizer, config["warmup_steps"], total_steps)

    # Overfit test mode
    if args.overfit_test:
        print("\n=== OVERFIT TEST: training on single batch for 500 steps ===")
        print("Using label_smoothing=0 for overfit diagnostic")

        # Label smoothing must be disabled for a true memorization test.
        if config["tokenization"] == "blt":
            overfit_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            overfit_criterion = nn.CrossEntropyLoss(
                ignore_index=tgt_tokenizer.pad_id
            )

        model.train()
        batch = next(iter(train_loader))

        for step in range(500):
            if config["tokenization"] == "blt":
                src = batch["src_ids"].to(device)
                tgt = batch["tgt_ids"].to(device)
                src_lens = batch["src_lens"].to(device)
                tgt_lens = batch["tgt_lens"].to(device)

                byte_logits, tgt_patch_ids, tgt_patch_lengths, tgt_patch_mask = (
                    model(src, tgt, src_lens, tgt_lens)
                )
                B_ov, P_ov, M_ov, _ = byte_logits.shape
                pos_idx = torch.arange(M_ov, device=device).view(1, 1, M_ov)
                real_mask = (pos_idx < tgt_patch_lengths.unsqueeze(-1)) & ~tgt_patch_mask.unsqueeze(-1)
                byte_labels = tgt_patch_ids - 1
                byte_labels[~real_mask] = -100
                loss = overfit_criterion(
                    byte_logits.reshape(-1, 256),
                    byte_labels.reshape(-1),
                )
            else:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]
                src_mask, tgt_mask, memory_mask = create_masks(src, tgt_input)
                logits = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
                vocab_size = logits.size(-1)
                loss = overfit_criterion(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"  Step {step:4d} | Loss: {loss.item():.4f}")

        if config["tokenization"] == "blt":
            model.eval()
            src = batch["src_ids"].to(device)
            src_lens = batch["src_lens"].to(device)

            with torch.no_grad():
                pred_ids, pred_lengths = model.generate(src, src_lens)

            print("\n=== BLT GREEDY DECODE MEMORIZATION TEST ===")
            for i in range(min(3, src.size(0))):
                target_text = batch["plain_texts"][i]
                L = pred_lengths[i].item()
                eids = pred_ids[i, :L].cpu().tolist()
                pred_bytes = [(eid - 1) % 256 for eid in eids]
                pred_text = bytes(pred_bytes).decode("utf-8", errors="replace")

                print(f"\nExample {i}:")
                print(f"  Target:     {target_text[:100]!r}")
                print(f"  Prediction: {pred_text[:100]!r}")
                print(f"  Exact: {pred_text == target_text}")

            print("\nIf predictions closely match targets, autoregressive decoding is working.")
        else:
            # Greedy-decode the same batch to verify autoregressive inference.
            model.eval()
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            src_mask = (src == tgt_tokenizer.pad_id).unsqueeze(1).unsqueeze(2)

            decoded = greedy_decode_batch(
                model,
                src,
                src_mask,
                config["max_tgt_len"],
                tgt_tokenizer.bos_id,
                tgt_tokenizer.eos_id,
                device,
            )

            print("\n=== GREEDY DECODE MEMORIZATION TEST ===")
            for i in range(min(3, src.size(0))):
                target_text = tgt_tokenizer.decode(
                    tgt[i].cpu().tolist(),
                    skip_special_tokens=True,
                )
                pred_text = tgt_tokenizer.decode(
                    decoded[i].cpu().tolist(),
                    skip_special_tokens=True,
                )

                print(f"\nExample {i}:")
                print(f"  Target:    {target_text[:100]!r}")
                print(f"  Prediction: {pred_text[:100]!r}")
                print(f"  Exact: {pred_text == target_text}")

            print("\nIf predictions closely match targets, autoregressive decoding is working.")

        if loss.item() > 0.1:
            print("  WARNING: Training loss did not converge.")
        else:
            print("  OK: Training loss converged.")
        return

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    save_dir = OUTPUT_DIR / "checkpoints" / config["name"]
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config["max_epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, config
        )
        val_loss = validate(model, val_loader, criterion, device, config)

        elapsed = time.time() - t0

        # GPU memory tracking
        peak_memory_mb = 0
        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats(device)

        lr = scheduler.get_last_lr()

        print(
            f"Epoch {epoch+1:3d}/{config['max_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {lr:.2e} | Time: {elapsed:.1f}s | Mem: {peak_memory_mb:.0f}MB"
        )

        if not args.no_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr,
                "epoch_time_s": elapsed,
            }
            if peak_memory_mb > 0:
                log_dict["peak_gpu_memory_mb"] = peak_memory_mb
            wandb.log(log_dict)

        # Early stopping & checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Final evaluation on test set
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))
    test_original_plains = data_meta["original_plains"]["test"]
    metrics = run_evaluation(model, test_loader, tgt_tokenizer, device, config, test_original_plains)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if not args.no_wandb:
        wandb.log({f"test/{k}": v for k, v in metrics.items()})
        wandb.finish()

    # Save metrics
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
