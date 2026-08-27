import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from dataset import (
    load_raw_data,
    train_val_test_split,
    train_bpe_tokenizer,
    load_bpe_tokenizer,
    get_dataloaders,
    TOKENIZER_DIR,
)
from models.transformer import TransformerSeq2Seq, BLTTransformerSeq2Seq, create_masks
from utils import evaluate_all, greedy_decode_batch

#torch.autograd.set_detect_anomaly(True)


# ---------- Configuration Presets ----------

BASE_CONFIG = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_len": 1024,
    "batch_size": 16,
    "max_src_len": 512,
    "max_tgt_len": 512,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 500,
    "max_epochs": 50,
    "patience": 7,
    "bpe_vocab_size": 400,
    "src_vocab_size": 257,  # 0=PAD, 1-256=byte values
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
        "patch_size": 4,
        "local_layers": 1,
        "local_heads": 4,
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
            src = batch["src_bytes"].to(device)
            tgt = batch["tgt_bytes"].to(device)
            src_pad_mask = batch["src_padding_mask"].to(device)
            tgt_pad_mask = batch["tgt_padding_mask"].to(device)

            # For BLT: target input is tgt, target labels are tgt itself (shifted inside model)
            logits = model(src, tgt, src_pad_mask, tgt_pad_mask)

            # Loss: compare logits against target bytes
            # logits: (batch, tgt_len, 256), target: byte values - 1 (0-indexed for classes)
            tgt_labels = tgt.clone()
            tgt_labels[tgt_labels > 0] -= 1  # shift back to 0-255 for loss
            tgt_labels[tgt == 0] = -100  # ignore padding

            # Truncate if logits shorter than target (due to patch alignment)
            min_len = min(logits.size(1), tgt_labels.size(1))
            logits = logits[:, :min_len, :]
            tgt_labels = tgt_labels[:, :min_len]

            #loss = criterion(logits.reshape(-1, 256), tgt_labels.reshape(-1))
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
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
            src = batch["src_bytes"].to(device)
            tgt = batch["tgt_bytes"].to(device)
            src_pad_mask = batch["src_padding_mask"].to(device)
            tgt_pad_mask = batch["tgt_padding_mask"].to(device)

            logits = model(src, tgt, src_pad_mask, tgt_pad_mask)
            tgt_labels = tgt.clone()
            tgt_labels[tgt_labels > 0] -= 1
            tgt_labels[tgt == 0] = -100

            min_len = min(logits.size(1), tgt_labels.size(1))
            logits = logits[:, :min_len, :]
            tgt_labels = tgt_labels[:, :min_len]

            #loss = criterion(logits.reshape(-1, 256), tgt_labels.reshape(-1))
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
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


def run_evaluation(model, dataloader, tokenizer, device, config) -> dict:
    """Run full evaluation with greedy decoding on a dataset split."""
    model.eval()
    all_predictions = []
    all_targets = []

    for batch in dataloader:
        if config["tokenization"] == "blt":
            src = batch["src_bytes"].to(device)
            src_pad_mask = batch["src_padding_mask"].to(device)

            # Use model's generate method
            logits = model.generate(src, src_pad_mask)
            # Convert logits to byte predictions
            pred_bytes = logits.argmax(dim=-1)  # (batch, pred_len)
            for i in range(pred_bytes.size(0)):
                byte_list = pred_bytes[i].cpu().tolist()
                # Convert byte values to string (filter 0s = padding-ish)
                text = bytes([b for b in byte_list if 0 <= b < 256]).decode(
                    "utf-8", errors="replace"
                )
                all_predictions.append(text)
        else:
            src = batch["src"].to(device)
            src_mask = (src == 0).unsqueeze(1).unsqueeze(2)

            bos_id = tokenizer.token_to_id("<bos>")
            eos_id = tokenizer.token_to_id("<eos>")

            decoded = greedy_decode_batch(
                model, src, src_mask, config["max_tgt_len"], bos_id, eos_id, device
            )

            for i in range(decoded.size(0)):
                token_ids = decoded[i].cpu().tolist()
                # Remove BOS/EOS/PAD
                token_ids = [t for t in token_ids if t not in (bos_id, eos_id, 0)]
                text = tokenizer.decode(token_ids)
                all_predictions.append(text)

        all_targets.extend(batch["plain_texts"])

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

    # Tokenizer
    tokenizer = None
    if config["tokenization"] == "subword":
        tokenizer_path = str(TOKENIZER_DIR / "bpe_tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer = load_bpe_tokenizer(tokenizer_path)
            print(f"Loaded tokenizer from {tokenizer_path}")
        else:
            print("Training BPE tokenizer...")
            _, plains = load_raw_data()
            tokenizer = train_bpe_tokenizer(
                plains, vocab_size=config["bpe_vocab_size"], save_path=tokenizer_path
            )
            print(f"Saved tokenizer to {tokenizer_path}")

        config["tgt_vocab_size"] = tokenizer.get_vocab_size()
        print(f"Target vocab size: {config['tgt_vocab_size']}")

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
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
    if config["tokenization"] == "blt":
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        pad_id = tokenizer.token_to_id("<pad>")
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

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
        model.train()
        batch = next(iter(train_loader))

        for step in range(500):
            if config["tokenization"] == "blt":
                src = batch["src_bytes"].to(device)
                tgt = batch["tgt_bytes"].to(device)
                src_pad_mask = batch["src_padding_mask"].to(device)
                tgt_pad_mask = batch["tgt_padding_mask"].to(device)
                logits = model(src, tgt, src_pad_mask, tgt_pad_mask)
                tgt_labels = tgt.clone()
                tgt_labels[tgt_labels > 0] -= 1
                tgt_labels[tgt == 0] = -100
                min_len = min(logits.size(1), tgt_labels.size(1))
                # loss = criterion(
                #     logits[:, :min_len].reshape(-1, 256),
                #     tgt_labels[:, :min_len].reshape(-1),
                # )
                loss = criterion(
                    logits[:, :min_len].reshape(-1, logits.size(-1)),
                    tgt_labels[:, :min_len].reshape(-1),
                )
            else:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]
                src_mask, tgt_mask, memory_mask = create_masks(src, tgt_input)
                logits = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
                vocab_size = logits.size(-1)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"  Step {step:4d} | Loss: {loss.item():.4f}")

        print(f"  Final loss: {loss.item():.6f}")
        if loss.item() > 0.1:
            print("  WARNING: Loss did not converge — likely a bug in masking or model!")
        else:
            print("  OK: Loss converged — model can memorize a batch.")
        return

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    save_dir = TOKENIZER_DIR / "checkpoints" / config["name"]
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
    metrics = run_evaluation(model, test_loader, tokenizer, device, config)

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
