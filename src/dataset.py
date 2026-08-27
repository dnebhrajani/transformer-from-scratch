import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import BPETokenizer, train_cipher_tokenizer, train_plaintext_tokenizer


DATA_DIR = Path(__file__).resolve().parent.parent / "Dataset_A1"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def load_raw_data() -> tuple[list[str], list[str]]:
    """Load cipher and plain text files, return as list of strings."""
    cipher_path = DATA_DIR / "brown_cipher.txt"
    plain_path = DATA_DIR / "brown_plain.txt"

    with open(cipher_path) as f:
        ciphers = [line.strip() for line in f.readlines()]
    with open(plain_path) as f:
        plains = [line.strip() for line in f.readlines()]

    assert len(ciphers) == len(plains), "Mismatched cipher/plain line counts"
    return ciphers, plains


def train_val_test_split(
    ciphers: list[str], plains: list[str], train_ratio=0.8, val_ratio=0.1
) -> dict:
    """Split data into train/val/test (80/10/10)."""
    n = len(ciphers)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": (ciphers[:train_end], plains[:train_end]),
        "val": (ciphers[train_end:val_end], plains[train_end:val_end]),
        "test": (ciphers[val_end:], plains[val_end:]),
    }


# ---------- Tokenizer Management ----------

def get_or_train_tokenizers(
    cipher_texts: list[str],
    plain_texts: list[str],
    src_vocab_size: int = 400,
    tgt_vocab_size: int = 400,
) -> tuple[BPETokenizer, BPETokenizer]:
    """
    Load saved tokenizers or train new ones.
    Source tokenizer: BPE on binary cipher strings.
    Target tokenizer: BPE on English plaintext.
    """
    src_path = str(OUTPUT_DIR / "src_bpe_tokenizer.json")
    tgt_path = str(OUTPUT_DIR / "tgt_bpe_tokenizer.json")

    if os.path.exists(src_path) and os.path.exists(tgt_path):
        src_tokenizer = BPETokenizer.load(src_path)
        tgt_tokenizer = BPETokenizer.load(tgt_path)
        print(f"Loaded tokenizers: src vocab={src_tokenizer.get_vocab_size()}, tgt vocab={tgt_tokenizer.get_vocab_size()}")
    else:
        print("Training source BPE tokenizer on cipher texts...")
        src_tokenizer = train_cipher_tokenizer(cipher_texts, vocab_size=src_vocab_size)
        os.makedirs(os.path.dirname(src_path), exist_ok=True)
        src_tokenizer.save(src_path)
        print(f"  Source vocab size: {src_tokenizer.get_vocab_size()}")

        print("Training target BPE tokenizer on plaintext...")
        tgt_tokenizer = train_plaintext_tokenizer(plain_texts, vocab_size=tgt_vocab_size)
        tgt_tokenizer.save(tgt_path)
        print(f"  Target vocab size: {tgt_tokenizer.get_vocab_size()}")

    return src_tokenizer, tgt_tokenizer


# ---------- Datasets ----------

class CipherToTextDataset(Dataset):
    """
    Dataset for configurations C1-C4 (learned subword tokenization).
    Source: cipher binary string -> BPE token IDs (learned subword on bit patterns)
    Target: plaintext -> BPE token IDs (learned subword on English)

    Pre-encodes all sequences at init time and caches to disk for reuse.
    """

    def __init__(
        self,
        ciphers: list[str],
        plains: list[str],
        src_tokenizer: BPETokenizer,
        tgt_tokenizer: BPETokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 512,
        cache_path: str | None = None,
    ):
        self.plains = plains
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            import json
            with open(cache_path) as f:
                cached = json.load(f)
            self.src_encoded = cached["src"]
            self.tgt_encoded = cached["tgt"]
            print(f"  Loaded encoded cache from {cache_path} ({len(self.src_encoded)} samples)")
            return

        # Pre-encode all sequences
        avg_bits_per_token = 10
        max_raw_bits = max_src_len * avg_bits_per_token

        print(f"  Encoding {len(ciphers)} sequences (this may take 1-2 min)...")
        self.src_encoded = []
        for i, cipher in enumerate(ciphers):
            truncated = cipher[:max_raw_bits]
            ids = src_tokenizer.encode(truncated, add_special_tokens=True)
            self.src_encoded.append(ids[:max_src_len])
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(ciphers)} encoded...")

        self.tgt_encoded = []
        for plain in plains:
            ids = tgt_tokenizer.encode(plain, add_special_tokens=True)
            self.tgt_encoded.append(ids[:max_tgt_len])

        # Save cache
        if cache_path:
            import json
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"src": self.src_encoded, "tgt": self.tgt_encoded}, f)
            print(f"  Saved encoded cache to {cache_path}")

    def __len__(self) -> int:
        return len(self.plains)

    def __getitem__(self, idx: int) -> dict:
        return {
            "src_ids": torch.tensor(self.src_encoded[idx], dtype=torch.long),
            "tgt_ids": torch.tensor(self.tgt_encoded[idx], dtype=torch.long),
            "plain_text": self.plains[idx],
        }


class CipherToTextBLTDataset(Dataset):
    """
    Dataset for configuration C5 (BLT / token-free).
    Source: cipher binary -> raw byte IDs (each bit as a byte: '0'->1, '1'->2, 0=PAD)
    Target: plaintext -> raw UTF-8 byte IDs (byte_value + 1, 0=PAD)
    """

    def __init__(
        self,
        ciphers: list[str],
        plains: list[str],
        max_src_len: int = 512,
        max_tgt_len: int = 512,
    ):
        self.ciphers = ciphers
        self.plains = plains
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.ciphers)

    def __getitem__(self, idx: int) -> dict:
        cipher = self.ciphers[idx]
        plain = self.plains[idx]

        # Source: each bit character '0' or '1' mapped to byte ID
        # '0' -> 1, '1' -> 2  (0 reserved for PAD)
        src_bytes = [int(b) + 1 for b in cipher]
        src_bytes = src_bytes[: self.max_src_len]

        # Target: plaintext -> UTF-8 bytes, shifted +1 so 0=PAD
        tgt_bytes = [b + 1 for b in plain.encode("utf-8")]
        tgt_bytes = tgt_bytes[: self.max_tgt_len]

        return {
            "src_bytes": torch.tensor(src_bytes, dtype=torch.long),
            "tgt_bytes": torch.tensor(tgt_bytes, dtype=torch.long),
            "plain_text": plain,
        }


# ---------- Collation (dynamic padding) ----------

class TokenizedCollator:
    """Picklable collate function for tokenized dataset (supports num_workers > 0)."""

    def __init__(self, src_pad_id: int = 0, tgt_pad_id: int = 0):
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def __call__(self, batch: list[dict]) -> dict:
        src_ids = [item["src_ids"] for item in batch]
        tgt_ids = [item["tgt_ids"] for item in batch]

        src_padded = torch.nn.utils.rnn.pad_sequence(
            src_ids, batch_first=True, padding_value=self.src_pad_id
        )
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_ids, batch_first=True, padding_value=self.tgt_pad_id
        )

        return {
            "src": src_padded,
            "tgt": tgt_padded,
            "plain_texts": [item["plain_text"] for item in batch],
        }


def collate_blt(batch: list[dict]) -> dict:
    """Collate function for BLT dataset — pads bytes to max length in batch."""
    src_bytes = [item["src_bytes"] for item in batch]
    tgt_bytes = [item["tgt_bytes"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_bytes, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_bytes, batch_first=True, padding_value=0)

    # Padding masks (True = padded position)
    src_padding_mask = src_padded == 0
    tgt_padding_mask = tgt_padded == 0

    return {
        "src_bytes": src_padded,
        "tgt_bytes": tgt_padded,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_padding_mask,
        "plain_texts": [item["plain_text"] for item in batch],
    }


def get_dataloaders(
    config: dict,
    src_tokenizer: BPETokenizer | None = None,
    tgt_tokenizer: BPETokenizer | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders based on config.

    config should have:
        - tokenization: "subword" or "blt"
        - batch_size: int
        - max_src_len: int
        - max_tgt_len: int
    """
    ciphers, plains = load_raw_data()
    splits = train_val_test_split(ciphers, plains)

    batch_size = config.get("batch_size", 16)
    max_src_len = config.get("max_src_len", 512)
    max_tgt_len = config.get("max_tgt_len", 512)

    if config.get("tokenization", "subword") == "blt":
        datasets = {
            split: CipherToTextBLTDataset(c, p, max_src_len, max_tgt_len)
            for split, (c, p) in splits.items()
        }
        collate_fn = collate_blt
    else:
        assert src_tokenizer is not None and tgt_tokenizer is not None, \
            "Both tokenizers required for subword mode"
        cache_dir = str(OUTPUT_DIR / "encoded_cache")
        datasets = {
            split: CipherToTextDataset(
                c, p, src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len,
                cache_path=os.path.join(cache_dir, f"{split}.json"),
            )
            for split, (c, p) in splits.items()
        }
        collate_fn = TokenizedCollator(src_tokenizer.pad_id, tgt_tokenizer.pad_id)

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=2, pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2, pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2, pin_memory=True,
        ),
    }

    return loaders["train"], loaders["val"], loaders["test"]
