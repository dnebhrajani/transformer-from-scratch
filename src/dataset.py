import os
import json
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


DATA_DIR = Path(__file__).resolve().parent.parent / "Dataset_A1"
TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "outputs"


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


# ---------- Source Tokenization (Cipher -> Byte tokens) ----------

def cipher_to_byte_ids(cipher_str: str) -> list[int]:
    """
    Convert binary string to byte-level token IDs.
    Groups bits into bytes (8 bits each), converts to integer 0-255,
    then shifts by +1 so that 0 can serve as PAD.
    Returns list of ints in range [1, 256].
    """
    assert len(cipher_str) % 8 == 0, f"Cipher length {len(cipher_str)} not divisible by 8"
    byte_ids = []
    for i in range(0, len(cipher_str), 8):
        byte_val = int(cipher_str[i : i + 8], 2)
        byte_ids.append(byte_val + 1)  # +1 so 0 = PAD
    return byte_ids


# ---------- Target Tokenization (BPE on English plaintext) ----------

def train_bpe_tokenizer(
    texts: list[str], vocab_size: int = 400, save_path: str | None = None
) -> Tokenizer:
    """
    Train a BPE tokenizer on the plaintext corpus.
    Small vocab (300-500) to avoid data sparsity on 5000 sentences.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add post-processor for BOS/EOS
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tokenizer.save(save_path)

    return tokenizer


def load_bpe_tokenizer(path: str) -> Tokenizer:
    """Load a saved BPE tokenizer."""
    return Tokenizer.from_file(path)


# ---------- Datasets ----------

class CipherToTextDataset(Dataset):
    """
    Dataset for configurations C1-C4 (tokenized).
    Source: cipher binary -> byte IDs (vocab 257: 0=PAD, 1-256=byte values)
    Target: plaintext -> BPE token IDs
    """

    def __init__(
        self,
        ciphers: list[str],
        plains: list[str],
        tokenizer: Tokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 512,
    ):
        self.ciphers = ciphers
        self.plains = plains
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.pad_id = tokenizer.token_to_id("<pad>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    def __len__(self) -> int:
        return len(self.ciphers)

    def __getitem__(self, idx: int) -> dict:
        cipher = self.ciphers[idx]
        plain = self.plains[idx]

        # Source: cipher -> byte IDs (truncate if needed)
        src_ids = cipher_to_byte_ids(cipher)
        src_ids = src_ids[: self.max_src_len]

        # Target: plaintext -> BPE IDs (includes BOS/EOS from post-processor)
        tgt_encoding = self.tokenizer.encode(plain)
        tgt_ids = tgt_encoding.ids[: self.max_tgt_len]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "plain_text": plain,
        }


class CipherToTextBLTDataset(Dataset):
    """
    Dataset for configuration C5 (BLT / token-free).
    Source: cipher binary -> byte IDs (same as above)
    Target: plaintext -> raw UTF-8 byte IDs (no tokenizer)
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

        # Source: cipher -> byte IDs
        src_ids = cipher_to_byte_ids(cipher)
        src_ids = src_ids[: self.max_src_len]

        # Target: plaintext -> UTF-8 bytes, shifted +1 so 0=PAD
        tgt_bytes = [b + 1 for b in plain.encode("utf-8")]
        tgt_bytes = tgt_bytes[: self.max_tgt_len]

        return {
            "src_bytes": torch.tensor(src_ids, dtype=torch.long),
            "tgt_bytes": torch.tensor(tgt_bytes, dtype=torch.long),
            "plain_text": plain,
        }


# ---------- Collation (dynamic padding) ----------

def collate_tokenized(batch: list[dict], pad_id: int = 0) -> dict:
    """Collate function for CipherToTextDataset — pads to max length in batch."""
    src_ids = [item["src_ids"] for item in batch]
    tgt_ids = [item["tgt_ids"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)

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
    config: dict, tokenizer: Tokenizer | None = None
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
        assert tokenizer is not None, "Tokenizer required for subword mode"
        datasets = {
            split: CipherToTextDataset(c, p, tokenizer, max_src_len, max_tgt_len)
            for split, (c, p) in splits.items()
        }
        pad_id = tokenizer.token_to_id("<pad>")
        #collate_fn = lambda batch: collate_tokenized(batch, pad_id)
        collate_fn = partial(collate_tokenized, pad_id=pad_id)

    # Auto-detect: 2 workers for Ada (CUDA), 0 workers for local Mac
    num_workers = 2 if torch.cuda.is_available() else 0

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
        ),
    }

    return loaders["train"], loaders["val"], loaders["test"]
