import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import BPETokenizer, train_cipher_tokenizer, train_plaintext_tokenizer, cipher_bits_to_byte_str


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

def segment_pairs(
    ciphers: list[str], plains: list[str], segment_bytes: int = 128
) -> tuple[list[str], list[str], list[int]]:
    """Split cipher/plain pairs into fixed-size segments.

    Each original example is split into segments of `segment_bytes` plaintext
    characters.  The cipher is split at corresponding bit boundaries (8 bits
    per byte).  Brown corpus is pure ASCII so character = byte.

    Segments are aligned so that each segment starts at a position that is
    a multiple of `segment_bytes` within the original text.  Since
    `segment_bytes` should be a multiple of 8 (the XOR key length), every
    segment starts with the same key offset, simplifying the task.

    Applied **after** train/val/test split to prevent data leakage.

    Returns:
        seg_ciphers: segmented cipher strings
        seg_plains: segmented plaintext strings
        line_indices: line_indices[i] = index of the original line that
                      segment i came from (for whole-line reconstruction)
    """
    seg_ciphers: list[str] = []
    seg_plains: list[str] = []
    line_indices: list[int] = []

    for line_idx, (cipher, plain) in enumerate(zip(ciphers, plains)):
        n = len(plain)
        for start in range(0, n, segment_bytes):
            end = min(start + segment_bytes, n)
            chunk_plain = plain[start:end]
            chunk_cipher = cipher[start * 8 : end * 8]
            if len(chunk_plain) >= 8:  # at least 1 full key cycle
                seg_ciphers.append(chunk_cipher)
                seg_plains.append(chunk_plain)
                line_indices.append(line_idx)

    return seg_ciphers, seg_plains, line_indices


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

def test_tokenizer_roundtrip(
    ciphers: list[str],
    plains: list[str],
    src_tokenizer: BPETokenizer,
    tgt_tokenizer: BPETokenizer,
    segment_bytes: int = 128,
):
    """Verify that tokenization + decoding preserves the data exactly."""

    print("\n" + "=" * 60)
    print("TOKENIZER ROUND-TRIP TEST")
    print("=" * 60)

    # Use exactly the same segmentation as training
    cipher_segments, plain_segments, _ = segment_pairs(
        ciphers,
        plains,
        segment_bytes=segment_bytes,
    )

    cipher = cipher_segments[0]
    plain = plain_segments[0]

    # ---------------- SOURCE ----------------
    byte_str = cipher_bits_to_byte_str(cipher)

    src_ids = src_tokenizer.encode(
        byte_str,
        add_special_tokens=True,
    )

    decoded_cipher = src_tokenizer.decode(src_ids)

    print("\nSOURCE / CIPHERTEXT")
    print(f"Cipher bits:  {len(cipher)}")
    print(f"Cipher bytes: {len(byte_str)}")
    print(f"Token count:  {len(src_ids)}")
    print(f"Round-trip:   {byte_str == decoded_cipher}")

    if byte_str != decoded_cipher:
        print("ERROR: Cipher BPE does NOT round-trip!")
        print(f"Original length: {len(byte_str)}")
        print(f"Decoded length:  {len(decoded_cipher)}")

    # ---------------- TARGET ----------------
    tgt_ids = tgt_tokenizer.encode(
        plain,
        add_special_tokens=True,
    )

    decoded_plain = tgt_tokenizer.decode(tgt_ids)

    print("\nTARGET / PLAINTEXT")
    print(f"Plain length: {len(plain)}")
    print(f"Token count:  {len(tgt_ids)}")
    print(f"Round-trip:   {plain == decoded_plain}")

    if plain != decoded_plain:
        print("ERROR: Plaintext BPE does NOT round-trip!")
        print(f"Original: {repr(plain)}")
        print(f"Decoded:  {repr(decoded_plain)}")

    print("\n" + "=" * 60)


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
        line_indices: list[int] | None = None,
    ):
        self.plains = plains
        self.line_indices = line_indices if line_indices is not None else list(range(len(plains)))
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
        # Encode the FULL cipher (no bit-level truncation) so the model
        # sees as much source information as possible; only truncate at
        # the BPE-token level to max_src_len.
        print(f"  Encoding {len(ciphers)} sequences (this may take a few min)...")
        self.src_encoded = []
        for i, cipher in enumerate(ciphers):
            # Convert binary cipher to byte-level string for byte-aligned BPE
            byte_str = cipher_bits_to_byte_str(cipher)
            ids = src_tokenizer.encode(byte_str, add_special_tokens=True)
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
            "line_idx": self.line_indices[idx],
        }


class CipherToTextBLTDataset(Dataset):
    """
    Dataset for configuration C5 (BLT / token-free).

    Source: groups of 8 cipher bits -> actual byte value -> embedding ID (byte+1).
    Target: plaintext UTF-8 bytes -> embedding ID (byte+1).
    Embedding ID 0 is reserved for PAD.
    """

    def __init__(
        self,
        ciphers: list[str],
        plains: list[str],
        max_src_len: int = 512,
        max_tgt_len: int = 512,
        line_indices: list[int] | None = None,
    ):
        self.ciphers = ciphers
        self.plains = plains
        self.line_indices = line_indices if line_indices is not None else list(range(len(plains)))
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.ciphers)

    def __getitem__(self, idx: int) -> dict:
        cipher = self.ciphers[idx]
        plain = self.plains[idx]

        # Source: group every 8 cipher bits into one byte, drop incomplete tail
        # byte value v in [0,255] -> embedding ID v+1;  PAD = 0
        src_ids: list[int] = []
        for i in range(0, len(cipher) - 7, 8):
            byte_val = int(cipher[i : i + 8], 2)
            src_ids.append(byte_val + 1)
        src_ids = src_ids[: self.max_src_len]

        # Target: plaintext -> UTF-8 bytes -> embedding ID (byte+1)
        tgt_ids = [b + 1 for b in plain.encode("utf-8")]
        tgt_ids = tgt_ids[: self.max_tgt_len]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_len": len(src_ids),
            "tgt_len": len(tgt_ids),
            "plain_text": plain,
            "line_idx": self.line_indices[idx],
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
            "line_indices": [item["line_idx"] for item in batch],
        }


def collate_blt(batch: list[dict]) -> dict:
    """Collate function for BLT dataset — pads byte IDs and returns explicit lengths."""
    src_ids = [item["src_ids"] for item in batch]
    tgt_ids = [item["tgt_ids"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=0)

    src_lens = torch.tensor([item["src_len"] for item in batch], dtype=torch.long)
    tgt_lens = torch.tensor([item["tgt_len"] for item in batch], dtype=torch.long)

    return {
        "src_ids": src_padded,
        "tgt_ids": tgt_padded,
        "src_lens": src_lens,
        "tgt_lens": tgt_lens,
        "plain_texts": [item["plain_text"] for item in batch],
        "line_indices": [item["line_idx"] for item in batch],
    }


def get_dataloaders(
    config: dict,
    src_tokenizer: BPETokenizer | None = None,
    tgt_tokenizer: BPETokenizer | None = None,
    splits: dict | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test dataloaders based on config.

    Args:
        config: should have tokenization, batch_size, max_src_len, max_tgt_len, segment_bytes
        src_tokenizer: source BPE tokenizer (required for subword mode)
        tgt_tokenizer: target BPE tokenizer (required for subword mode)
        splits: pre-split data dict {"train": (ciphers, plains), ...}.
                If None, loads raw data and splits internally.

    Returns:
        train_loader, val_loader, test_loader, metadata
        metadata["original_plains"] maps split name → list of original
        (unsegmented) plaintext lines, for whole-line evaluation.
    """
    if splits is None:
        ciphers, plains = load_raw_data()
        splits = train_val_test_split(ciphers, plains)

    # Store original unsegmented plains for whole-line evaluation
    original_plains = {name: list(p) for name, (c, p) in splits.items()}

    # C5 BLT operates on full examples and segments them internally into fixed patches
    segment_bytes = 0 if config.get("tokenization") == "blt" else config.get("segment_bytes", 0)

    # Tokenizer round-trip test (before segmentation, on unsegmented train data)
    if config.get("tokenization", "subword") != "blt":
        test_tokenizer_roundtrip(
            splits["train"][0],
            splits["train"][1],
            src_tokenizer,
            tgt_tokenizer,
            segment_bytes=segment_bytes,
        )

    # --- Dataset segmentation (applied consistently to all splits) ---
    line_indices_map: dict[str, list[int]] = {}
    if segment_bytes > 0:
        segmented = {}
        for name, (c, p) in splits.items():
            seg_c, seg_p, line_idx = segment_pairs(c, p, segment_bytes)
            segmented[name] = (seg_c, seg_p)
            line_indices_map[name] = line_idx
            print(f"  {name}: {len(seg_c)} segments (segment_bytes={segment_bytes})")
        splits = segmented
    else:
        for name, (c, p) in splits.items():
            line_indices_map[name] = list(range(len(p)))

    batch_size = config.get("batch_size", 16)
    max_src_len = config.get("max_src_len", 512)
    max_tgt_len = config.get("max_tgt_len", 512)

    if config.get("tokenization", "subword") == "blt":
        datasets = {
            split: CipherToTextBLTDataset(
                c, p, max_src_len, max_tgt_len,
                line_indices=line_indices_map[split],
            )
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
                line_indices=line_indices_map[split],
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

    metadata = {
        "original_plains": original_plains,
    }
    return loaders["train"], loaders["val"], loaders["test"], metadata
