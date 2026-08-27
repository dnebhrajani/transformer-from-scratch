"""
BPE (Byte Pair Encoding) tokenizer implemented from scratch.

Trains by iteratively merging the most frequent adjacent symbol pairs.
Supports both character-level (for plaintext) and bit-level (for cipher) initial splits.
"""

import json
import re
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """
    From-scratch BPE tokenizer.

    Training:
        1. Split corpus into words (or treat each sequence as one "word")
        2. Initialize vocabulary with individual characters/symbols + special tokens
        3. Iteratively find the most frequent adjacent pair, merge it into a new token
        4. Repeat until vocab_size is reached

    Encoding:
        Apply learned merges greedily to the input text.

    Decoding:
        Concatenate token strings back together.
    """

    def __init__(
        self,
        vocab_size: int = 400,
        special_tokens: list[str] | None = None,
        split_pattern: str | None = None,
    ):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.split_pattern = split_pattern

        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def train(self, corpus: list[str]):
        """
        Train BPE on a corpus of strings.
        Uses indexed pair tracking for efficient incremental updates.
        """
        # Step 1: Pre-tokenize corpus into "words" with frequency counts
        word_freqs = self._get_word_frequencies(corpus)

        # Step 2: Initialize — represent each word as a list of symbols
        # Store as list of (symbols_list, frequency) for efficient access
        words = []
        for word, freq in word_freqs.items():
            words.append([list(word), freq])

        # Step 3: Compute base vocabulary
        base_vocab = set()
        for symbols, _ in words:
            for ch in symbols:
                base_vocab.add(ch)

        self.vocab = {}
        for i, tok in enumerate(self.special_tokens):
            self.vocab[tok] = i
        for ch in sorted(base_vocab):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        # Step 4: Build initial pair counts
        pair_counts = Counter()
        # Track which word indices contain each pair (for incremental updates)
        pair_to_words = {}
        for idx, (symbols, freq) in enumerate(words):
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(idx)

        # Step 5: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)
        self.merges = []

        for _ in range(num_merges):
            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 1:
                break

            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab[new_token] = len(self.vocab)

            # Apply merge only to words that contain this pair
            affected_indices = pair_to_words.pop(best_pair, set())
            del pair_counts[best_pair]

            for idx in affected_indices:
                symbols, freq = words[idx]

                # Remove old pair counts for this word
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    if pair != best_pair:
                        pair_counts[pair] -= freq
                        if pair_counts[pair] <= 0:
                            del pair_counts[pair]
                        if pair in pair_to_words:
                            pair_to_words[pair].discard(idx)

                # Apply merge
                new_symbols = self._apply_merge(symbols, best_pair)
                words[idx][0] = new_symbols

                # Add new pair counts for this word
                for i in range(len(new_symbols) - 1):
                    pair = (new_symbols[i], new_symbols[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
                    if pair not in pair_to_words:
                        pair_to_words[pair] = set()
                    pair_to_words[pair].add(idx)

            if len(self.vocab) >= self.vocab_size:
                break

        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def _get_word_frequencies(self, corpus: list[str]) -> dict[str, int]:
        """Split corpus into words and count frequencies."""
        word_freqs = Counter()

        if self.split_pattern:
            pattern = re.compile(self.split_pattern)
            for text in corpus:
                words = pattern.findall(text)
                for word in words:
                    word_freqs[word] += 1
        else:
            for text in corpus:
                word_freqs[text] += 1

        return word_freqs

    @staticmethod
    def _apply_merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
        """Merge all occurrences of `pair` in the symbol list."""
        merged = []
        i = 0
        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode a string into a list of token IDs.
        Uses priority-based BPE: iteratively merges the pair with lowest merge rank
        until no more merges are possible. Much faster than applying all merges sequentially.
        """
        if self.split_pattern:
            pattern = re.compile(self.split_pattern)
            words = pattern.findall(text)
        else:
            words = [text]

        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab["<bos>"])

        unk_id = self.vocab.get("<unk>", 0)
        for word in words:
            symbols = self._encode_word(word)
            for sym in symbols:
                token_ids.append(self.vocab.get(sym, unk_id))

        if add_special_tokens:
            token_ids.append(self.vocab["<eos>"])

        return token_ids

    def _encode_word(self, word: str) -> list[str]:
        """
        Encode a single word by applying merges sequentially.
        Skips merges whose symbols are not present in the current word (set check).
        """
        if not self.merges:
            return list(word)

        symbols = list(word)
        active = set(symbols)

        for pair in self.merges:
            if pair[0] not in active or pair[1] not in active:
                continue
            new_symbols = self._apply_merge(symbols, pair)
            if len(new_symbols) < len(symbols):
                symbols = new_symbols
                active = set(symbols)

        return symbols

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back into a string."""
        special_ids = set()
        if skip_special_tokens:
            special_ids = {self.vocab[t] for t in self.special_tokens if t in self.vocab}

        tokens = []
        for id_ in ids:
            if id_ in special_ids:
                continue
            token = self.id_to_token.get(id_, "")
            tokens.append(token)

        return "".join(tokens)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    def save(self, path: str):
        """Save tokenizer state to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "split_pattern": self.split_pattern,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer state from JSON."""
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"],
            split_pattern=data["split_pattern"],
        )
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.vocab = data["vocab"]
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.vocab.items()}
        return tokenizer


def train_cipher_tokenizer(
    cipher_texts: list[str], vocab_size: int = 400
) -> BPETokenizer:
    """
    Train a BPE tokenizer on cipher binary strings.

    The initial vocabulary is {'0', '1'} + special tokens.
    BPE learns to merge frequent bit patterns into subword tokens.

    We chunk long cipher strings and sample to keep training tractable.
    """
    import random
    random.seed(42)

    # Chunk cipher strings into fixed-size segments for BPE training
    CHUNK_SIZE = 64
    MAX_CHUNKS = 50000  # cap for training efficiency

    chunked_corpus = []
    for text in cipher_texts:
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i : i + CHUNK_SIZE]
            if len(chunk) >= 16:
                chunked_corpus.append(chunk)

    # Subsample if too many chunks
    if len(chunked_corpus) > MAX_CHUNKS:
        chunked_corpus = random.sample(chunked_corpus, MAX_CHUNKS)

    print(f"  Cipher BPE training on {len(chunked_corpus)} chunks (chunk_size={CHUNK_SIZE})")

    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        split_pattern=None,
    )
    tokenizer.train(chunked_corpus)
    return tokenizer


def train_plaintext_tokenizer(
    plain_texts: list[str], vocab_size: int = 400
) -> BPETokenizer:
    """
    Train a BPE tokenizer on English plaintext.
    Uses word-boundary pre-tokenization so BPE learns subword units.
    """
    split_pattern = r"""[ ]?[a-zA-Z]+|[ ]?[0-9]+|[^ a-zA-Z0-9]+|\s+"""

    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        split_pattern=split_pattern,
    )
    tokenizer.train(plain_texts)
    return tokenizer
