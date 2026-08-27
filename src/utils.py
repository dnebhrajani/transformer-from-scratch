import torch
import numpy as np
from collections import Counter


# ---------- Bit-Level Accuracy ----------

def text_to_bits(text: str) -> str:
    """Convert a string to its UTF-8 binary representation (0/1 string)."""
    byte_array = text.encode("utf-8")
    return "".join(f"{b:08b}" for b in byte_array)


def bit_level_accuracy(predicted: str, target: str) -> float:
    """
    Compute bit-level accuracy between predicted and target strings.
    Both are first converted to their UTF-8 binary representations,
    then compared bit-by-bit. Shorter string is padded with '0'.
    """
    pred_bits = text_to_bits(predicted)
    tgt_bits = text_to_bits(target)

    max_len = max(len(pred_bits), len(tgt_bits))
    if max_len == 0:
        return 1.0

    pred_bits = pred_bits.ljust(max_len, "0")
    tgt_bits = tgt_bits.ljust(max_len, "0")

    matches = sum(p == t for p, t in zip(pred_bits, tgt_bits))
    return matches / max_len


# ---------- Sequence Accuracy ----------

def sequence_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Percentage of sequences that are perfectly reconstructed."""
    assert len(predictions) == len(targets)
    if len(predictions) == 0:
        return 0.0
    exact = sum(p == t for p, t in zip(predictions, targets))
    return exact / len(predictions)


# ---------- Levenshtein Distance ----------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two strings (dynamic programming)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insert, delete, replace
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def avg_levenshtein(predictions: list[str], targets: list[str]) -> float:
    """Average Levenshtein distance across a set of predictions."""
    assert len(predictions) == len(targets)
    if len(predictions) == 0:
        return 0.0
    total = sum(levenshtein_distance(p, t) for p, t in zip(predictions, targets))
    return total / len(predictions)


# ---------- BLEU Score ----------

def compute_ngrams(tokens: list[str], n: int) -> Counter:
    """Compute n-gram counts for a token sequence."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    prediction: str, reference: str, max_n: int = 4
) -> float:
    """
    Compute sentence-level BLEU score (up to max_n-grams).
    Simple implementation with brevity penalty.
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    if len(pred_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

    # Modified precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = compute_ngrams(pred_tokens, n)
        ref_ngrams = compute_ngrams(ref_tokens, n)

        # Clipped counts
        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    # Geometric mean of precisions (with smoothing for zero precisions)
    log_avg = 0.0
    for p in precisions:
        if p == 0:
            return 0.0
        log_avg += np.log(p)
    log_avg /= max_n

    return bp * np.exp(log_avg)


def corpus_bleu(predictions: list[str], references: list[str], max_n: int = 4) -> float:
    """Average sentence-level BLEU across corpus."""
    if len(predictions) == 0:
        return 0.0
    scores = [bleu_score(p, r, max_n) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


# ---------- ROUGE-L ----------

def lcs_length(x: list, y: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    # Optimized space: only need two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute ROUGE-L (F1, precision, recall) based on longest common subsequence.
    Operates on word-level tokens.
    """
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def corpus_rouge_l(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Average ROUGE-L F1 across corpus."""
    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    scores = [rouge_l(p, r) for p, r in zip(predictions, references)]
    return {
        "precision": sum(s["precision"] for s in scores) / len(scores),
        "recall": sum(s["recall"] for s in scores) / len(scores),
        "f1": sum(s["f1"] for s in scores) / len(scores),
    }


# ---------- Full Evaluation ----------

def evaluate_all(
    predictions: list[str],
    targets: list[str],
    include_bleu_rouge: bool = True,
) -> dict[str, float]:
    """
    Compute all metrics for a list of predictions vs targets.
    Set include_bleu_rouge=False for BLT (C5) which is token-free.
    """
    metrics = {}

    # Bit-level accuracy
    bit_accs = [bit_level_accuracy(p, t) for p, t in zip(predictions, targets)]
    metrics["bit_level_accuracy"] = sum(bit_accs) / max(len(bit_accs), 1)

    # Sequence accuracy
    metrics["sequence_accuracy"] = sequence_accuracy(predictions, targets)

    # Levenshtein distance
    metrics["avg_levenshtein"] = avg_levenshtein(predictions, targets)

    if include_bleu_rouge:
        # BLEU
        metrics["bleu"] = corpus_bleu(predictions, targets)
        # ROUGE-L
        rouge = corpus_rouge_l(predictions, targets)
        metrics["rouge_l_f1"] = rouge["f1"]
        metrics["rouge_l_precision"] = rouge["precision"]
        metrics["rouge_l_recall"] = rouge["recall"]

    return metrics


# ---------- Greedy Decoding ----------

def greedy_decode(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Greedy autoregressive decoding for TransformerSeq2Seq (C1-C4).

    Args:
        model: TransformerSeq2Seq instance
        src: (1, src_len) source token IDs
        src_mask: (1, 1, 1, src_len) source padding mask
        max_len: maximum decode length
        bos_id: beginning-of-sequence token ID
        eos_id: end-of-sequence token ID
        device: torch device
    Returns:
        decoded: (1, decoded_len) predicted token IDs
    """
    model.eval()
    with torch.no_grad():
        enc_output = model.encode(src, src_mask)

        # Start with BOS token
        tgt = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_len = tgt.size(1)
            # Causal mask for decoder
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

            logits = model.decode(tgt, enc_output, causal_mask, src_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == eos_id:
                break

    return tgt


def greedy_decode_batch(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Batched greedy decoding. Sequences that hit EOS are frozen.

    Returns:
        (batch, max_decoded_len) tensor of token IDs
    """
    model.eval()
    batch_size = src.size(0)

    with torch.no_grad():
        enc_output = model.encode(src, src_mask)
        tgt = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_len = tgt.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

            logits = model.decode(tgt, enc_output, causal_mask, src_mask)
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Replace with PAD for finished sequences
            next_tokens[finished] = 0
            tgt = torch.cat([tgt, next_tokens], dim=1)

            # Mark sequences that produced EOS
            finished = finished | (next_tokens.squeeze(-1) == eos_id)
            if finished.all():
                break

    return tgt
