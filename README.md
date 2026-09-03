# Assignment 1: Transformers from Scratch & BLT

**Author:** Durga Nebhrajani

## Links

* **Weights & Biases Dashboard:** 
* **Hugging Face Model Checkpoints:** https://huggingface.co/dnebh/transformer-scratch

## Overview

This project implements an Encoder-Decoder Transformer from scratch in PyTorch for binary ciphertext decryption. Five configurations are implemented:

* **C1:** Base Transformer
* **C2:** RoPE
* **C3:** GQA
* **C4:** RMSNorm
* **C5:** Byte Latent Transformer (BLT)

The implementation does not use `nn.Transformer` or `nn.MultiheadAttention`.

## Repository Structure

```text
2024101144_assignment1/
├── Dataset_A1/
│   ├── README.md
│   ├── brown_cipher.txt
│   └── brown_plain.txt
├── outputs/
│   └── checkpoints/
├── src/
│   ├── models/
│   │   ├── attention.py
│   │   ├── blt.py
│   │   ├── norm.py
│   │   ├── positional.py
│   │   └── transformer.py
│   ├── dataset.py
│   ├── tokenizer.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
├── Report.pdf
└── README.md
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

From the repository root, run:

```bash
python src/train.py --config C1
```

Replace `C1` with `C2`, `C3`, `C4`, or `C5` to train the corresponding configuration.

The best model checkpoint and test metrics are saved under:

```text
outputs/checkpoints/
```

## Configuration Summary

| Configuration | Variant                                                                  |
| ------------- | ------------------------------------------------------------------------ |
| C1            | Base Transformer with sinusoidal positional encoding, MHA, and LayerNorm |
| C2            | C1 with RoPE                                                             |
| C3            | C1 with GQA                                                              |
| C4            | C1 with RMSNorm                                                          |
| C5            | Token-free BLT with byte-level patching                                  |

## Training

The training script supports configuration-specific model and training settings through the `--config` argument.

Example:

```bash
python src/train.py --config C5
```

Training logs and experiment tracking are available through the linked Weights & Biases dashboard.

## Outputs

For each configuration, the following files are generated in its checkpoint directory:

```text
outputs/checkpoints/<config>/
├── best_model.pt
└── test_metrics.json
```

`best_model.pt` contains the best validation-loss checkpoint, while `test_metrics.json` contains the final test-set evaluation metrics. 

## Evaluation

Models are evaluated using greedy decoding. Evaluation includes bit-level accuracy, sequence accuracy, Levenshtein distance, and, where applicable, BLEU and ROUGE-L scores.

## Checkpoints

Trained checkpoints are available through the linked Hugging Face repository:

https://huggingface.co/dnebh/transformer-scratch
