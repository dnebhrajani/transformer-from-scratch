# Assignment 1: Transformers from Scratch & BLT

**Author:** Durga Nebhrajani

## Links

* **Weights & Biases Dashboard:** https://api.wandb.ai/links/dnebhrajani-v/v6ahrvi6 
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
├── README.md
├── Report.pdf
├── requirements.txt
├── outputs/
│   ├── C1_Base_2685842.log
│   ├── C2_RoPE_2685843.log
│   ├── C3_GQA_2685845.log
│   ├── C4_RMSNorm_2685846.log
│   ├── C5_BLT_2686655.log
│   ├── ablation_accuracy.png
│   ├── ablation_quality.png
│   ├── peak_gpu_memory.png
│   ├── train_loss.png
│   ├── val_loss.png
│   └── checkpoints/
│       ├── C1_base/
│       │   └── test_metrics.json
│       ├── C2_rope/
│       │   └── test_metrics.json
│       ├── C3_gqa/
│       │   └── test_metrics.json
│       ├── C4_rmsnorm/
│       │   └── test_metrics.json
│       └── C5_blt/
│           └── test_metrics.json
└── src/
    ├── dataset.py
    ├── tokenizer.py
    ├── train.py
    ├── utils.py
    └── models/
        ├── attention.py
        ├── blt.py
        ├── norm.py
        ├── positional.py
        └── transformer.py

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
