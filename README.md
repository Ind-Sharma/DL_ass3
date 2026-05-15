# DA6401 Assignment 3: Transformer-Based Machine Translation (PyTorch)

## Overview

This project implements the **Transformer architecture** from *Attention Is All You Need* for **German to English neural machine translation** on **Multi30k**.  
Core components include:

- **Scaled Dot-Product Attention**
- **Custom Multi-Head Attention** (without `torch.nn.MultiheadAttention`)
- **Sinusoidal Positional Encoding**
- **Encoder-Decoder Transformer stack**
- **Noam Learning Rate Scheduler**
- **Label Smoothing**
- **Greedy decoding-based inference**

The submission-facing API is compatible with the assignment autograder, including `Transformer().infer(...)`.

## Links

- **Weights & Biases Report (public):** _to be added_
- **GitHub Repository:** https://github.com/Ind-Sharma/DL_ass3

## Usage

Run commands from the **`da6401_assignment_3`** directory:

```bash
cd da6401_assignment_3
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install gdown evaluate sacrebleu
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### Training

Run full training pipeline:

```bash
python train.py
```

This runs `run_training_experiment()` and trains the model with Noam scheduler + label smoothing.

### Export inference artifacts from checkpoint

```bash
python -c "from train import export_inference_artifacts_from_checkpoint; export_inference_artifacts_from_checkpoint('checkpoint.pt', artifacts_dir='artifacts')"
```

Expected files:

- `artifacts/best_model.pt`
- `artifacts/src_vocab.pt`
- `artifacts/tgt_vocab.pt`
- `artifacts/config.json`

### Inference sanity check

```bash
python -c "from model import Transformer; m=Transformer(); print(m.infer('ein mann spielt gitarre'))"
```

## Project Files

```text
da6401_assignment_3/
├── requirements.txt
├── README.md
├── model.py
├── dataset.py
├── lr_scheduler.py
└── train.py
```
