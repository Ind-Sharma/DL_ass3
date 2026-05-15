"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import math
from tqdm import tqdm
import evaluate
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_batch
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0

            pad_mask = target.eq(self.pad_idx)
            true_dist[pad_mask] = 0

        loss = torch.sum(-true_dist * log_probs, dim=1)
        non_pad = target.ne(self.pad_idx)
        return loss[non_pad].mean()


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0

    iterator = tqdm(data_iter, desc=f"{'Train' if is_train else 'Eval'} epoch {epoch_num}", leave=True)
    for src, tgt in iterator:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src).to(device)
        tgt_mask = make_tgt_mask(tgt_input).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt_out.reshape(-1)

        loss = loss_fn(logits_flat, tgt_flat)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_batches += 1
        iterator.set_postfix(loss=loss.item())

    return total_loss / max(total_batches, 1)


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1, dtype=torch.long, device=device) * start_symbol

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys).to(device)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)
            if next_word.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    metric = evaluate.load("bleu")
    predictions = []
    references = []

    pad_idx = 1
    sos_idx = 2
    eos_idx = 3

    def idx_to_token(idx: int) -> str:
        if isinstance(tgt_vocab, dict):
            if "itos" in tgt_vocab:
                return tgt_vocab["itos"][idx]
            if "idx_to_token" in tgt_vocab:
                return tgt_vocab["idx_to_token"][idx]
        if hasattr(tgt_vocab, "itos"):
            return tgt_vocab.itos[idx]
        if hasattr(tgt_vocab, "lookup_token"):
            return tgt_vocab.lookup_token(idx)
        raise ValueError("Unsupported tgt_vocab format.")

    model.eval()
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                src = src_batch[i : i + 1]
                tgt = tgt_batch[i]
                src_mask = make_src_mask(src, pad_idx).to(device)

                pred_ids = greedy_decode(
                    model,
                    src,
                    src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                ).squeeze(0).tolist()

                pred_tokens = []
                for tid in pred_ids:
                    if tid == eos_idx:
                        break
                    if tid in (pad_idx, sos_idx):
                        continue
                    pred_tokens.append(idx_to_token(tid))

                ref_ids = tgt.tolist()
                ref_tokens = []
                for tid in ref_ids:
                    if tid == eos_idx:
                        break
                    if tid in (pad_idx, sos_idx):
                        continue
                    ref_tokens.append(idx_to_token(tid))

                predictions.append(pred_tokens)
                references.append([ref_tokens])

    bleu = metric.compute(predictions=predictions, references=references)["bleu"]
    return bleu * 100.0


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    model_config = getattr(model, "config", None)
    if model_config is None:
        model_config = {
            "src_vocab_size": model.src_embed.num_embeddings,
            "tgt_vocab_size": model.tgt_embed.num_embeddings,
            "d_model": model.src_embed.embedding_dim,
            "N": len(model.encoder.layers),
            "num_heads": model.encoder.layers[0].self_attn.num_heads,
            "d_ff": model.encoder.layers[0].ffn.linear1.out_features,
            "dropout": model.pos_enc.dropout.p,
        }

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model_config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return int(ckpt.get("epoch", 0))


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    config = {
        "batch_size": 64,
        "num_epochs": 10,
        "lr": 1.0,
        "d_model": 256,
        "N": 4,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "label_smoothing": 0.1,
        "pad_idx": 1,
        "checkpoint_path": "checkpoint.pt",
    }

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    train_ds = Multi30kDataset(split="train")
    src_vocab, tgt_vocab = train_ds.build_vocab()
    train_ds.process_data()
    train_ds.save_vocab(".")

    val_ds = Multi30kDataset(split="validation")
    val_ds.src_vocab = src_vocab
    val_ds.tgt_vocab = tgt_vocab
    val_ds.src_stoi = dict(src_vocab["stoi"])
    val_ds.tgt_stoi = dict(tgt_vocab["stoi"])
    val_ds.src_itos = list(src_vocab["itos"])
    val_ds.tgt_itos = list(tgt_vocab["itos"])
    val_ds.process_data()

    test_ds = Multi30kDataset(split="test")
    test_ds.src_vocab = src_vocab
    test_ds.tgt_vocab = tgt_vocab
    test_ds.src_stoi = dict(src_vocab["stoi"])
    test_ds.tgt_stoi = dict(tgt_vocab["stoi"])
    test_ds.src_itos = list(src_vocab["itos"])
    test_ds.tgt_itos = list(tgt_vocab["itos"])
    test_ds.process_data()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(
        src_vocab_size=len(src_vocab["itos"]),
        tgt_vocab_size=len(tgt_vocab["itos"]),
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        checkpoint_path=None,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab["itos"]),
        pad_idx=cfg.pad_idx,
        smoothing=cfg.label_smoothing,
    )

    best_val_loss = math.inf
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )

        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=cfg.checkpoint_path)

    load_checkpoint(cfg.checkpoint_path, model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
