import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import wandb

import model as model_mod
from dataset import Multi30kDataset, collate_batch
from lr_scheduler import NoamScheduler
from model import Transformer
from train import LabelSmoothingLoss, evaluate_bleu, run_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, _ = x.size()
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return self.dropout(x + self.pos_embed(pos_ids))


def make_bleu_loader(val_ds, batch_size: int, pad_idx: int, bleu_limit: int):
    if bleu_limit > 0:
        n = min(bleu_limit, len(val_ds))
        ds = Subset(val_ds, list(range(n)))
    else:
        ds = val_ds
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, pad_idx=pad_idx))


def run_condition(
    name: str,
    use_learned_pos: bool,
    train_loader,
    val_loader,
    bleu_loader,
    src_vocab_size: int,
    tgt_vocab_size: int,
    tgt_vocab,
    args,
    device: str,
):
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        N=args.N,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        checkpoint_path=None,
    ).to(device)

    if use_learned_pos:
        model.pos_enc = LearnedPositionalEncoding(
            d_model=args.d_model,
            dropout=args.dropout,
            max_len=args.max_len_pos,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
    loss_fn = LabelSmoothingLoss(vocab_size=tgt_vocab_size, pad_idx=args.pad_idx, smoothing=args.label_smoothing)

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
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

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        wandb.log(
            {
                "epoch": epoch,
                f"{name}/train_loss": train_loss,
                f"{name}/val_loss": val_loss,
                f"{name}/lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[{name}] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )

    val_bleu = evaluate_bleu(
        model=model,
        test_dataloader=bleu_loader,
        tgt_vocab=tgt_vocab,
        device=device,
        max_len=args.max_decode_len,
    )
    wandb.log({f"{name}/val_bleu": val_bleu})
    print(f"[{name}] Validation BLEU = {val_bleu:.2f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_bleu": val_bleu,
    }


def log_overlay_plots(sin_hist: dict, learned_hist: dict) -> None:
    epochs = list(range(1, len(sin_hist["train_losses"]) + 1))

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, sin_hist["train_losses"], label="Sinusoidal")
    ax1.plot(epochs, learned_hist["train_losses"], label="Learned Pos Emb")
    ax1.set_title("Q2.4 Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    wandb.log({"q4/overlay_train_loss": wandb.Image(fig1)})
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(epochs, sin_hist["val_losses"], label="Sinusoidal")
    ax2.plot(epochs, learned_hist["val_losses"], label="Learned Pos Emb")
    ax2.set_title("Q2.4 Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)
    wandb.log({"q4/overlay_val_loss": wandb.Image(fig2)})
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    labels = ["Sinusoidal", "Learned"]
    values = [sin_hist["val_bleu"], learned_hist["val_bleu"]]
    ax3.bar(labels, values)
    ax3.set_title("Q2.4 Validation BLEU")
    ax3.set_ylabel("BLEU")
    ax3.grid(axis="y", alpha=0.3)
    wandb.log({"q4/val_bleu_bar": wandb.Image(fig3)})
    plt.close(fig3)


def main():
    parser = argparse.ArgumentParser(description="Q2.4: Sinusoidal vs Learned positional embeddings")
    parser.add_argument("--project", type=str, default="da6401-a3-q4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--pad_idx", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_limit", type=int, default=0, help="0 means full train set")
    parser.add_argument("--val_limit", type=int, default=0, help="0 means full val set for val-loss")
    parser.add_argument("--bleu_limit", type=int, default=500, help="0 means full val set for BLEU")
    parser.add_argument("--max_decode_len", type=int, default=100)
    parser.add_argument("--max_len_pos", type=int, default=5000)
    args = parser.parse_args()

    # Train script should not attempt Drive artifact refresh
    model_mod.Transformer._ensure_artifacts_available = lambda self, force_download=False: None

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)
    wandb.init(project=args.project, config=vars(args))

    train_ds = Multi30kDataset("train")
    src_vocab, tgt_vocab = train_ds.build_vocab()
    train_ds.process_data()
    if args.train_limit > 0:
        train_ds.processed_data = train_ds.processed_data[: args.train_limit]

    val_ds = Multi30kDataset("validation")
    val_ds.src_vocab = src_vocab
    val_ds.tgt_vocab = tgt_vocab
    val_ds.src_stoi = dict(src_vocab["stoi"])
    val_ds.tgt_stoi = dict(tgt_vocab["stoi"])
    val_ds.src_itos = list(src_vocab["itos"])
    val_ds.tgt_itos = list(tgt_vocab["itos"])
    val_ds.process_data()
    if args.val_limit > 0:
        val_ds.processed_data = val_ds.processed_data[: args.val_limit]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_idx=args.pad_idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_idx=args.pad_idx),
    )
    bleu_loader = make_bleu_loader(
        val_ds=val_ds,
        batch_size=args.batch_size,
        pad_idx=args.pad_idx,
        bleu_limit=args.bleu_limit,
    )

    # Keep same random init for fair comparison
    set_seed(args.seed)
    sin_hist = run_condition(
        name="sinusoidal",
        use_learned_pos=False,
        train_loader=train_loader,
        val_loader=val_loader,
        bleu_loader=bleu_loader,
        src_vocab_size=len(src_vocab["itos"]),
        tgt_vocab_size=len(tgt_vocab["itos"]),
        tgt_vocab=tgt_vocab,
        args=args,
        device=device,
    )

    set_seed(args.seed)
    learned_hist = run_condition(
        name="learned_pos",
        use_learned_pos=True,
        train_loader=train_loader,
        val_loader=val_loader,
        bleu_loader=bleu_loader,
        src_vocab_size=len(src_vocab["itos"]),
        tgt_vocab_size=len(tgt_vocab["itos"]),
        tgt_vocab=tgt_vocab,
        args=args,
        device=device,
    )

    log_overlay_plots(sin_hist, learned_hist)
    wandb.summary["q4/sinusoidal_val_bleu"] = sin_hist["val_bleu"]
    wandb.summary["q4/learned_val_bleu"] = learned_hist["val_bleu"]
    wandb.summary["q4/bleu_gap_learned_minus_sinusoidal"] = (
        learned_hist["val_bleu"] - sin_hist["val_bleu"]
    )
    wandb.finish()


if __name__ == "__main__":
    main()
