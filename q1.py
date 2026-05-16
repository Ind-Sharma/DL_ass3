import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

import model as model_mod
from dataset import Multi30kDataset, collate_batch
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask
from train import LabelSmoothingLoss, run_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def val_token_accuracy(model, val_loader, device: str = "cpu", pad_idx: int = 1) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx).to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            pred = logits.argmax(dim=-1)

            non_pad = tgt_out.ne(pad_idx)
            correct += (pred.eq(tgt_out) & non_pad).sum().item()
            total += non_pad.sum().item()
    return correct / max(total, 1)


def run_condition(
    name: str,
    use_noam: bool,
    train_loader,
    val_loader,
    src_vocab_size: int,
    tgt_vocab_size: int,
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

    if use_noam:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.fixed_lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = None

    loss_fn = LabelSmoothingLoss(vocab_size=tgt_vocab_size, pad_idx=args.pad_idx, smoothing=0.1)

    train_losses = []
    val_accs = []
    lrs = []

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
        val_acc = val_token_accuracy(model, val_loader, device=device, pad_idx=args.pad_idx)
        lr_now = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        lrs.append(lr_now)

        wandb.log(
            {
                "epoch": epoch,
                f"{name}/train_loss": train_loss,
                f"{name}/val_token_acc": val_acc,
                f"{name}/lr": lr_now,
            }
        )

        print(
            f"[{name}] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.6e}"
        )

    return train_losses, val_accs, lrs


def log_overlay_plots(noam_losses, fixed_losses, noam_accs, fixed_accs) -> None:
    epochs = list(range(1, len(noam_losses) + 1))

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, noam_losses, label="Noam")
    ax1.plot(epochs, fixed_losses, label="Fixed LR")
    ax1.set_title("Q1: Training Loss (Noam vs Fixed LR)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    wandb.log({"q1/overlay_train_loss": wandb.Image(fig1)})
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(epochs, noam_accs, label="Noam")
    ax2.plot(epochs, fixed_accs, label="Fixed LR")
    ax2.set_title("Q1: Validation Token Accuracy (Noam vs Fixed LR)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Token Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    wandb.log({"q1/overlay_val_token_acc": wandb.Image(fig2)})
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Q1 experiment: Noam vs Fixed LR")
    parser.add_argument("--project", type=str, default="da6401-a3-q1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--fixed_lr", type=float, default=1e-4)
    parser.add_argument("--pad_idx", type=int, default=1)
    parser.add_argument("--train_limit", type=int, default=0, help="0 means full train set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # keep training path clean in notebooks/servers where Drive links are used for infer only
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # Keep same seed before each condition for fair init
    set_seed(args.seed)
    noam_losses, noam_accs, _ = run_condition(
        "noam",
        True,
        train_loader,
        val_loader,
        len(src_vocab["itos"]),
        len(tgt_vocab["itos"]),
        args,
        device,
    )

    set_seed(args.seed)
    fixed_losses, fixed_accs, _ = run_condition(
        "fixed",
        False,
        train_loader,
        val_loader,
        len(src_vocab["itos"]),
        len(tgt_vocab["itos"]),
        args,
        device,
    )

    log_overlay_plots(noam_losses, fixed_losses, noam_accs, fixed_accs)
    wandb.finish()


if __name__ == "__main__":
    main()
