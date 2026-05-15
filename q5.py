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
from train import LabelSmoothingLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, pad_idx: int = 1) -> float:
    pred = logits.argmax(dim=-1)
    non_pad = target.ne(pad_idx)
    correct = (pred.eq(target) & non_pad).sum().item()
    total = non_pad.sum().item()
    return correct / max(total, 1)


def prediction_confidence(logits: torch.Tensor, target: torch.Tensor, pad_idx: int = 1) -> float:
    probs = torch.softmax(logits, dim=-1)
    rows = torch.arange(target.size(0), device=target.device)
    p_true = probs[rows, target]
    non_pad = target.ne(pad_idx)
    if non_pad.any():
        return p_true[non_pad].mean().item()
    return 0.0


def evaluate_epoch(model, data_loader, device: str = "cpu", pad_idx: int = 1):
    model.eval()
    loss_vals = []
    acc_vals = []
    conf_vals = []
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx).to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_out.reshape(-1)

            # plain CE for a consistent validation loss across both settings
            ce = torch.nn.functional.cross_entropy(
                logits_flat, tgt_flat, ignore_index=pad_idx
            )
            loss_vals.append(ce.item())
            acc_vals.append(token_accuracy(logits_flat, tgt_flat, pad_idx))
            conf_vals.append(prediction_confidence(logits_flat, tgt_flat, pad_idx))

    return (
        float(np.mean(loss_vals)) if loss_vals else 0.0,
        float(np.mean(acc_vals)) if acc_vals else 0.0,
        float(np.mean(conf_vals)) if conf_vals else 0.0,
    )


def run_condition(
    name: str,
    smoothing: float,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
    train_loss_fn = LabelSmoothingLoss(vocab_size=tgt_vocab_size, pad_idx=args.pad_idx, smoothing=smoothing)

    train_loss_hist = []
    train_conf_hist = []
    val_loss_hist = []
    val_acc_hist = []
    val_conf_hist = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_confs = []
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, args.pad_idx).to(device)
            tgt_mask = make_tgt_mask(tgt_in, args.pad_idx).to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_out.reshape(-1)

            loss = train_loss_fn(logits_flat, tgt_flat)
            conf = prediction_confidence(logits_flat, tgt_flat, args.pad_idx)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            epoch_confs.append(conf)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_conf = float(np.mean(epoch_confs)) if epoch_confs else 0.0
        val_loss, val_acc, val_conf = evaluate_epoch(model, val_loader, device=device, pad_idx=args.pad_idx)

        train_loss_hist.append(train_loss)
        train_conf_hist.append(train_conf)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        val_conf_hist.append(val_conf)

        wandb.log(
            {
                "epoch": epoch,
                f"{name}/train_loss": train_loss,
                f"{name}/train_confidence": train_conf,
                f"{name}/val_loss": val_loss,
                f"{name}/val_token_acc": val_acc,
                f"{name}/val_confidence": val_conf,
                f"{name}/lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"[{name}] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_conf={train_conf:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_conf={val_conf:.4f}"
        )

    return {
        "train_loss": train_loss_hist,
        "train_conf": train_conf_hist,
        "val_loss": val_loss_hist,
        "val_acc": val_acc_hist,
        "val_conf": val_conf_hist,
    }


def log_overlay_plots(ls_hist: dict, ce_hist: dict) -> None:
    epochs = list(range(1, len(ls_hist["train_loss"]) + 1))

    def _plot(key, title, ylab, wandb_key):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, ls_hist[key], label="eps=0.1")
        ax.plot(epochs, ce_hist[key], label="eps=0.0")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.legend()
        ax.grid(alpha=0.3)
        wandb.log({wandb_key: wandb.Image(fig)})
        plt.close(fig)

    _plot("train_loss", "Q2.5 Train Loss", "Loss", "q5/overlay_train_loss")
    _plot("val_loss", "Q2.5 Validation Loss (CE)", "Loss", "q5/overlay_val_loss")
    _plot("val_acc", "Q2.5 Validation Token Accuracy", "Accuracy", "q5/overlay_val_token_acc")
    _plot("train_conf", "Q2.5 Prediction Confidence (Train)", "Confidence", "q5/overlay_train_confidence")
    _plot("val_conf", "Q2.5 Prediction Confidence (Val)", "Confidence", "q5/overlay_val_confidence")


def main():
    parser = argparse.ArgumentParser(description="Q2.5 ablation: label smoothing sensitivity")
    parser.add_argument("--project", type=str, default="da6401-a3-q5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--pad_idx", type=int, default=1)
    parser.add_argument("--train_limit", type=int, default=0, help="0 means full train set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # For experiment runs we dont need Drive infer downloads
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

    set_seed(args.seed)
    ls_hist = run_condition(
        "ls_0.1",
        0.1,
        train_loader,
        val_loader,
        len(src_vocab["itos"]),
        len(tgt_vocab["itos"]),
        args,
        device,
    )

    set_seed(args.seed)
    ce_hist = run_condition(
        "ls_0.0",
        0.0,
        train_loader,
        val_loader,
        len(src_vocab["itos"]),
        len(tgt_vocab["itos"]),
        args,
        device,
    )

    log_overlay_plots(ls_hist, ce_hist)
    wandb.finish()


if __name__ == "__main__":
    main()
