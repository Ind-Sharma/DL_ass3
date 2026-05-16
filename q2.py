import argparse
import math
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


def scaled_attention_with_scale(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


def scaled_attention_without_scale(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


def mean_qk_grad_norm(model: torch.nn.Module) -> tuple[float, float]:
    q_norms = []
    k_norms = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "W_q.weight" in name:
            q_norms.append(param.grad.detach().norm(2).item())
        if "W_k.weight" in name:
            k_norms.append(param.grad.detach().norm(2).item())

    q_mean = float(np.mean(q_norms)) if q_norms else 0.0
    k_mean = float(np.mean(k_norms)) if k_norms else 0.0
    return q_mean, k_mean


def run_condition(
    name: str,
    use_scaling: bool,
    train_loader,
    src_vocab_size: int,
    tgt_vocab_size: int,
    args,
    device: str,
):
    model_mod.scaled_dot_product_attention = (
        scaled_attention_with_scale if use_scaling else scaled_attention_without_scale
    )

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
    loss_fn = LabelSmoothingLoss(vocab_size=tgt_vocab_size, pad_idx=args.pad_idx, smoothing=0.1)

    step = 0
    losses = []
    q_grads = []
    k_grads = []

    train_iter = iter(train_loader)
    model.train()
    while step < args.max_steps:
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt = next(train_iter)

        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, args.pad_idx).to(device)
        tgt_mask = make_tgt_mask(tgt_in, args.pad_idx).to(device)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_out = tgt_out.reshape(-1)
        loss = loss_fn(logits, tgt_out)

        optimizer.zero_grad()
        loss.backward()

        q_norm, k_norm = mean_qk_grad_norm(model)
        q_grads.append(q_norm)
        k_grads.append(k_norm)
        losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        wandb.log(
            {
                "step": step,
                f"{name}/loss": loss.item(),
                f"{name}/grad_norm_q": q_norm,
                f"{name}/grad_norm_k": k_norm,
                f"{name}/lr": optimizer.param_groups[0]["lr"],
            }
        )

        if step % args.print_every == 0:
            print(
                f"[{name}] step {step}/{args.max_steps} "
                f"loss={loss.item():.4f} q_grad={q_norm:.6f} k_grad={k_norm:.6f}"
            )

    return losses, q_grads, k_grads


def log_plots(
    scaled_losses,
    noscale_losses,
    scaled_q,
    noscale_q,
    scaled_k,
    noscale_k,
):
    steps = list(range(1, len(scaled_losses) + 1))

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, scaled_losses, label="with 1/sqrt(dk)")
    ax1.plot(steps, noscale_losses, label="without scaling")
    ax1.set_title("Q2: Training Loss vs Step")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    wandb.log({"q2/loss_overlay": wandb.Image(fig1)})
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(steps, scaled_q, label="Q grad (scaled)")
    ax2.plot(steps, noscale_q, label="Q grad (no scale)")
    ax2.set_title("Q2: Query Gradient Norm (first 1000 steps)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Gradient Norm")
    ax2.legend()
    ax2.grid(alpha=0.3)
    wandb.log({"q2/query_grad_overlay": wandb.Image(fig2)})
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(steps, scaled_k, label="K grad (scaled)")
    ax3.plot(steps, noscale_k, label="K grad (no scale)")
    ax3.set_title("Q2: Key Gradient Norm (first 1000 steps)")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Gradient Norm")
    ax3.legend()
    ax3.grid(alpha=0.3)
    wandb.log({"q2/key_grad_overlay": wandb.Image(fig3)})
    plt.close(fig3)


def main():
    parser = argparse.ArgumentParser(description="Q2 ablation: scaling factor in attention")
    parser.add_argument("--project", type=str, default="da6401-a3-q2")
    parser.add_argument("--max_steps", type=int, default=1000)
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
    parser.add_argument("--print_every", type=int, default=100)
    args = parser.parse_args()

    # training in this script does not need Drive infer artifacts
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    set_seed(args.seed)
    scaled_losses, scaled_q, scaled_k = run_condition(
        "scaled",
        True,
        train_loader,
        len(src_vocab["itos"]),
        len(tgt_vocab["itos"]),
        args,
        device,
    )

    set_seed(args.seed)
    noscale_losses, noscale_q, noscale_k = run_condition(
        "no_scale",
        False,
        train_loader,
        len(src_vocab["itos"]),
        len(tgt_vocab["itos"]),
        args,
        device,
    )

    log_plots(
        scaled_losses,
        noscale_losses,
        scaled_q,
        noscale_q,
        scaled_k,
        noscale_k,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
