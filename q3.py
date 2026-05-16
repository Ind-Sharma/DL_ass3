import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

import model as model_mod
from dataset import Multi30kDataset
from model import Transformer, make_src_mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_artifacts(artifacts_dir: str, device: str):
    config_path = os.path.join(artifacts_dir, "config.json")
    src_vocab_path = os.path.join(artifacts_dir, "src_vocab.pt")
    tgt_vocab_path = os.path.join(artifacts_dir, "tgt_vocab.pt")
    best_model_path = os.path.join(artifacts_dir, "best_model.pt")

    for path in [config_path, src_vocab_path, tgt_vocab_path, best_model_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing artifact file: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_cfg = cfg.get("model", cfg)

    src_vocab = torch.load(src_vocab_path, map_location="cpu")
    tgt_vocab = torch.load(tgt_vocab_path, map_location="cpu")
    state = torch.load(best_model_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)

    model = Transformer(
        src_vocab_size=int(model_cfg["src_vocab_size"]),
        tgt_vocab_size=int(model_cfg["tgt_vocab_size"]),
        d_model=int(model_cfg["d_model"]),
        N=int(model_cfg["N"]),
        num_heads=int(model_cfg["num_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(model_cfg["dropout"]),
        checkpoint_path=None,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, src_vocab, tgt_vocab, cfg


def select_sentence(args, src_vocab):
    src_stoi = dict(src_vocab["stoi"])
    src_itos = list(src_vocab["itos"])
    pad_idx = src_stoi.get("<pad>", 1)
    sos_idx = src_stoi.get("<sos>", 2)
    eos_idx = src_stoi.get("<eos>", 3)
    unk_idx = src_stoi.get("<unk>", 0)

    if args.sentence is not None:
        temp_ds = Multi30kDataset(split="train")
        tokens = [tok.text.lower() for tok in temp_ds.spacy_de(args.sentence)]
        src_ids = [sos_idx] + [src_stoi.get(tok, unk_idx) for tok in tokens] + [eos_idx]
        src_tokens = ["<sos>"] + tokens + ["<eos>"]
        return torch.tensor(src_ids, dtype=torch.long), src_tokens, args.sentence, pad_idx

    ds = Multi30kDataset(split=args.split)
    ds.src_vocab = src_vocab
    ds.src_stoi = dict(src_vocab["stoi"])
    ds.src_itos = list(src_vocab["itos"])
    # tgt vocab is required by process_data, but content is irrelevant for this script
    ds.tgt_vocab = {"stoi": {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}, "itos": ["<unk>", "<pad>", "<sos>", "<eos>"]}
    ds.tgt_stoi = dict(ds.tgt_vocab["stoi"])
    ds.tgt_itos = list(ds.tgt_vocab["itos"])
    ds.process_data()

    if args.sample_index < 0 or args.sample_index >= len(ds.processed_data):
        raise IndexError(f"sample_index out of range: {args.sample_index}, dataset size: {len(ds.processed_data)}")

    src_tensor, _ = ds.processed_data[args.sample_index]
    src_tokens = [src_itos[idx] if 0 <= idx < len(src_itos) else "<unk>" for idx in src_tensor.tolist()]
    raw_de, raw_en = ds._extract_text_pair(ds.ds[args.sample_index])
    raw_text = f"de: {raw_de} | en: {raw_en}"
    return src_tensor, src_tokens, raw_text, pad_idx


def extract_last_encoder_attention(model: Transformer, src_batch: torch.Tensor, src_mask: torch.Tensor):
    target_attn = model.encoder.layers[-1].self_attn

    def save_attention_hook(module, inputs, output):
        query = inputs[0]
        key = inputs[1]
        value = inputs[2]
        mask = inputs[3] if len(inputs) > 3 else None
        with torch.no_grad():
            q = module._split_heads(module.W_q(query))
            k = module._split_heads(module.W_k(key))
            v = module._split_heads(module.W_v(value))
            _, attn_w = model_mod.scaled_dot_product_attention(q, k, v, mask)
            module.last_attn_weights = attn_w.detach().cpu()

    handle = target_attn.register_forward_hook(save_attention_hook)
    try:
        with torch.no_grad():
            _ = model.encode(src_batch, src_mask)
    finally:
        handle.remove()

    if not hasattr(target_attn, "last_attn_weights"):
        raise RuntimeError("Failed to capture attention weights from last encoder layer.")

    # shape: [batch, heads, seq, seq]
    return target_attn.last_attn_weights[0]


def compute_head_metrics(attn_heads: torch.Tensor) -> dict:
    # attn_heads: [H, S, S]
    H, S, _ = attn_heads.shape
    diag = torch.eye(S, dtype=torch.bool)
    next_diag = torch.zeros((S, S), dtype=torch.bool)
    if S > 1:
        idx = torch.arange(S - 1)
        next_diag[idx, idx + 1] = True
    long_range = torch.abs(torch.arange(S).unsqueeze(1) - torch.arange(S).unsqueeze(0)) >= 3

    diag_scores = []
    next_scores = []
    long_scores = []
    entropy_scores = []

    for h in range(H):
        A = attn_heads[h]
        diag_scores.append(A[diag].mean().item())
        next_scores.append(A[next_diag].mean().item() if S > 1 else 0.0)
        long_scores.append(A[long_range].mean().item() if long_range.any() else 0.0)
        entropy = -(A * (A.clamp_min(1e-9).log())).sum(dim=-1).mean().item()
        entropy_scores.append(entropy)

    flat = attn_heads.reshape(H, -1).float()
    flat = flat / (flat.norm(dim=1, keepdim=True).clamp_min(1e-12))
    sim = torch.matmul(flat, flat.t()).cpu().numpy()

    return {
        "diag": diag_scores,
        "next": next_scores,
        "long": long_scores,
        "entropy": entropy_scores,
        "similarity": sim,
    }


def log_head_heatmaps(attn_heads: torch.Tensor, tokens: list[str], max_tokens_plot: int) -> None:
    H, S, _ = attn_heads.shape
    show_len = min(S, max_tokens_plot)
    show_tokens = tokens[:show_len]

    for h in range(H):
        A = attn_heads[h, :show_len, :show_len].numpy()
        fig, ax = plt.subplots(figsize=(max(6, 0.42 * show_len), max(5, 0.42 * show_len)))
        im = ax.imshow(A, cmap="viridis", aspect="auto")
        ax.set_title(f"Q3: Last Encoder Layer - Head {h}")
        ax.set_xticks(range(show_len))
        ax.set_yticks(range(show_len))
        ax.set_xticklabels(show_tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(show_tokens, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        wandb.log({f"q3/head_{h}_heatmap": wandb.Image(fig)})
        plt.close(fig)


def log_similarity_heatmap(sim: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="magma", vmin=-1.0, vmax=1.0)
    ax.set_title("Q3: Head Similarity (cosine)")
    ax.set_xlabel("Head")
    ax.set_ylabel("Head")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    wandb.log({"q3/head_similarity": wandb.Image(fig)})
    plt.close(fig)


def summarize_metrics(metrics: dict, redundancy_threshold: float) -> str:
    diag = np.array(metrics["diag"])
    nxt = np.array(metrics["next"])
    lng = np.array(metrics["long"])
    ent = np.array(metrics["entropy"])
    sim = metrics["similarity"]

    best_diag = int(diag.argmax())
    best_next = int(nxt.argmax())
    best_long = int(lng.argmax())
    low_ent = int(ent.argmin())

    redundant_pairs = []
    H = sim.shape[0]
    for i in range(H):
        for j in range(i + 1, H):
            if sim[i, j] >= redundancy_threshold:
                redundant_pairs.append((i, j, float(sim[i, j])))

    lines = [
        f"Head with strongest self-focus (diagonal): h{best_diag} (score={diag[best_diag]:.4f})",
        f"Head with strongest next-token bias: h{best_next} (score={nxt[best_next]:.4f})",
        f"Head with strongest long-range focus: h{best_long} (score={lng[best_long]:.4f})",
        f"Lowest-entropy head (most peaky): h{low_ent} (entropy={ent[low_ent]:.4f})",
    ]

    if redundant_pairs:
        show = ", ".join([f"(h{i},h{j})={s:.2f}" for i, j, s in redundant_pairs[:6]])
        lines.append(f"Potential head redundancy (sim>={redundancy_threshold:.2f}): {show}")
    else:
        lines.append(f"No strong redundancy pairs at threshold {redundancy_threshold:.2f}.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Q3: Attention head visualization and specialization")
    parser.add_argument("--project", type=str, default="da6401-a3-q3")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sentence", type=str, default=None, help="Optional custom source sentence")
    parser.add_argument("--max_tokens_plot", type=int, default=32)
    parser.add_argument("--redundancy_threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # this script loads artifacts directly and should not trigger Drive downloads
    model_mod.Transformer._ensure_artifacts_available = lambda self, force_download=False: None

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    run = wandb.init(project=args.project, config=vars(args))

    model, src_vocab, _, cfg = load_artifacts(args.artifacts_dir, device=device)
    src_tensor, src_tokens, raw_text, pad_idx = select_sentence(args, src_vocab)

    src_batch = src_tensor.unsqueeze(0).to(device)
    src_mask = make_src_mask(src_batch, pad_idx).to(device)
    attn_heads = extract_last_encoder_attention(model, src_batch, src_mask)  # [H, S, S]

    print("selected text:", raw_text)
    print("token count:", attn_heads.shape[-1], "| heads:", attn_heads.shape[0])

    log_head_heatmaps(attn_heads, src_tokens, args.max_tokens_plot)
    metrics = compute_head_metrics(attn_heads)
    log_similarity_heatmap(metrics["similarity"])

    summary_text = summarize_metrics(metrics, args.redundancy_threshold)
    print("\n=== Q3 quick analysis ===")
    print(summary_text)

    run.summary["q3_selected_text"] = raw_text
    run.summary["q3_token_count"] = int(attn_heads.shape[-1])
    run.summary["q3_num_heads"] = int(attn_heads.shape[0])
    run.summary["q3_analysis"] = summary_text
    run.summary["q3_pad_idx"] = int(cfg.get("pad_idx", 1))
    run.finish()


if __name__ == "__main__":
    main()
