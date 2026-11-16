#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FREQS_GHZ = [2.4, 3.65, 4.9, 5.8, 24.0, 28.0, 39.0]


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.att_src = nn.Parameter(torch.empty(num_heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(num_heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads * out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        n = x.shape[0]
        h = self.lin(x).view(n, self.num_heads, self.out_dim)
        h_src = h[src]
        h_dst = h[dst]
        e = (h_src * self.att_src).sum(-1) + (h_dst * self.att_dst).sum(-1)
        e = F.leaky_relu(e, 0.2)
        dst_h = dst[:, None].expand(-1, self.num_heads)
        max_per_dst = torch.full((n, self.num_heads), -torch.inf, device=x.device, dtype=x.dtype)
        max_per_dst.scatter_reduce_(0, dst_h, e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - max_per_dst[dst])
        denom = torch.zeros((n, self.num_heads), device=x.device, dtype=x.dtype)
        denom.scatter_add_(0, dst_h, exp_e)
        alpha = self.dropout(exp_e / denom[dst].clamp_min(1e-12))
        out = torch.zeros((n, self.num_heads, self.out_dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, alpha[:, :, None] * h_src)
        return out.reshape(n, self.num_heads * self.out_dim) + self.bias


class SuperpixelGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim * num_heads)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.norm1(self.gat1(x, edge_index)))
        x = F.elu(self.norm2(self.gat2(x, edge_index)))
        return self.head(x)


class IDWResidualSuperpixelGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.out_dim = out_dim
        self.gat = SuperpixelGAT(in_dim, hidden_dim, out_dim, num_heads, dropout)
        self.residual_scale = nn.Parameter(torch.full((out_dim,), 0.2))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        idw = x[:, -self.out_dim :]
        return idw + self.gat(x, edge_index) * self.residual_scale


class NodeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x)


class IDWResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = NodeMLP(in_dim, hidden_dim, out_dim, dropout)
        self.residual_scale = nn.Parameter(torch.full((out_dim,), 0.2))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        idw = x[:, -self.out_dim :]
        return idw + self.mlp(x) * self.residual_scale


class MLPPlusGATResidual(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float,
        residual_scale_init: float,
    ):
        super().__init__()
        self.base = NodeMLP(in_dim, hidden_dim, out_dim, dropout)
        self.gat = SuperpixelGAT(in_dim, hidden_dim, out_dim, num_heads, dropout)
        self.residual_scale = nn.Parameter(torch.full((out_dim,), residual_scale_init))
        self.base_frozen = False

    def freeze_base(self) -> None:
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.base_frozen = True
        self.base.eval()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.base_frozen:
            self.base.eval()
            with torch.no_grad():
                base = self.base(x)
        else:
            base = self.base(x)
        return base + self.gat(x, edge_index) * self.residual_scale


def load_mlp_checkpoint_into_base(model: MLPPlusGATResidual, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    mapped = {}
    for key, value in state.items():
        if key.startswith("net."):
            mapped[f"base.{key}"] = value
        elif key.startswith("base.net."):
            mapped[key] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    loaded = len(mapped)
    print(
        f"[warm-start] loaded {loaded} MLP tensors from {checkpoint_path}; "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


class Example:
    def __init__(self, path: Path, input_mode: str):
        d = np.load(path, allow_pickle=False)
        self.path = path
        x = d["feature_matrix"].astype(np.float32)
        self.idw = torch.tensor(d["idw_vectors"], dtype=torch.float32)
        if input_mode == "no_idw":
            x = x[:, :-len(FREQS_GHZ)]
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(d["matrix_gt"], dtype=torch.float32)
        self.train_mask = torch.tensor(d["train_mask"], dtype=torch.bool)
        self.val_mask = torch.tensor(d["val_mask"], dtype=torch.bool)
        self.test_mask = torch.tensor(d["test_mask"], dtype=torch.bool)
        self.edge_index = torch.tensor(d["edge_index"], dtype=torch.long)
        self.labels = d["labels"]
        self.city = str(d["city"])
        self.name = path.stem


def norm_to_rss(x: torch.Tensor) -> torch.Tensor:
    return x * 40.0 - 120.0


def masked_rmse_db(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    pred_db = norm_to_rss(pred).clamp(-120.0, -80.0)
    y_db = norm_to_rss(y)
    return float(torch.sqrt(F.mse_loss(pred_db[mask], y_db[mask])).detach().cpu())


def per_freq_rmse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> list[float]:
    return [masked_rmse_db(pred[:, i], y[:, i], mask[:, i]) for i in range(y.shape[1])]


def load_graphs(root: Path, cities: list[str], max_graphs: int, seed: int, input_mode: str) -> list[Example]:
    paths = []
    for city in cities:
        paths.extend(glob.glob(str(root / city / "**" / "*.npz"), recursive=True))
    paths = [Path(p) for p in sorted(paths)]
    rng = random.Random(seed)
    rng.shuffle(paths)
    if max_graphs > 0:
        paths = paths[:max_graphs]
    return [Example(p, input_mode) for p in paths]


def evaluate(model, examples, device):
    model.eval()
    vals, tests, freq_tests = [], [], []
    with torch.no_grad():
        for ex in examples:
            x = ex.x.to(device)
            y = ex.y.to(device)
            edge = ex.edge_index.to(device)
            pred = model(x, edge)
            vals.append(masked_rmse_db(pred, y, ex.val_mask.to(device)))
            tests.append(masked_rmse_db(pred, y, ex.test_mask.to(device)))
            freq_tests.append(per_freq_rmse(pred, y, ex.test_mask.to(device)))
    return {
        "val_rmse_db": float(np.nanmean(vals)),
        "test_rmse_db": float(np.nanmean(tests)),
        "test_rmse_db_per_freq": np.nanmean(np.array(freq_tests, dtype=np.float32), axis=0).tolist(),
    }


def evaluate_idw(examples):
    vals, freq_vals = [], []
    for ex in examples:
        idw = ex.idw
        vals.append(masked_rmse_db(idw, ex.y, ex.test_mask))
        freq_vals.append(per_freq_rmse(idw, ex.y, ex.test_mask))
    return {
        "idw_test_rmse_db": float(np.nanmean(vals)),
        "idw_test_rmse_db_per_freq": np.nanmean(np.array(freq_vals, dtype=np.float32), axis=0).tolist(),
    }


def make_figure(model, ex: Example, device, out_path: Path, freq_idx: int = 1):
    model.eval()
    with torch.no_grad():
        pred = model(ex.x.to(device), ex.edge_index.to(device)).cpu()
    labels = ex.labels
    h, w = labels.shape
    y_map = np.full((h, w), np.nan, dtype=np.float32)
    p_map = np.full((h, w), np.nan, dtype=np.float32)
    obs_map = np.full((h, w), np.nan, dtype=np.float32)
    y_db = norm_to_rss(ex.y[:, freq_idx]).numpy()
    p_db = norm_to_rss(pred[:, freq_idx]).clamp(-120, -80).numpy()
    obs_vals = ex.x[:, freq_idx].numpy() * 40 - 120
    obs_mask = ex.x[:, len(FREQS_GHZ) + freq_idx].numpy() > 0.5
    for node in range(ex.y.shape[0]):
        pix = labels == node + 1
        y_map[pix] = y_db[node]
        p_map[pix] = p_db[node]
        if obs_mask[node]:
            obs_map[pix] = obs_vals[node]
    err = np.abs(p_map - y_map)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    fig.suptitle(f"Clean superpixel multi-frequency GAT | {ex.name} | {FREQS_GHZ[freq_idx]:.2f} GHz")
    im = axes[0].imshow(obs_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[0].set_title("Sparse train observation")
    axes[1].imshow(p_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[1].set_title("GAT prediction")
    axes[2].imshow(y_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[2].set_title("Ground truth")
    im3 = axes[3].imshow(err, cmap="magma", vmin=0, vmax=15)
    axes[3].set_title("|error| dB")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[:3], fraction=0.025, pad=0.01, label="RSS / path gain (dB)")
    fig.colorbar(im3, ax=axes[3], fraction=0.045, pad=0.01, label="absolute error (dB)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cities", default="Davis,Woodland")
    parser.add_argument("--max-graphs", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--model",
        choices=["gat", "idw_residual_gat", "mlp", "idw_residual_mlp", "mlp_gat_residual"],
        default="idw_residual_gat",
    )
    parser.add_argument("--warm-start-mlp", type=Path, default=None)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--residual-scale-init", type=float, default=0.0)
    parser.add_argument(
        "--input-mode",
        choices=["all", "no_idw"],
        default="all",
        help="all uses sparse/mask/geometry/IDW features; no_idw drops the final IDW feature block.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if args.input_mode == "no_idw" and args.model.startswith("idw_residual"):
        raise ValueError("--input-mode no_idw cannot be used with idw_residual models")
    examples = load_graphs(args.graph_root, cities, args.max_graphs, args.seed, args.input_mode)
    if not examples:
        raise RuntimeError("No graphs found")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if args.model == "idw_residual_gat":
        model = IDWResidualSuperpixelGAT(
            examples[0].x.shape[1], args.hidden_dim, len(FREQS_GHZ), args.num_heads, args.dropout
        )
    elif args.model == "gat":
        model = SuperpixelGAT(
            examples[0].x.shape[1], args.hidden_dim, len(FREQS_GHZ), args.num_heads, args.dropout
        )
    elif args.model == "idw_residual_mlp":
        model = IDWResidualMLP(examples[0].x.shape[1], args.hidden_dim, len(FREQS_GHZ), args.dropout)
    elif args.model == "mlp_gat_residual":
        model = MLPPlusGATResidual(
            examples[0].x.shape[1],
            args.hidden_dim,
            len(FREQS_GHZ),
            args.num_heads,
            args.dropout,
            args.residual_scale_init,
        )
        if args.warm_start_mlp is not None:
            load_mlp_checkpoint_into_base(model, args.warm_start_mlp)
        if args.freeze_base:
            model.freeze_base()
    else:
        model = NodeMLP(examples[0].x.shape[1], args.hidden_dim, len(FREQS_GHZ), args.dropout)
    model = model.to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    baseline = evaluate_idw(examples)
    print(f"[load] graphs={len(examples)} nodes~{examples[0].x.shape[0]} feat={examples[0].x.shape[1]} device={device}")
    print(f"[baseline] {baseline}", flush=True)

    best_val = float("inf")
    best_state = None
    history = []
    initial_metrics = evaluate(model, examples, device)
    best_val = initial_metrics["val_rmse_db"]
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(best_state, args.out_dir / "best_model.pt")
    print(
        f"[initial] val_rmse={initial_metrics['val_rmse_db']:.3f}dB "
        f"test_rmse={initial_metrics['test_rmse_db']:.3f}dB",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        losses = []
        t0 = time.time()
        model.train()
        for ex in examples:
            x = ex.x.to(device)
            y = ex.y.to(device)
            edge = ex.edge_index.to(device)
            mask = ex.train_mask.to(device)
            pred = model(x, edge)
            loss = F.mse_loss(pred[mask], y[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, examples, device)
        metrics["epoch"] = epoch
        metrics["train_loss_norm"] = float(np.mean(losses))
        metrics["seconds"] = time.time() - t0
        history.append(metrics)
        if metrics["val_rmse_db"] < best_val:
            best_val = metrics["val_rmse_db"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, args.out_dir / "best_model.pt")
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:04d}] train_norm={metrics['train_loss_norm']:.5f} "
                f"val_rmse={metrics['val_rmse_db']:.3f}dB test_rmse={metrics['test_rmse_db']:.3f}dB "
                f"sec={metrics['seconds']:.1f}",
                flush=True,
            )
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if best_state:
        model.load_state_dict(best_state)
    final = evaluate(model, examples, device)
    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "num_examples": len(examples),
        "baseline": baseline,
        "best_val_rmse_db": best_val,
        "final": final,
        "freqs_ghz": FREQS_GHZ,
        "examples": [str(ex.path) for ex in examples],
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(model, examples[0], device, args.out_dir / "prediction_example.png")
    print(json.dumps(final, indent=2), flush=True)
    print(f"[done] out_dir={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
