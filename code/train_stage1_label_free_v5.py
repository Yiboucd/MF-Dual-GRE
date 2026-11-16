#!/usr/bin/env python3
"""Train Stage-1 models without dense labels or interpolation priors."""

from __future__ import annotations

import argparse
import copy
import json
import math
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

from train_clean_superpixel_gat import FREQS_GHZ, GATLayer, NodeMLP, SuperpixelGAT
from train_clean_superpixel_gat_v2 import ExampleV2, load_examples


NF = len(FREQS_GHZ)
SUB6 = (0, 1, 2, 3)
MMWAVE = (4, 5, 6)
GEOM = slice(2 * NF, 2 * NF + 5)


class ResidualGATBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.norm1 = nn.LayerNorm(width)
        self.attn = GATLayer(width, width // heads, heads, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), edge_index))
        return x + self.dropout(self.ffn(self.norm2(x)))


class PureFrequencyGroupBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        width: int,
        out_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.local_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_dim))
        self.blocks = nn.ModuleList(
            [ResidualGATBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.graph_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_dim))
        self.gate_logits = nn.Parameter(torch.zeros(out_dim))

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        local_hidden = self.encoder(features)
        local = torch.sigmoid(self.local_head(local_hidden))
        graph_hidden = local_hidden
        for block in self.blocks:
            graph_hidden = block(graph_hidden, edge_index)
        graph = torch.sigmoid(self.graph_head(graph_hidden))
        gate = torch.sigmoid(self.gate_logits)[None, :]
        return local + gate * (graph - local)


def group_features(x: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    sparse = x.index_select(1, idx)
    masks = x.index_select(1, idx + NF)
    return torch.cat([sparse, masks, x[:, GEOM]], dim=1)


class PureGroupedMFGAT(nn.Module):
    """Separate sub-6 and mmWave GAT branches with no IDW input or output prior."""

    def __init__(self, width: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.sub6 = PureFrequencyGroupBranch(
            2 * len(SUB6) + 5, width, len(SUB6), heads, layers, dropout
        )
        self.mmwave = PureFrequencyGroupBranch(
            2 * len(MMWAVE) + 5, width, len(MMWAVE), heads, layers, dropout
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.sub6(group_features(x, SUB6), edge_index),
                self.mmwave(group_features(x, MMWAVE), edge_index),
            ],
            dim=1,
        )


def graph_batches(examples: list[ExampleV2], batch_size: int):
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        xs, targets, masks, edges = [], [], [], []
        offset = 0
        for example in batch:
            xs.append(example.x)
            targets.append(example.train_target)
            masks.append(example.train_mask)
            edges.append(example.edge_index + offset)
            offset += example.x.shape[0]
        yield torch.cat(xs), torch.cat(targets), torch.cat(masks), torch.cat(edges, dim=1)


def frequency_balanced_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for fi in range(NF):
        active = mask[:, fi]
        if active.any():
            losses.append(F.smooth_l1_loss(pred[active, fi], target[active, fi], beta=0.025))
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def graph_smoothness(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    active = src != dst
    if not active.any():
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[src[active]], pred[dst[active]], beta=0.025)


def target_mask_weight(example: ExampleV2, split: str):
    if split == "val":
        return (
            example.val_target,
            example.val_mask,
            example.val_pixel_weights,
            example.val_within_sse,
        )
    if split == "test":
        return (
            example.test_target,
            example.test_mask,
            example.test_pixel_weights,
            example.test_within_sse,
        )
    raise ValueError(split)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    examples: list[ExampleV2],
    device: torch.device,
    split: str,
) -> dict[str, object]:
    stratum_names = (
        "all",
        "same_frequency_context",
        "no_same_frequency_context",
        "auxiliary_only_context",
        "no_context_any_frequency",
    )
    states = {
        name: {
            "projected_sse": torch.zeros(NF, dtype=torch.float64),
            "pixel_counts": torch.zeros(NF, dtype=torch.float64),
            "regional_sse": torch.zeros(NF, dtype=torch.float64),
            "region_counts": torch.zeros(NF, dtype=torch.float64),
        }
        for name in stratum_names
    }
    model.eval()
    for example in examples:
        pred = model(example.x.to(device), example.edge_index.to(device)).clamp(0.0, 1.0)
        target, mask, weights, within_sse = target_mask_weight(example, split)
        target = target.to(device)
        mask = mask.to(device)
        weights = weights.to(device)
        within_sse = within_sse.to(device)
        context = example.train_mask.to(device)
        any_context = context.any(dim=1, keepdim=True)
        stratum_masks = {
            "all": mask,
            "same_frequency_context": mask & context,
            "no_same_frequency_context": mask & ~context,
            "auxiliary_only_context": mask & ~context & any_context,
            "no_context_any_frequency": mask & ~any_context,
        }
        pred_db = pred * 40.0 - 120.0
        target_db = target * 40.0 - 120.0
        for name, stratum_mask in stratum_masks.items():
            state = states[name]
            for fi in range(NF):
                active = stratum_mask[:, fi]
                if not active.any():
                    continue
                w = weights[active, fi].double()
                error = (pred_db[active, fi] - target_db[active, fi]).double()
                squared_error = error.square()
                projected_mean_sse = torch.sum(w * squared_error)
                pixel_variation_sse = 1600.0 * within_sse[active, fi].double().sum()
                state["projected_sse"][fi] += (
                    projected_mean_sse + pixel_variation_sse
                ).cpu()
                state["pixel_counts"][fi] += w.sum().cpu()
                state["regional_sse"][fi] += squared_error.sum().cpu()
                state["region_counts"][fi] += float(active.sum().cpu())

    def finalize(state: dict[str, torch.Tensor]) -> dict[str, object]:
        pixel_counts = state["pixel_counts"]
        region_counts = state["region_counts"]
        projected_per_freq = torch.sqrt(
            state["projected_sse"] / pixel_counts.clamp_min(1.0)
        )
        regional_per_freq = torch.sqrt(
            state["regional_sse"] / region_counts.clamp_min(1.0)
        )
        return {
            "projected_pixel_rmse_db": float(
                torch.sqrt(
                    state["projected_sse"].sum()
                    / pixel_counts.sum().clamp_min(1.0)
                )
            ),
            "projected_pixel_rmse_db_per_freq": projected_per_freq.tolist(),
            "regional_rmse_db": float(
                torch.sqrt(
                    state["regional_sse"].sum()
                    / region_counts.sum().clamp_min(1.0)
                )
            ),
            "regional_rmse_db_per_freq": regional_per_freq.tolist(),
            "pixel_counts_per_freq": pixel_counts.tolist(),
            "region_frequency_counts_per_freq": region_counts.tolist(),
        }

    finalized = {name: finalize(state) for name, state in states.items()}
    overall = finalized["all"]
    return {
        # Keep the legacy keys for checkpoint selection and existing aggregators.
        "rmse_db": overall["projected_pixel_rmse_db"],
        "rmse_db_per_freq": overall["projected_pixel_rmse_db_per_freq"],
        "counts_per_freq": overall["pixel_counts_per_freq"],
        "projected_pixel_rmse_db": overall["projected_pixel_rmse_db"],
        "projected_pixel_rmse_db_per_freq": overall[
            "projected_pixel_rmse_db_per_freq"
        ],
        "regional_rmse_db": overall["regional_rmse_db"],
        "regional_rmse_db_per_freq": overall["regional_rmse_db_per_freq"],
        "strata": finalized,
        "weighting": "exact_heldout_pixel_sse_after_superpixel_projection",
        "strata_definition": {
            "same_frequency_context": "test regions with a context anchor at the evaluated frequency",
            "auxiliary_only_context": "test regions with context only at other frequencies",
            "no_context_any_frequency": "test regions with no context anchor at any frequency",
        },
    }


@torch.no_grad()
def make_figure(
    model: nn.Module,
    example: ExampleV2,
    device: torch.device,
    out_path: Path,
    model_label: str,
    freq_idx: int,
) -> None:
    pred = model(example.x.to(device), example.edge_index.to(device)).clamp(0.0, 1.0).cpu().numpy()
    labels = example.labels
    padded_pred = np.full(pred.shape[0] + 1, np.nan, dtype=np.float32)
    padded_gt = np.full(pred.shape[0] + 1, np.nan, dtype=np.float32)
    padded_pred[1:] = pred[:, freq_idx] * 40.0 - 120.0
    padded_gt[1:] = example.test_target[:, freq_idx].numpy() * 40.0 - 120.0
    pred_map = padded_pred[labels]
    gt_map = padded_gt[labels]
    sparse_map = np.full(labels.shape, np.nan, dtype=np.float32)
    observed = example.sparse_maps[..., freq_idx] != 0
    sparse_map[observed] = example.sparse_maps[..., freq_idx][observed] * 40.0 - 120.0

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    ecmap = plt.get_cmap("magma").copy()
    ecmap.set_bad("white")
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4), constrained_layout=True)
    im = axes[0].imshow(sparse_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[0].set_title("Sparse target-environment measurements")
    axes[1].imshow(pred_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[1].set_title(model_label)
    axes[2].imshow(gt_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[2].set_title("Held-out regional target")
    error = np.abs(pred_map - gt_map)
    eim = axes[3].imshow(error, cmap=ecmap, vmin=0, vmax=8)
    axes[3].set_title("Absolute error")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(im, ax=axes[:3], fraction=0.022, pad=0.01, label="RSS / path gain (dB)")
    fig.colorbar(eim, ax=axes[3], fraction=0.05, pad=0.01, label="error (dB)")
    fig.suptitle(f"{example.name} | {FREQS_GHZ[freq_idx]:.2f} GHz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_model(args: argparse.Namespace, in_dim: int) -> tuple[nn.Module, str, str]:
    if args.model == "node_mlp":
        return (
            NodeMLP(in_dim, args.hidden_dim, NF, args.dropout),
            "Node MLP",
            "node_mlp_without_graph_edges",
        )
    if args.model == "graph_gat":
        per_head = args.hidden_dim // args.num_heads
        if per_head < 1 or per_head * args.num_heads != args.hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads")
        return (
            SuperpixelGAT(in_dim, per_head, NF, args.num_heads, args.dropout),
            "Graph-only GAT",
            "joint_output_graph_only_gat",
        )
    return (
        PureGroupedMFGAT(
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.dropout,
        ),
        "MFGAT",
        "pure_sub6_mmwave_local_plus_graph_mfgat",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["node_mlp", "graph_gat", "mfgat"], required=True)
    parser.add_argument("--cities", default="Davis,Woodland,Vacaville")
    parser.add_argument("--max-graphs", type=int, default=1160)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoothness-weight", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--target-val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--figure-freq-index", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cities = [city.strip() for city in args.cities.split(",") if city.strip()]
    examples = load_examples(
        args.graph_root,
        cities,
        args.max_graphs,
        args.seed,
        "no_idw",
        args.target_cache_root,
        args.target_val_fraction,
        False,
    )
    if not examples:
        raise RuntimeError(f"No graphs under {args.graph_root}")
    if examples[0].x.shape[1] != 2 * NF + 5:
        raise RuntimeError(f"Expected no-IDW input width {2 * NF + 5}, got {examples[0].x.shape[1]}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, model_label, architecture = build_model(args, examples[0].x.shape[1])
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    print(
        f"[v5 label-free] model={args.model} graphs={len(examples)} "
        f"nodes~{examples[0].x.shape[0]} features={examples[0].x.shape[1]} device={device}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        model.train()
        losses = []
        data_losses = []
        smooth_losses = []
        started = time.time()
        for x, target, mask, edge in graph_batches(examples, args.batch_size):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            edge = edge.to(device, non_blocking=True)
            pred = model(x, edge)
            data_loss = frequency_balanced_loss(pred, target, mask)
            smooth_loss = graph_smoothness(pred, edge) if args.model == "mfgat" else pred.sum() * 0.0
            loss = data_loss + args.smoothness_weight * smooth_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            data_losses.append(float(data_loss.detach().cpu()))
            smooth_losses.append(float(smooth_loss.detach().cpu()))
        scheduler.step()

        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "data_loss": float(np.mean(data_losses)),
            "smoothness_loss": float(np.mean(smooth_losses)),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            val = evaluate(model, examples, device, "val")
            record["val"] = val
            if val["rmse_db"] < best_val:
                best_val = float(val["rmse_db"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, args.out_dir / "best_model.pt")
                stale = 0
            else:
                stale += 1
        history.append(record)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if should_eval and (epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs):
            print(
                f"[epoch {epoch:04d}] loss={record['loss']:.6f} "
                f"val={val['rmse_db']:.3f}dB sec={record['seconds']:.1f}",
                flush=True,
            )
        if should_eval and args.patience > 0 and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was produced")
    model.load_state_dict(best_state)
    torch.save(best_state, args.out_dir / "best_model.pt")
    result = {
        "method": model_label,
        "protocol": "stage1_target_environment_transductive_label_free_v5",
        "architecture": architecture,
        "input": "per_frequency_sparse_node_values_and_masks_plus_geometry_no_idw",
        "training_supervision": "observed_sparse_node_frequency_entries_only",
        "validation_supervision": "disjoint_sparse_validation_pixels_only",
        "dense_target_usage": "final_evaluation_only",
        "target_schema": examples[0].target_schema,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "num_examples": len(examples),
        "best_epoch": best_epoch,
        "best_val_rmse_db": best_val,
        "val": evaluate(model, examples, device, "val"),
        "test": evaluate(model, examples, device, "test"),
        "freqs_ghz": list(FREQS_GHZ),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(
        model,
        examples[0],
        device,
        args.out_dir / "prediction_example.png",
        model_label,
        args.figure_freq_index,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
