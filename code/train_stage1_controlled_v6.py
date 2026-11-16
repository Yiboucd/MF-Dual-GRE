#!/usr/bin/env python3
"""Train Stage-1 models under controlled cross-frequency observation budgets."""

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

from build_clean_superpixel_graphs import FREQS_GHZ
from controlled_multifrequency_protocol_v6 import NF, ControlledExample, load_controlled_examples
from train_clean_superpixel_gat import GATLayer


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


class LocalGraphBranch(nn.Module):
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


class FrequencyAwareMFGAT(nn.Module):
    """Frequency-token attention followed by propagation-aware graph attention."""

    def __init__(self, width: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.measurement_encoder = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.frequency_embedding = nn.Parameter(torch.empty(1, NF, width))
        nn.init.normal_(self.frequency_embedding, std=0.02)
        self.frequency_norm1 = nn.LayerNorm(width)
        self.frequency_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.frequency_norm2 = nn.LayerNorm(width)
        self.frequency_ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.local_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.node_projection = nn.Sequential(
            nn.Linear(NF * width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.blocks = nn.ModuleList(
            [ResidualGATBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.graph_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, NF))
        self.gate_logits = nn.Parameter(torch.zeros(NF))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        values = x[:, :NF]
        masks = x[:, NF : 2 * NF]
        measurement = torch.stack([values, masks], dim=-1)
        tokens = self.measurement_encoder(measurement)
        geometry = self.geometry_encoder(x[:, GEOM])[:, None, :]
        tokens = tokens + geometry + self.frequency_embedding
        normalized = self.frequency_norm1(tokens)
        attended, _ = self.frequency_attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.dropout(attended)
        tokens = tokens + self.dropout(self.frequency_ffn(self.frequency_norm2(tokens)))
        local = torch.sigmoid(self.local_head(tokens).squeeze(-1))
        hidden = self.node_projection(tokens.reshape(tokens.shape[0], -1))
        for block in self.blocks:
            hidden = block(hidden, edge_index)
        graph = torch.sigmoid(self.graph_head(hidden))
        gate = torch.sigmoid(self.gate_logits)[None, :]
        return local + gate * (graph - local)


class NaiveJointMFGAT(nn.Module):
    """Concatenate all bands once, with no explicit frequency-token operation."""

    def __init__(self, width: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.branch = LocalGraphBranch(2 * NF + 5, width, NF, heads, layers, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.branch(x, edge_index)


class SingleFrequencyGAT(nn.Module):
    """One target-band GAT that cannot access auxiliary-frequency measurements."""

    def __init__(
        self,
        target_index: int,
        width: int,
        heads: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.target_index = int(target_index)
        self.branch = LocalGraphBranch(7, width, 1, heads, layers, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        fi = self.target_index
        features = torch.cat(
            [x[:, fi : fi + 1], x[:, NF + fi : NF + fi + 1], x[:, GEOM]], dim=1
        )
        target = self.branch(features, edge_index)
        output = x.new_zeros((x.shape[0], NF))
        output[:, fi : fi + 1] = target
        return output


class IndependentFrequencyGAT(nn.Module):
    """Seven disjoint single-frequency GATs evaluated in one process.

    Each branch has its own parameters and receives only one carrier's sparse
    values/indicators plus shared geometry. Joint optimization is merely a
    compute convenience because no parameter or feature crosses frequencies.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                LocalGraphBranch(7, width, 1, heads, layers, dropout)
                for _ in range(NF)
            ]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        predictions = []
        for frequency, branch in enumerate(self.branches):
            features = torch.cat(
                [
                    x[:, frequency : frequency + 1],
                    x[:, NF + frequency : NF + frequency + 1],
                    x[:, GEOM],
                ],
                dim=1,
            )
            predictions.append(branch(features, edge_index))
        return torch.cat(predictions, dim=1)


def graph_batches(examples: list[ControlledExample], batch_size: int):
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


def mask_training_anchors(
    x: torch.Tensor,
    train_mask: torch.Tensor,
    target_indices: list[int],
    drop_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide target-band anchors from both values and indicators before loss."""

    eligible = torch.zeros_like(train_mask)
    eligible[:, target_indices] = train_mask[:, target_indices]
    hide = eligible & (
        torch.rand_like(train_mask, dtype=torch.float32) < drop_probability
    )
    for frequency in target_indices:
        active = torch.nonzero(eligible[:, frequency], as_tuple=False).flatten()
        if active.numel() and not hide[:, frequency].any():
            selected = active[torch.randint(active.numel(), (1,), device=active.device)]
            hide[selected, frequency] = True
        if active.numel() > 1 and torch.all(hide[active, frequency]):
            selected = active[torch.randint(active.numel(), (1,), device=active.device)]
            hide[selected, frequency] = False

    masked_x = x.clone()
    masked_x[:, :NF].masked_fill_(hide, 0.0)
    masked_x[:, NF : 2 * NF].masked_fill_(hide, 0.0)
    return masked_x, hide


def selected_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_indices: list[int],
    loss_type: str,
) -> torch.Tensor:
    losses = []
    for fi in target_indices:
        active = mask[:, fi]
        if active.any():
            if loss_type == "mse":
                losses.append(F.mse_loss(pred[active, fi], target[active, fi]))
            else:
                losses.append(
                    F.smooth_l1_loss(
                        pred[active, fi], target[active, fi], beta=0.025
                    )
                )
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def graph_smoothness(
    pred: torch.Tensor,
    edge_index: torch.Tensor,
    target_indices: list[int],
) -> torch.Tensor:
    src, dst = edge_index
    active = src != dst
    if not active.any():
        return pred.sum() * 0.0
    return (pred[src[active]][:, target_indices] - pred[dst[active]][:, target_indices]).square().mean()


def split_target(example: ControlledExample, split: str):
    if split == "val":
        return (
            example.val_target,
            example.val_mask,
            example.val_pixel_weights,
            example.val_within_sse,
            example.val_nonfloor_target,
            example.val_nonfloor_mask,
            example.val_nonfloor_pixel_weights,
            example.val_nonfloor_within_sse,
        )
    if split == "test":
        return (
            example.test_target,
            example.test_mask,
            example.test_pixel_weights,
            example.test_within_sse,
            example.test_nonfloor_target,
            example.test_nonfloor_mask,
            example.test_nonfloor_pixel_weights,
            example.test_nonfloor_within_sse,
        )
    raise ValueError(split)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    examples: list[ControlledExample],
    device: torch.device,
    split: str,
    target_indices: list[int],
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
            "projected_sse": np.zeros(NF, dtype=np.float64),
            "pixel_counts": np.zeros(NF, dtype=np.float64),
            "regional_sse": np.zeros(NF, dtype=np.float64),
            "region_counts": np.zeros(NF, dtype=np.float64),
            "nonfloor_projected_sse": np.zeros(NF, dtype=np.float64),
            "nonfloor_pixel_counts": np.zeros(NF, dtype=np.float64),
            "path_tp": np.zeros(NF, dtype=np.float64),
            "path_tn": np.zeros(NF, dtype=np.float64),
            "path_fp": np.zeros(NF, dtype=np.float64),
            "path_fn": np.zeros(NF, dtype=np.float64),
        }
        for name in stratum_names
    }
    model.eval()
    for example in examples:
        pred = model(example.x.to(device), example.edge_index.to(device)).clamp(0.0, 1.0)
        (
            target,
            mask,
            weights,
            within_sse,
            nonfloor_target,
            nonfloor_mask,
            nonfloor_weights,
            nonfloor_within_sse,
        ) = split_target(example, split)
        target = target.to(device)
        mask = mask.to(device)
        weights = weights.to(device)
        within_sse = within_sse.to(device)
        nonfloor_target = nonfloor_target.to(device)
        nonfloor_mask = nonfloor_mask.to(device)
        nonfloor_weights = nonfloor_weights.to(device)
        nonfloor_within_sse = nonfloor_within_sse.to(device)
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
            for fi in target_indices:
                active = stratum_mask[:, fi]
                if not active.any():
                    continue
                w = weights[active, fi].double()
                error = (pred_db[active, fi] - target_db[active, fi]).double()
                squared_error = error.square()
                projected_mean_sse = torch.sum(w * squared_error)
                state["projected_sse"][fi] += float(
                    (
                        projected_mean_sse
                        + 1600.0 * within_sse[active, fi].double().sum()
                    ).cpu()
                )
                state["pixel_counts"][fi] += float(w.sum().cpu())
                state["regional_sse"][fi] += float(squared_error.sum().cpu())
                state["region_counts"][fi] += float(active.sum().cpu())

                active_nonfloor = active & nonfloor_mask[:, fi]
                if active_nonfloor.any():
                    nonfloor_w = nonfloor_weights[active_nonfloor, fi].double()
                    nonfloor_error = (
                        pred_db[active_nonfloor, fi]
                        - (nonfloor_target[active_nonfloor, fi] * 40.0 - 120.0)
                    ).double()
                    state["nonfloor_projected_sse"][fi] += float(
                        (
                            torch.sum(nonfloor_w * nonfloor_error.square())
                            + 1600.0
                            * nonfloor_within_sse[active_nonfloor, fi]
                            .double()
                            .sum()
                        ).cpu()
                    )
                    state["nonfloor_pixel_counts"][fi] += float(
                        nonfloor_w.sum().cpu()
                    )

                all_w = weights[active, fi].double()
                path_w = nonfloor_weights[active, fi].double()
                floor_w = (all_w - path_w).clamp_min(0.0)
                predicted_path = pred_db[active, fi] > -119.5
                state["path_tp"][fi] += float(path_w[predicted_path].sum().cpu())
                state["path_fp"][fi] += float(floor_w[predicted_path].sum().cpu())
                state["path_fn"][fi] += float(path_w[~predicted_path].sum().cpu())
                state["path_tn"][fi] += float(floor_w[~predicted_path].sum().cpu())

    def finalize(state: dict[str, np.ndarray]) -> dict[str, object]:
        projected_total = float(state["projected_sse"][target_indices].sum())
        pixel_total = float(state["pixel_counts"][target_indices].sum())
        regional_total = float(state["regional_sse"][target_indices].sum())
        region_total = float(state["region_counts"][target_indices].sum())
        nonfloor_sse_total = float(
            state["nonfloor_projected_sse"][target_indices].sum()
        )
        nonfloor_count_total = float(
            state["nonfloor_pixel_counts"][target_indices].sum()
        )

        def path_scores(indices: list[int]) -> dict[str, float]:
            tp = float(state["path_tp"][indices].sum())
            tn = float(state["path_tn"][indices].sum())
            fp = float(state["path_fp"][indices].sum())
            fn = float(state["path_fn"][indices].sum())
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            specificity = tn / max(tn + fp, 1.0)
            return {
                "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
                "balanced_accuracy": 0.5 * (recall + specificity),
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "true_path_pixels": tp + fn,
                "true_floor_pixels": tn + fp,
            }

        return {
            "projected_pixel_rmse_db": math.sqrt(
                projected_total / max(pixel_total, 1.0)
            ),
            "projected_pixel_rmse_db_per_target_frequency": {
                f"{FREQS_GHZ[fi]:g}": math.sqrt(
                    state["projected_sse"][fi]
                    / max(state["pixel_counts"][fi], 1.0)
                )
                for fi in target_indices
            },
            "regional_rmse_db": math.sqrt(
                regional_total / max(region_total, 1.0)
            ),
            "regional_rmse_db_per_target_frequency": {
                f"{FREQS_GHZ[fi]:g}": math.sqrt(
                    state["regional_sse"][fi]
                    / max(state["region_counts"][fi], 1.0)
                )
                for fi in target_indices
            },
            "nonfloor_projected_pixel_rmse_db": math.sqrt(
                nonfloor_sse_total / max(nonfloor_count_total, 1.0)
            ),
            "nonfloor_projected_pixel_rmse_db_per_target_frequency": {
                f"{FREQS_GHZ[fi]:g}": math.sqrt(
                    state["nonfloor_projected_sse"][fi]
                    / max(state["nonfloor_pixel_counts"][fi], 1.0)
                )
                for fi in target_indices
            },
            "path_existence": path_scores(target_indices),
            "path_existence_per_target_frequency": {
                f"{FREQS_GHZ[fi]:g}": path_scores([fi]) for fi in target_indices
            },
            "pixel_counts_per_target_frequency": {
                f"{FREQS_GHZ[fi]:g}": float(state["pixel_counts"][fi])
                for fi in target_indices
            },
            "region_frequency_counts_per_target_frequency": {
                f"{FREQS_GHZ[fi]:g}": float(state["region_counts"][fi])
                for fi in target_indices
            },
            "nonfloor_pixel_counts_per_target_frequency": {
                f"{FREQS_GHZ[fi]:g}": float(state["nonfloor_pixel_counts"][fi])
                for fi in target_indices
            },
        }

    finalized = {name: finalize(state) for name, state in states.items()}
    overall = finalized["all"]
    return {
        # Keep rmse_db as the exact projected-pixel metric for old selectors.
        "rmse_db": overall["projected_pixel_rmse_db"],
        "projected_pixel_rmse_db": overall["projected_pixel_rmse_db"],
        "projected_pixel_rmse_db_per_target_frequency": overall[
            "projected_pixel_rmse_db_per_target_frequency"
        ],
        "regional_rmse_db": overall["regional_rmse_db"],
        "regional_rmse_db_per_target_frequency": overall[
            "regional_rmse_db_per_target_frequency"
        ],
        "nonfloor_projected_pixel_rmse_db": overall[
            "nonfloor_projected_pixel_rmse_db"
        ],
        "nonfloor_projected_pixel_rmse_db_per_target_frequency": overall[
            "nonfloor_projected_pixel_rmse_db_per_target_frequency"
        ],
        "path_existence": overall["path_existence"],
        "path_existence_per_target_frequency": overall[
            "path_existence_per_target_frequency"
        ],
        "counts_per_target_frequency": overall[
            "pixel_counts_per_target_frequency"
        ],
        "strata": finalized,
        "weighting": "exact_heldout_pixel_sse_after_superpixel_projection",
        "path_existence_definition": {
            "ground_truth": "normalized RSS > 0, i.e. above the -120 dB no-path floor",
            "prediction_threshold_db": -119.5,
        },
        "strata_definition": {
            "same_frequency_context": "test regions with a context anchor at the evaluated frequency",
            "auxiliary_only_context": "test regions with context only at other frequencies",
            "no_context_any_frequency": "test regions with no context anchor at any frequency",
        },
    }


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    example: ControlledExample,
    device: torch.device,
    repetitions: int = 30,
) -> float:
    model.eval()
    x = example.x.to(device)
    edge = example.edge_index.to(device)
    for _ in range(5):
        model(x, edge)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repetitions):
        model(x, edge)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - started) / repetitions


@torch.no_grad()
def make_figure(
    model: nn.Module,
    example: ControlledExample,
    device: torch.device,
    out_path: Path,
    target_index: int,
    label: str,
) -> None:
    pred = model(example.x.to(device), example.edge_index.to(device)).clamp(0.0, 1.0).cpu().numpy()
    observed = example.x[:, NF + target_index].numpy() > 0.5
    sparse = example.x[:, target_index].numpy() * 40.0 - 120.0
    target = example.test_target[:, target_index].numpy() * 40.0 - 120.0
    padded_sparse = np.full(pred.shape[0] + 1, np.nan, dtype=np.float32)
    padded_pred = np.full(pred.shape[0] + 1, np.nan, dtype=np.float32)
    padded_target = np.full(pred.shape[0] + 1, np.nan, dtype=np.float32)
    padded_sparse[np.flatnonzero(observed) + 1] = sparse[observed]
    padded_pred[1:] = pred[:, target_index] * 40.0 - 120.0
    padded_target[1:] = target
    sparse_map = padded_sparse[example.labels]
    pred_map = padded_pred[example.labels]
    target_map = padded_target[example.labels]
    error = np.abs(pred_map - target_map)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    ecmap = plt.get_cmap("magma").copy()
    ecmap.set_bad("white")
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4), constrained_layout=True)
    im = axes[0].imshow(sparse_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[0].set_title("Observed target-band regions")
    axes[1].imshow(pred_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[1].set_title(label)
    axes[2].imshow(target_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[2].set_title("Held-out regional target")
    eim = axes[3].imshow(error, cmap=ecmap, vmin=0, vmax=8)
    axes[3].set_title("Absolute error")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(im, ax=axes[:3], fraction=0.022, pad=0.01, label="RSS / path gain (dB)")
    fig.colorbar(eim, ax=axes[3], fraction=0.05, pad=0.01, label="error (dB)")
    fig.suptitle(f"{example.name} | target {FREQS_GHZ[target_index]:.2f} GHz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_model(args: argparse.Namespace) -> tuple[nn.Module, str, str]:
    if args.model == "mfgat":
        if args.mfgat_architecture == "frequency_joint_anchor":
            # Import lazily because train_stage1_mfgat_v7 reuses the baseline
            # blocks in this module.
            from train_stage1_mfgat_v7 import (
                AnchorPreservingMFGAT,
                JointAnchorAwareMFGAT,
            )

            core = JointAnchorAwareMFGAT(
                args.hidden_dim,
                args.num_heads,
                args.num_layers,
                args.dropout,
                args.anchor_neighbors,
            )
            return (
                AnchorPreservingMFGAT(core, args.consistency_alpha),
                "MFGAT",
                "joint_cross_frequency_sparse_anchor_attention_plus_spatial_gat",
            )
        return (
            FrequencyAwareMFGAT(args.hidden_dim, args.num_heads, args.num_layers, args.dropout),
            "MFGAT",
            "frequency_token_attention_plus_spatial_gat_joint_seven_output",
        )
    if args.model == "naive_mfgat":
        return (
            NaiveJointMFGAT(args.hidden_dim, args.num_heads, args.num_layers, args.dropout),
            "Naive joint GAT",
            "concatenated_multifrequency_input_joint_seven_output_gat",
        )
    if args.model == "sf_gat":
        if len(args.target_indices) != 1:
            raise ValueError("sf_gat requires exactly one target frequency")
        return (
            SingleFrequencyGAT(
                args.target_indices[0],
                args.hidden_dim,
                args.num_heads,
                args.num_layers,
                args.dropout,
            ),
            "Single-frequency GAT",
            "one_target_band_input_one_target_output_gat",
        )
    if args.model == "sf_gat_all":
        return (
            IndependentFrequencyGAT(
                args.hidden_dim,
                args.num_heads,
                args.num_layers,
                args.dropout,
            ),
            "Independent single-frequency GATs",
            "seven_disjoint_single_input_single_output_gats",
        )
    raise ValueError(args.model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["mfgat", "naive_mfgat", "sf_gat", "sf_gat_all"],
        required=True,
    )
    parser.add_argument(
        "--mfgat-architecture",
        choices=["frequency_joint_anchor", "frequency_token"],
        default="frequency_joint_anchor",
    )
    parser.add_argument("--rates", type=float, nargs=NF, required=True)
    parser.add_argument("--target-indices", type=int, nargs="+", required=True)
    parser.add_argument("--mask-mode", choices=["independent", "aligned"], required=True)
    parser.add_argument("--auxiliary-mode", choices=["true", "shuffled"], required=True)
    parser.add_argument("--cities", default="Davis,Woodland,Vacaville")
    parser.add_argument("--max-graphs", type=int, default=1160)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--anchor-dropout", type=float, default=0.50)
    parser.add_argument("--anchor-neighbors", type=int, default=4)
    parser.add_argument("--consistency-alpha", type=float, default=0.25)
    parser.add_argument("--loss-type", choices=["huber", "mse"], default="mse")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoothness-weight", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--fixed-val-rate", type=float, default=0.2)
    parser.add_argument("--fixed-test-rate", type=float, default=20.0)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=5)
    args = parser.parse_args()

    if any(index < 0 or index >= NF for index in args.target_indices):
        raise ValueError(args.target_indices)
    if not 0.0 < args.anchor_dropout < 1.0:
        raise ValueError("anchor_dropout must be in (0, 1)")
    if args.anchor_neighbors < 1:
        raise ValueError("anchor_neighbors must be positive")
    if not 0.0 <= args.consistency_alpha <= 1.0:
        raise ValueError("consistency_alpha must be in [0, 1]")
    args.target_indices = sorted(set(args.target_indices))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cities = [city.strip() for city in args.cities.split(",") if city.strip()]
    examples = load_controlled_examples(
        args.graph_root,
        cities,
        args.max_graphs,
        args.rates,
        args.target_indices[0],
        args.mask_mode,
        args.auxiliary_mode,
        args.mask_seed,
        args.selection_seed,
        args.fixed_val_rate,
        args.fixed_test_rate,
        args.shuffle_offset,
    )
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    model, model_label, architecture = build_model(args)
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    print(
        f"[controlled v6] model={args.model} graphs={len(examples)} rates={args.rates} "
        f"target={args.target_indices} mask={args.mask_mode} aux={args.auxiliary_mode} "
        f"anchor_dropout={args.anchor_dropout} device={device}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        model.train()
        losses = []
        data_losses = []
        smooth_losses = []
        hidden_fractions = []
        started = time.time()
        for x, target, mask, edge in graph_batches(examples, args.batch_size):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            edge = edge.to(device, non_blocking=True)
            masked_x, reconstruction_mask = mask_training_anchors(
                x, mask, args.target_indices, args.anchor_dropout
            )
            pred = model(masked_x, edge)
            data_loss = selected_loss(
                pred,
                target,
                reconstruction_mask,
                args.target_indices,
                args.loss_type,
            )
            smooth_loss = graph_smoothness(pred, edge, args.target_indices)
            loss = data_loss + args.smoothness_weight * smooth_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            data_losses.append(float(data_loss.detach().cpu()))
            smooth_losses.append(float(smooth_loss.detach().cpu()))
            hidden_fractions.append(
                float(reconstruction_mask.sum().cpu())
                / max(float(mask[:, args.target_indices].sum().cpu()), 1.0)
            )
        scheduler.step()
        record: dict[str, object] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "data_loss": float(np.mean(data_losses)),
            "smoothness_loss": float(np.mean(smooth_losses)),
            "hidden_anchor_fraction": float(np.mean(hidden_fractions)),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            val = evaluate(model, examples, device, "val", args.target_indices)
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
    inference_ms = benchmark_inference(model, examples[0], device)
    result = {
        "method": model_label,
        "protocol": "target_environment_cross_frequency_budget_control_masked_anchor_v7",
        "architecture": architecture,
        "dense_target_usage": "final_evaluation_only",
        "training_supervision": "randomly_hidden_target_frequency_anchors_only",
        "input_target_separation": (
            "each supervised target anchor is removed from both its value and indicator"
        ),
        "target_indices": args.target_indices,
        "target_frequencies_ghz": [FREQS_GHZ[index] for index in args.target_indices],
        "rates_percent": args.rates,
        "total_rate_budget_percent": float(sum(args.rates)),
        "mask_mode": args.mask_mode,
        "auxiliary_mode": args.auxiliary_mode,
        "mask_seed": args.mask_seed,
        "selection_seed": args.selection_seed,
        "model_seed": args.seed,
        "num_examples": len(examples),
        "mean_num_nodes": float(np.mean([example.metadata["num_nodes"] for example in examples])),
        "mean_num_directed_edges": float(
            np.mean([example.metadata["num_directed_edges"] for example in examples])
        ),
        "mean_context_pixel_counts": np.mean(
            [example.metadata["context_pixel_counts"] for example in examples], axis=0
        ).tolist(),
        "mean_observed_node_counts": np.mean(
            [example.metadata["observed_node_counts"] for example in examples], axis=0
        ).tolist(),
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "inference_ms_per_graph": inference_ms,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
        ),
        "best_epoch": best_epoch,
        "best_val_rmse_db": best_val,
        "val": evaluate(model, examples, device, "val", args.target_indices),
        "test": evaluate(model, examples, device, "test", args.target_indices),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(
        model,
        examples[0],
        device,
        args.out_dir / "prediction_example.png",
        args.target_indices[0],
        model_label,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
