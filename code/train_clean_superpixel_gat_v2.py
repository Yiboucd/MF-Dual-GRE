#!/usr/bin/env python3
"""Train leakage-free semi-supervised or grouped-inductive coarse graph models."""

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
import torch.nn.functional as F

from train_clean_superpixel_gat import FREQS_GHZ, NodeMLP, SuperpixelGAT
from graph_protocol_v3 import load_or_build_pixel_disjoint_targets


class ExampleV2:
    def __init__(
        self,
        path: Path,
        input_mode: str,
        target_cache_root: Path | None = None,
        target_seed: int = 2026,
        target_val_fraction: float = 0.20,
        exclude_training_queries_from_eval: bool = False,
    ):
        data = np.load(path, allow_pickle=False)
        schema = str(data["graph_schema"]) if "graph_schema" in data else "legacy"
        if not schema.startswith("clean_superpixel_graph_v2"):
            raise ValueError(f"Expected v2 graph, got {schema}: {path}")
        x = data["feature_matrix"].astype(np.float32)
        if input_mode == "no_idw":
            x = x[:, : -len(FREQS_GHZ)]
        self.path = path
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_dense = torch.tensor(data["matrix_gt"], dtype=torch.float32)
        self.train_target = torch.tensor(data["train_target_vectors"], dtype=torch.float32)
        self.query_target = torch.tensor(data["val_target_vectors"], dtype=torch.float32)
        self.val_target = torch.tensor(data["val_target_vectors"], dtype=torch.float32)
        self.test_target = self.y_dense
        self.train_mask = torch.tensor(data["train_mask"], dtype=torch.bool)
        self.query_mask = torch.tensor(data["val_mask"], dtype=torch.bool)
        self.val_mask = torch.tensor(data["val_mask"], dtype=torch.bool)
        self.test_mask = torch.tensor(data["test_mask"], dtype=torch.bool)
        self.val_pixel_weights = self.val_mask.float()
        self.test_pixel_weights = self.test_mask.float()
        self.val_within_sse = torch.zeros_like(self.val_target)
        self.test_within_sse = torch.zeros_like(self.test_target)
        self.target_schema = "legacy_node_disjoint"
        if target_cache_root is not None:
            heldout = load_or_build_pixel_disjoint_targets(
                path,
                data,
                target_cache_root,
                seed=target_seed,
                val_fraction=target_val_fraction,
                exclude_training_queries=exclude_training_queries_from_eval,
            )
            self.val_target = torch.tensor(heldout["val_target_vectors"], dtype=torch.float32)
            self.test_target = torch.tensor(heldout["test_target_vectors"], dtype=torch.float32)
            self.val_mask = torch.tensor(heldout["val_mask"], dtype=torch.bool)
            self.test_mask = torch.tensor(heldout["test_mask"], dtype=torch.bool)
            self.val_pixel_weights = torch.tensor(heldout["val_node_pixel_counts"], dtype=torch.float32)
            self.test_pixel_weights = torch.tensor(heldout["test_node_pixel_counts"], dtype=torch.float32)
            self.val_within_sse = torch.tensor(heldout["val_within_sse"], dtype=torch.float32)
            self.test_within_sse = torch.tensor(heldout["test_within_sse"], dtype=torch.float32)
            self.target_schema = str(heldout["target_schema"])
        self.edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        self.labels = data["labels"].astype(np.int32)
        self.sparse_maps = data["sparse_train_maps"].astype(np.float32)
        self.building = data["building_mask"].astype(bool)
        self.city = str(data["city"])
        self.xml_idx = int(data["xml_idx"])
        self.tx_idx = int(data["tx_idx"])
        self.name = path.stem


def load_examples(
    root: Path,
    cities: list[str],
    max_graphs: int,
    seed: int,
    input_mode: str,
    target_cache_root: Path | None = None,
    target_val_fraction: float = 0.20,
    exclude_training_queries_from_eval: bool = False,
) -> list[ExampleV2]:
    paths: list[Path] = []
    for city in cities:
        paths.extend(Path(path) for path in glob.glob(str(root / city / "**" / "*.npz"), recursive=True))
    paths = sorted(paths)
    random.Random(seed).shuffle(paths)
    if max_graphs > 0:
        paths = paths[:max_graphs]
    return [
        ExampleV2(
            path,
            input_mode,
            target_cache_root,
            seed,
            target_val_fraction,
            exclude_training_queries_from_eval,
        )
        for path in paths
    ]


def grouped_xml_split(
    examples: list[ExampleV2],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[ExampleV2], list[ExampleV2], list[ExampleV2], dict[str, list[str]]]:
    train_groups: set[tuple[str, int]] = set()
    val_groups: set[tuple[str, int]] = set()
    test_groups: set[tuple[str, int]] = set()
    for city in sorted({example.city for example in examples}):
        groups = sorted({(example.city, example.xml_idx) for example in examples if example.city == city})
        random.Random(seed + sum(ord(char) for char in city)).shuffle(groups)
        n = len(groups)
        n_train = max(1, int(round(train_ratio * n)))
        n_val = max(1, int(round(val_ratio * n)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train_groups.update(groups[:n_train])
        val_groups.update(groups[n_train : n_train + n_val])
        test_groups.update(groups[n_train + n_val :])
    train = [example for example in examples if (example.city, example.xml_idx) in train_groups]
    val = [example for example in examples if (example.city, example.xml_idx) in val_groups]
    test = [example for example in examples if (example.city, example.xml_idx) in test_groups]
    split = {
        "train": [f"{city}:xml_{xml_idx}" for city, xml_idx in sorted(train_groups)],
        "val": [f"{city}:xml_{xml_idx}" for city, xml_idx in sorted(val_groups)],
        "test": [f"{city}:xml_{xml_idx}" for city, xml_idx in sorted(test_groups)],
    }
    return train, val, test, split


def all_node_mask(example: ExampleV2) -> torch.Tensor:
    return example.train_mask | example.val_mask | example.test_mask


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    examples: list[ExampleV2],
    device: torch.device,
    target_kind: str,
    mask_kind: str,
) -> dict[str, object]:
    nf = len(FREQS_GHZ)
    sse = torch.zeros(nf, dtype=torch.float64)
    counts = torch.zeros(nf, dtype=torch.float64)
    graph_rmses: list[float] = []
    model.eval()
    for example in examples:
        pred = model(example.x.to(device), example.edge_index.to(device)).clamp(0.0, 1.0)
        if target_kind == "dense":
            target = example.y_dense.to(device)
        elif target_kind == "train_sparse":
            target = example.train_target.to(device)
        elif target_kind == "val_sparse":
            target = example.val_target.to(device)
        elif target_kind == "test_disjoint":
            target = example.test_target.to(device)
        else:
            raise ValueError(target_kind)
        if mask_kind == "all":
            mask = all_node_mask(example).to(device)
        else:
            mask = getattr(example, f"{mask_kind}_mask").to(device)
        pred_db = pred * 40.0 - 120.0
        target_db = target * 40.0 - 120.0
        if mask.any():
            graph_rmses.append(float(torch.sqrt(F.mse_loss(pred_db[mask], target_db[mask])).cpu()))
        for fi in range(nf):
            mf = mask[:, fi]
            if mf.any():
                sse[fi] += torch.sum((pred_db[mf, fi] - target_db[mf, fi]).double().square()).cpu()
                counts[fi] += int(mf.sum())
    per_freq = torch.sqrt(sse / counts.clamp_min(1.0))
    overall = torch.sqrt(sse.sum() / counts.sum().clamp_min(1.0))
    return {
        "rmse_db": float(overall),
        "rmse_db_per_freq": per_freq.tolist(),
        "mean_graph_rmse_db": float(np.mean(graph_rmses)) if graph_rmses else float("nan"),
        "counts_per_freq": counts.tolist(),
    }


@torch.no_grad()
def make_figure(
    model: torch.nn.Module,
    example: ExampleV2,
    device: torch.device,
    out_path: Path,
    freq_idx: int,
) -> None:
    pred = model(example.x.to(device), example.edge_index.to(device)).clamp(0.0, 1.0).cpu().numpy()
    labels = example.labels
    padded_pred = np.full(pred.shape[0] + 1, np.nan, dtype=np.float32)
    padded_gt = np.full(pred.shape[0] + 1, np.nan, dtype=np.float32)
    padded_pred[1:] = pred[:, freq_idx] * 40.0 - 120.0
    padded_gt[1:] = example.y_dense[:, freq_idx].numpy() * 40.0 - 120.0
    pred_map = padded_pred[labels]
    gt_map = padded_gt[labels]
    sparse_map = example.sparse_maps[..., freq_idx] * 40.0 - 120.0
    obs = np.load(example.path, allow_pickle=False)["obs_train_pixel_masks"][..., freq_idx]
    sparse_display = np.full(labels.shape, np.nan, dtype=np.float32)
    sparse_display[obs] = sparse_map[obs]
    error = np.abs(pred_map - gt_map)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    ecmap = plt.get_cmap("magma").copy()
    ecmap.set_bad("white")
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4), constrained_layout=True)
    im = axes[0].imshow(sparse_display, cmap=cmap, vmin=-120, vmax=-80)
    axes[0].set_title("Raw sparse observations")
    axes[1].imshow(pred_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[1].set_title("Coarse prediction")
    axes[2].imshow(gt_map, cmap=cmap, vmin=-120, vmax=-80)
    axes[2].set_title("Coarse ground truth")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cities", default="Davis,Woodland,Vacaville")
    parser.add_argument("--max-graphs", type=int, default=1160)
    parser.add_argument("--model", choices=["mlp", "gat"], required=True)
    parser.add_argument("--protocol", choices=["semi", "inductive"], default="semi")
    parser.add_argument("--input-mode", choices=["all", "no_idw"], default="no_idw")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--figure-freq-index", type=int, default=1)
    parser.add_argument(
        "--evaluation-target-protocol",
        "--semi-target-protocol",
        dest="semi_target_protocol",
        choices=["legacy_node_disjoint", "pixel_disjoint"],
        default="legacy_node_disjoint",
    )
    parser.add_argument("--target-cache-root", type=Path)
    parser.add_argument("--target-val-fraction", type=float, default=0.20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cities = [city.strip() for city in args.cities.split(",") if city.strip()]
    target_cache_root = None
    if args.semi_target_protocol == "pixel_disjoint":
        if args.target_cache_root is None:
            raise ValueError("--target-cache-root is required for pixel_disjoint targets")
        target_cache_root = args.target_cache_root
    examples = load_examples(
        args.graph_root,
        cities,
        args.max_graphs,
        args.seed,
        args.input_mode,
        target_cache_root,
        args.target_val_fraction,
    )
    if not examples:
        raise RuntimeError(f"No v2 graphs under {args.graph_root}")
    if args.protocol == "semi":
        train_examples = list(examples)
        val_examples = list(examples)
        test_examples = list(examples)
        split_groups: dict[str, list[str]] | None = None
        train_target_kind = "train_sparse"
        train_mask_kind = "train"
        val_target_kind = "val_sparse"
        val_mask_kind = "val"
    else:
        train_examples, val_examples, test_examples, split_groups = grouped_xml_split(
            examples, args.seed, args.train_ratio, args.val_ratio
        )
        train_target_kind = "dense"
        train_mask_kind = "all"
        val_target_kind = "test_disjoint" if args.semi_target_protocol == "pixel_disjoint" else "dense"
        val_mask_kind = "test"

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    in_dim = examples[0].x.shape[1]
    if args.model == "mlp":
        model = NodeMLP(in_dim, args.hidden_dim, len(FREQS_GHZ), args.dropout)
    else:
        model = SuperpixelGAT(in_dim, args.hidden_dim, len(FREQS_GHZ), args.num_heads, args.dropout)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[load v2] protocol={args.protocol} model={args.model} graphs={len(examples)} "
        f"train={len(train_examples)} val={len(val_examples)} test={len(test_examples)} "
        f"nodes~{examples[0].x.shape[0]} feat={in_dim} device={device}",
        flush=True,
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_examples)
        model.train()
        losses = []
        started = time.time()
        for example in train_examples:
            pred = model(example.x.to(device), example.edge_index.to(device))
            if train_target_kind == "dense":
                target = example.y_dense.to(device)
            else:
                target = example.train_target.to(device)
            mask = all_node_mask(example).to(device) if train_mask_kind == "all" else example.train_mask.to(device)
            loss = F.mse_loss(pred[mask], target[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val = evaluate(model, val_examples, device, val_target_kind, val_mask_kind)
        test_kind = "test_disjoint" if args.semi_target_protocol == "pixel_disjoint" else "dense"
        test = evaluate(model, test_examples, device, test_kind, "test")
        record = {
            "epoch": epoch,
            "train_loss_norm": float(np.mean(losses)),
            "val": val,
            "test": test,
            "seconds": time.time() - started,
        }
        history.append(record)
        if val["rmse_db"] < best_val:
            best_val = float(val["rmse_db"])
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, args.out_dir / "best_model.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            test_label = "heldout_test" if args.semi_target_protocol == "pixel_disjoint" else "dense_test"
            print(
                f"[epoch {epoch:04d}] train={record['train_loss_norm']:.6f} "
                f"val={val['rmse_db']:.3f}dB {test_label}={test['rmse_db']:.3f}dB "
                f"sec={record['seconds']:.1f}",
                flush=True,
            )
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.patience > 0 and stale >= args.patience:
            print(f"[early stop] epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was produced")
    model.load_state_dict(best_state)
    final_val = evaluate(model, val_examples, device, val_target_kind, val_mask_kind)
    test_kind = "test_disjoint" if args.semi_target_protocol == "pixel_disjoint" else "dense"
    final_test = evaluate(model, test_examples, device, test_kind, "test")
    result = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "graph_schema": "clean_superpixel_graph_v2_geometry_no_gt",
        "protocol": args.protocol,
        "target_schema": examples[0].target_schema,
        "num_examples": len(examples),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "num_test_examples": len(test_examples),
        "split_groups": split_groups,
        "best_epoch": best_epoch,
        "best_val_rmse_db": best_val,
        "val": final_val,
        "test": final_test,
        "freqs_ghz": FREQS_GHZ,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(model, test_examples[0], device, args.out_dir / "prediction_example.png", args.figure_freq_index)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
