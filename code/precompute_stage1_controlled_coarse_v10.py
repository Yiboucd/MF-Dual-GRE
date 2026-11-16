#!/usr/bin/env python3
"""Build a dense Stage-2 cache from a corrected masked-anchor Stage-1 run.

The cache reproduces the exact controlled pixel observations recorded by the
Stage-1 run. It never reads the legacy sparse maps baked into the graph file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from build_clean_superpixel_graphs import FREQS_GHZ
from build_clean_superpixel_graphs_v2 import masked_region_means, stable_seed
from controlled_multifrequency_protocol_v10 import (
    GEOM,
    NF,
    load_manifest_maps,
    make_context_masks,
    make_fixed_test_mask,
    select_paths,
    shuffled_pairs,
)
from eval_stage1_controlled_pixel_baselines_v10 import query_idw
from train_stage1_controlled_v10 import build_model


EXPECTED_PROTOCOLS = {
    "budget_internal_anchor_validation_transductive_v10",
    "budget_internal_fixed_pixel_holdout_transductive_v11",
    "budget_internal_fixed_pixel_holdout_xml_inductive_v11",
}

FIXED_CACHE_PROTOCOL = "budget_internal_stage1_projection_fixed_holdout_v11"
INDUCTIVE_FIXED_CACHE_PROTOCOL = (
    "budget_internal_stage1_projection_xml_inductive_fixed_holdout_v11"
)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(value: np.ndarray | str) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return str(value.item())
    return str(value)


def nodes_to_pixel(node_values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if node_values.ndim != 2:
        raise ValueError(f"Expected NxF node values, got {node_values.shape}")
    padded = np.zeros((node_values.shape[0] + 1, node_values.shape[1]), dtype=np.float32)
    padded[1:] = node_values.astype(np.float32)
    return padded[labels].transpose(2, 0, 1).astype(np.float32)


def native_pixel_idw_prior(
    sparse_maps: np.ndarray,
    context_masks: np.ndarray,
    free: np.ndarray,
    neighbors: int,
    power: float,
) -> np.ndarray:
    """Build a seven-band IDW prior without graph or superpixel prediction."""

    query_coordinates = np.argwhere(free)
    coarse = np.zeros_like(sparse_maps, dtype=np.float32)
    for frequency in range(sparse_maps.shape[0]):
        observed = context_masks[..., frequency] & free
        observed_coordinates = np.argwhere(observed)
        observed_values = sparse_maps[frequency][observed].astype(np.float64)
        if observed_values.size == 0:
            raise RuntimeError(f"IDW cache has no observations for band {frequency}")
        prediction = query_idw(
            observed_coordinates,
            observed_values,
            query_coordinates,
            neighbors,
            power,
        )
        band = np.zeros(free.shape, dtype=np.float32)
        band[free] = np.clip(prediction, 0.0, 1.0).astype(np.float32)
        band[observed] = sparse_maps[frequency][observed]
        coarse[frequency] = band
    return coarse


def model_namespace(saved: dict[str, object]) -> Namespace:
    return Namespace(
        model=str(saved["model"]),
        mfgat_architecture=str(saved.get("mfgat_architecture", "frequency_joint_anchor")),
        target_indices=[int(value) for value in saved["target_indices"]],
        hidden_dim=int(saved["hidden_dim"]),
        naive_hidden_dim=int(saved.get("naive_hidden_dim", 248)),
        num_heads=int(saved["num_heads"]),
        num_layers=int(saved["num_layers"]),
        dropout=float(saved["dropout"]),
        anchor_neighbors=int(saved.get("anchor_neighbors", 4)),
        consistency_alpha=float(saved.get("consistency_alpha", 0.25)),
        radiogat_source_policy=str(
            saved.get("radiogat_source_policy", "target_nearest")
        ),
        radiogat_depth_numerator=float(
            saved.get("radiogat_depth_numerator", 0.1)
        ),
    )


def output_path(out_root: Path, graph_path: Path, city: str) -> Path:
    return out_root / city / f"{graph_path.stem}_coarse.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--max-graphs", type=int, default=1160)
    parser.add_argument("--mask-seed-override", type=int)
    parser.add_argument("--prior-method", choices=["model", "idw"], default="model")
    parser.add_argument("--idw-neighbors", type=int, default=32)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    metrics_path = args.run_dir / "metrics.json"
    checkpoint_path = args.run_dir / "best_model.pt"
    metrics = read_json(metrics_path)
    if metrics.get("protocol") not in EXPECTED_PROTOCOLS:
        raise ValueError(f"Wrong Stage-1 protocol in {metrics_path}: {metrics.get('protocol')}")
    if metrics.get("training_supervision") != "randomly_hidden_target_frequency_anchors_only":
        raise ValueError(f"Stage-1 run did not use masked-anchor supervision: {metrics_path}")
    # The post-repair Stage-1 trainer (aligned-query repair, 2026-08-13) appends a
    # clause describing the aligned-query policy; accept both phrasings.
    if not str(metrics.get("input_target_separation", "")).startswith(
        "each supervised target anchor is removed from both its value and indicator"
    ):
        raise ValueError(f"Stage-1 input/target separation is not verified: {metrics_path}")
    if int(metrics.get("num_examples", -1)) != int(args.max_graphs):
        raise ValueError(
            f"Expected {args.max_graphs} Stage-1 examples, got {metrics.get('num_examples')}"
        )

    source_split_mode = str(
        metrics.get("split_mode", metrics.get("args", {}).get("split_mode", ""))
    )
    if source_split_mode not in {"transductive", "xml_inductive"}:
        raise ValueError(f"Unsupported Stage-1 split mode: {source_split_mode}")
    expected_source_protocol = (
        "budget_internal_fixed_pixel_holdout_xml_inductive_v11"
        if source_split_mode == "xml_inductive"
        else "budget_internal_fixed_pixel_holdout_transductive_v11"
    )
    if float(metrics.get("fixed_test_rate_percent", 0.0)) > 0 and (
        metrics.get("protocol") != expected_source_protocol
    ):
        raise ValueError(
            "Stage-1 protocol and split mode disagree: "
            f"{metrics.get('protocol')} vs {source_split_mode}"
        )
    source_split_groups = metrics.get("split_groups", {})
    if source_split_mode == "xml_inductive":
        train_groups = set(source_split_groups.get("train", []))
        val_groups = set(source_split_groups.get("val", []))
        test_groups = set(source_split_groups.get("test", []))
        if not train_groups or not val_groups or not test_groups:
            raise ValueError("XML-inductive Stage-1 split groups are incomplete")
        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise ValueError("XML-inductive Stage-1 split groups overlap")

    saved = dict(metrics["args"])
    run_model_key = str(saved.get("model"))
    if args.prior_method == "model" and run_model_key not in {"mfgat", "sf_gat_all"}:
        raise ValueError(
            "Formal Stage-2 cache requires MFGAT or the seven independent "
            "single-frequency GAT control"
        )
    if args.prior_method == "idw" and run_model_key != "mfgat":
        raise ValueError("The IDW cache protocol must be anchored to the formal MFGAT run specification")
    source_model_key = run_model_key if args.prior_method == "model" else "idw"
    target_indices = [int(value) for value in saved["target_indices"]]
    if target_indices != list(range(NF)):
        raise ValueError(f"Stage-2 cache requires seven output bands, got {target_indices}")
    rates = [float(value) for value in saved["rates"]]
    graph_root = Path(str(saved["graph_root"]))
    cities = [item.strip() for item in str(saved["cities"]).split(",") if item.strip()]
    mask_mode = str(saved["mask_mode"])
    auxiliary_mode = str(saved["auxiliary_mode"])
    training_mask_seed = int(saved["mask_seed"])
    mask_seed = (
        int(args.mask_seed_override)
        if args.mask_seed_override is not None
        else training_mask_seed
    )
    selection_seed = int(saved["selection_seed"])
    shuffle_offset = int(saved.get("shuffle_offset", 1))
    val_anchor_fraction = float(saved["val_anchor_fraction"])
    sampling_mode = str(saved.get("sampling_mode", "random"))
    measurement_noise_db = float(saved.get("measurement_noise_db", 0.0))
    fixed_test_rate_percent = float(saved.get("fixed_test_rate_percent", 0.0))
    fixed_test_seed = int(saved.get("fixed_test_seed", 9001))

    paths = select_paths(graph_root, cities, args.max_graphs, selection_seed)
    if len(paths) != args.max_graphs:
        raise RuntimeError(f"Expected exactly {args.max_graphs} graphs, found {len(paths)}")
    if auxiliary_mode == "shuffled":
        auxiliary_pool = select_paths(graph_root, cities, 0, selection_seed)
        pairs = shuffled_pairs(paths, auxiliary_pool, shuffle_offset)
    elif auxiliary_mode == "true":
        pairs = {path: path for path in paths}
    else:
        raise ValueError(auxiliary_mode)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    model = None
    if args.prior_method == "model":
        model, model_label, architecture = build_model(model_namespace(saved))
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model = model.to(device).eval()
    else:
        model_label = "2D-IDW"
        architecture = (
            f"native_pixel_{args.idw_neighbors}_neighbor_"
            f"inverse_distance_weighting_power_{args.idw_power:g}"
        )

    index: list[dict[str, object]] = []
    with torch.no_grad():
        for number, graph_path in enumerate(paths, start=1):
            with np.load(graph_path, allow_pickle=False) as data:
                labels = data["labels"].astype(np.int32)
                building = data["building_mask"].astype(bool)
                edge_index = data["edge_index"].astype(np.int64)
                geometry_nodes = data["feature_matrix"][:, GEOM].astype(np.float32)
                city = scalar_text(data["city"])
                xml_idx = int(np.asarray(data["xml_idx"]).item())
                tx_idx = int(np.asarray(data["tx_idx"]).item())

            manifest_path, manifest, target_maps = load_manifest_maps(graph_path)
            auxiliary_maps = target_maps
            auxiliary_manifest_path = manifest_path
            if auxiliary_mode == "shuffled":
                auxiliary_manifest_path, _, auxiliary_maps = load_manifest_maps(pairs[graph_path])

            free = ~building
            sample_key = f"{city}|xml={xml_idx}|tx={tx_idx}"
            fixed_test_mask = (
                make_fixed_test_mask(
                    free,
                    fixed_test_seed,
                    sample_key,
                    fixed_test_rate_percent,
                )
                if fixed_test_rate_percent > 0
                else None
            )
            context_masks = make_context_masks(
                free,
                rates,
                target_indices[0],
                mask_mode,
                mask_seed,
                sample_key,
                sampling_mode,
                fixed_test_mask,
            )

            num_nodes = int(labels.max())
            sparse_nodes = np.zeros((num_nodes, NF), dtype=np.float32)
            observed_nodes = np.zeros((num_nodes, NF), dtype=np.float32)
            sparse_maps = np.zeros((NF, *building.shape), dtype=np.float32)
            for frequency in range(NF):
                input_map = target_maps[..., frequency]
                if auxiliary_mode == "shuffled" and frequency != target_indices[0]:
                    input_map = auxiliary_maps[..., frequency]
                measurement_map = input_map
                if measurement_noise_db > 0:
                    measurement_map = input_map.copy()
                    active = context_masks[..., frequency]
                    rng = np.random.default_rng(
                        stable_seed(
                            mask_seed,
                            sample_key,
                            frequency,
                            sampling_mode,
                            measurement_noise_db,
                            "measurement-noise-v10",
                        )
                    )
                    measurement_map[active] += rng.normal(
                        0.0,
                        measurement_noise_db / 40.0,
                        int(active.sum()),
                    )
                values, counts = masked_region_means(
                    measurement_map,
                    context_masks[..., frequency],
                    labels,
                    num_nodes,
                )
                sparse_nodes[:, frequency] = values
                observed_nodes[:, frequency] = counts > 0
                sparse_maps[frequency][context_masks[..., frequency]] = measurement_map[
                    context_masks[..., frequency]
                ]

            if args.prior_method == "model":
                assert model is not None
                x = np.concatenate([sparse_nodes, observed_nodes, geometry_nodes], axis=1)
                prediction = model(
                    torch.tensor(x, dtype=torch.float32, device=device),
                    torch.tensor(edge_index, dtype=torch.long, device=device),
                ).clamp(0.0, 1.0)
                node_prediction = prediction.cpu().numpy()
                coarse = nodes_to_pixel(node_prediction, labels)
            else:
                coarse = native_pixel_idw_prior(
                    sparse_maps,
                    context_masks,
                    free,
                    args.idw_neighbors,
                    args.idw_power,
                )

            height, width = building.shape
            yy, xx = np.meshgrid(
                np.linspace(0.0, 1.0, height, dtype=np.float32),
                np.linspace(0.0, 1.0, width, dtype=np.float32),
                indexing="ij",
            )
            tx_col = width / 2.0 + float(manifest["tx"][0]) / 4.0
            tx_row = height / 2.0 + float(manifest["tx"][1]) / 4.0
            py, px = np.meshgrid(
                np.arange(height, dtype=np.float32),
                np.arange(width, dtype=np.float32),
                indexing="ij",
            )
            tx_distance = np.sqrt((px - tx_col) ** 2 + (py - tx_row) ** 2)
            tx_distance /= np.sqrt(float(height * height + width * width))
            building_context_and_depth = nodes_to_pixel(geometry_nodes[:, 3:5], labels)
            side = np.concatenate(
                [
                    building.astype(np.float32)[None],
                    xx[None],
                    yy[None],
                    tx_distance.astype(np.float32)[None],
                    building_context_and_depth,
                ],
                axis=0,
            ).astype(np.float32)
            evaluation_mask = (
                np.broadcast_to(
                    free[..., None] & fixed_test_mask[..., None],
                    context_masks.shape,
                )
                if fixed_test_mask is not None
                else free[..., None] & ~context_masks
            )

            out_path = output_path(args.out_root, graph_path, city)
            if not (args.skip_existing and out_path.exists()):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_path,
                    coarse=coarse.astype(np.float16),
                    target=target_maps.transpose(2, 0, 1).astype(np.float16),
                    side=side.astype(np.float16),
                    valid=free,
                    labels=labels.astype(np.uint16),
                    sparse=sparse_maps.astype(np.float16),
                    obs_mask=context_masks.transpose(2, 0, 1),
                    stage1_unobserved_pixel_mask=evaluation_mask.transpose(2, 0, 1),
                    evaluation_mask=evaluation_mask.transpose(2, 0, 1),
                    graph_path=str(graph_path),
                    manifest_path=str(manifest_path),
                    auxiliary_manifest_path=str(auxiliary_manifest_path),
                    city=city,
                    xml_idx=xml_idx,
                    tx_idx=tx_idx,
                )
            index.append(
                {
                    "cache_path": str(out_path),
                    "graph_path": str(graph_path),
                    "city": city,
                    "xml_idx": xml_idx,
                    "tx_idx": tx_idx,
                    "sample_key": sample_key,
                }
            )
            if number == 1 or number % 25 == 0 or number == len(paths):
                print(f"[{number}/{len(paths)}] {out_path}", flush=True)

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    metadata = {
        "protocol": (
            INDUCTIVE_FIXED_CACHE_PROTOCOL
            if fixed_test_rate_percent > 0 and source_split_mode == "xml_inductive"
            else FIXED_CACHE_PROTOCOL
            if fixed_test_rate_percent > 0
            else "budget_internal_stage1_projection_v10"
        ),
        "num_examples": len(index),
        "source_stage1_protocol": metrics["protocol"],
        "stage1_split_mode": source_split_mode,
        "stage1_operating_point": source_split_mode,
        "source_stage1_split_groups": source_split_groups,
        "source_run_dir": str(args.run_dir),
        "source_checkpoint": (
            str(checkpoint_path) if args.prior_method == "model" else None
        ),
        "source_checkpoint_sha256": (
            sha256(checkpoint_path) if args.prior_method == "model" else None
        ),
        "source_metrics_sha256": sha256(metrics_path),
        "model": model_label,
        "source_model_key": source_model_key,
        "architecture": architecture,
        "rates_percent": rates,
        "mask_mode": mask_mode,
        "auxiliary_mode": auxiliary_mode,
        "mask_seed": mask_seed,
        "training_mask_seed": training_mask_seed,
        "model_seed": (
            int(saved["seed"]) if args.prior_method == "model" else None
        ),
        "source_protocol_anchor_model_seed": int(saved["seed"]),
        "selection_seed": selection_seed,
        "validation_anchor_fraction_within_budget": val_anchor_fraction,
        "sampling_mode": sampling_mode,
        "measurement_noise_db": measurement_noise_db,
        "fixed_test_rate_percent": fixed_test_rate_percent,
        "fixed_test_seed": fixed_test_seed,
        "test_pixel_policy": (
            "fixed_all_band_excluded_holdout_v11"
            if fixed_test_rate_percent > 0
            else "all_unobserved_pixels_v10"
        ),
        "test_definition": (
            "deterministic free-space pixel holdout excluded from every carrier observation pool"
            if fixed_test_rate_percent > 0
            else "all free-space pixels absent from each frequency observation mask"
        ),
        "sparse_mask_source": "recomputed from controlled protocol; legacy graph masks ignored",
        "geometry_mask_source": "same-XML footprint stored in the selected graph",
        "valid_floor_policy": "valid -120 dB no-path pixels retained; geometry buildings excluded",
        "idw_injected": bool(args.prior_method == "idw"),
        "prediction_grid": (
            "native_pixel_grid" if args.prior_method == "idw" else "superpixel_projection"
        ),
        "idw_parameters": (
            {"neighbors": int(args.idw_neighbors), "power": float(args.idw_power)}
            if args.prior_method == "idw"
            else None
        ),
        "freqs_ghz": [float(value) for value in FREQS_GHZ],
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
