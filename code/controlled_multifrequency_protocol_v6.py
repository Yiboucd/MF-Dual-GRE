#!/usr/bin/env python3
"""Controlled multi-frequency sparse-observation protocol.

This loader keeps the geometry graph and dense evaluation target fixed while
changing only the per-frequency context measurements. It supports independent
or aligned masks and a deterministic shuffled-auxiliary control.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from build_clean_superpixel_graphs import FREQS_GHZ, load_manifest, load_maps, rss_to_norm
from build_clean_superpixel_graphs_v2 import masked_region_means, stable_seed


NF = len(FREQS_GHZ)
GEOM = slice(2 * NF, 2 * NF + 5)


def scalar_text(value: np.ndarray | str) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return str(value.item())
    return str(value)


@dataclass
class ControlledExample:
    path: Path
    name: str
    x: torch.Tensor
    train_target: torch.Tensor
    train_mask: torch.Tensor
    val_target: torch.Tensor
    val_mask: torch.Tensor
    val_pixel_weights: torch.Tensor
    val_within_sse: torch.Tensor
    val_nonfloor_target: torch.Tensor
    val_nonfloor_mask: torch.Tensor
    val_nonfloor_pixel_weights: torch.Tensor
    val_nonfloor_within_sse: torch.Tensor
    test_target: torch.Tensor
    test_mask: torch.Tensor
    test_pixel_weights: torch.Tensor
    test_within_sse: torch.Tensor
    test_nonfloor_target: torch.Tensor
    test_nonfloor_mask: torch.Tensor
    test_nonfloor_pixel_weights: torch.Tensor
    test_nonfloor_within_sse: torch.Tensor
    edge_index: torch.Tensor
    labels: np.ndarray
    building: np.ndarray
    metadata: dict[str, object]


def validate_rates(rates: list[float]) -> list[float]:
    if len(rates) != NF:
        raise ValueError(f"Expected {NF} observation rates, got {len(rates)}")
    values = [float(rate) for rate in rates]
    if any(rate < 0 or rate > 100 for rate in values):
        raise ValueError(f"Observation rates must lie in [0, 100]: {values}")
    return values


def select_paths(
    graph_root: Path,
    cities: list[str],
    max_graphs: int,
    selection_seed: int,
) -> list[Path]:
    paths: list[Path] = []
    for city in cities:
        paths.extend(sorted((graph_root / city).rglob("*.npz")))
    random.Random(selection_seed).shuffle(paths)
    if max_graphs > 0:
        paths = paths[:max_graphs]
    if not paths:
        raise RuntimeError(f"No template graphs under {graph_root}")
    return paths


def city_of(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        return scalar_text(data["city"])


def shuffled_pairs(
    paths: list[Path],
    auxiliary_pool: list[Path],
    offset: int,
) -> dict[Path, Path]:
    by_city: dict[str, list[Path]] = {}
    for path in auxiliary_pool:
        by_city.setdefault(city_of(path), []).append(path)
    result: dict[Path, Path] = {}
    for path in paths:
        city = city_of(path)
        ordered = sorted(by_city[city])
        if len(ordered) < 2:
            raise RuntimeError(f"Shuffled auxiliary control needs two {city} examples")
        index = ordered.index(path)
        shift = max(1, int(offset)) % len(ordered)
        if shift == 0:
            shift = 1
        paired = ordered[(index + shift) % len(ordered)]
        if paired == path:
            raise RuntimeError(f"Shuffled auxiliary paired an example with itself: {path}")
        result[path] = paired
    return result


def load_manifest_maps(path: Path) -> tuple[Path, dict[str, object], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        manifest_path = Path(scalar_text(data["manifest_path"]))
    manifest = load_manifest(manifest_path)
    return manifest_path, manifest, rss_to_norm(load_maps(manifest)).astype(np.float32)


def make_fixed_query_masks(
    free: np.ndarray,
    mask_seed: int,
    sample_key: str,
    val_rate_percent: float,
    test_rate_percent: float,
) -> tuple[np.ndarray, np.ndarray]:
    if val_rate_percent <= 0 or test_rate_percent <= 0:
        raise ValueError("Fixed validation and test rates must both be positive")
    if val_rate_percent + test_rate_percent >= 100:
        raise ValueError("Fixed validation and test rates leave no context pool")

    h, w = free.shape
    valid = np.flatnonzero(free)
    rng = np.random.default_rng(stable_seed(mask_seed, sample_key, "fixed-query-v6"))
    permutation = rng.permutation(valid)
    val_count = max(1, int(round(val_rate_percent / 100.0 * valid.size)))
    test_count = max(1, int(round(test_rate_percent / 100.0 * valid.size)))
    if val_count + test_count >= valid.size:
        raise RuntimeError(f"Fixed query masks exhaust free pixels for {sample_key}")

    val_flat = np.zeros(h * w, dtype=bool)
    test_flat = np.zeros(h * w, dtype=bool)
    val_flat[permutation[:val_count]] = True
    test_flat[permutation[val_count : val_count + test_count]] = True
    val_mask = val_flat.reshape(h, w)
    test_mask = test_flat.reshape(h, w)
    if np.any(val_mask & test_mask):
        raise RuntimeError(f"Fixed query overlap for {sample_key}")
    return val_mask, test_mask


def make_context_masks(
    free: np.ndarray,
    query_val_mask: np.ndarray,
    query_test_mask: np.ndarray,
    rates: list[float],
    target_index: int,
    mask_mode: str,
    mask_seed: int,
    sample_key: str,
) -> np.ndarray:
    h, w = free.shape
    valid = np.flatnonzero(free)
    available = np.flatnonzero(free & ~query_val_mask & ~query_test_mask)
    masks = np.zeros((h, w, NF), dtype=bool)
    rng = np.random.default_rng(
        stable_seed(mask_seed, sample_key, target_index, "target-context-v6")
    )
    target_permutation = rng.permutation(available)
    if mask_mode not in {"aligned", "independent"}:
        raise ValueError(mask_mode)

    for fi, rate in enumerate(rates):
        count = min(valid.size, int(round(rate / 100.0 * valid.size)))
        if count == 0:
            continue
        if count > available.size:
            raise ValueError(
                f"Rate {rate}% needs {count} pixels but only {available.size} are available"
            )
        if mask_mode == "aligned" or fi == target_index:
            permutation = target_permutation
        else:
            rng = np.random.default_rng(
                stable_seed(mask_seed, sample_key, fi, "aux-context-v6")
            )
            permutation = rng.permutation(available)
        flat = np.zeros(h * w, dtype=bool)
        flat[permutation[:count]] = True
        masks[..., fi] = flat.reshape(h, w)
        if np.any(masks[..., fi] & query_val_mask) or np.any(
            masks[..., fi] & query_test_mask
        ):
            raise RuntimeError(f"Context/query overlap for {sample_key}, frequency {fi}")
    return masks


def heldout_targets(
    maps_norm: np.ndarray,
    labels: np.ndarray,
    query_val_mask: np.ndarray,
    query_test_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    num_nodes = int(labels.max())
    val_target = np.zeros((num_nodes, NF), dtype=np.float32)
    test_target = np.zeros((num_nodes, NF), dtype=np.float32)
    val_mask = np.zeros((num_nodes, NF), dtype=bool)
    test_mask = np.zeros((num_nodes, NF), dtype=bool)
    val_weights = np.zeros((num_nodes, NF), dtype=np.float32)
    test_weights = np.zeros((num_nodes, NF), dtype=np.float32)
    val_within_sse = np.zeros((num_nodes, NF), dtype=np.float32)
    test_within_sse = np.zeros((num_nodes, NF), dtype=np.float32)
    val_nonfloor_target = np.zeros((num_nodes, NF), dtype=np.float32)
    test_nonfloor_target = np.zeros((num_nodes, NF), dtype=np.float32)
    val_nonfloor_mask = np.zeros((num_nodes, NF), dtype=bool)
    test_nonfloor_mask = np.zeros((num_nodes, NF), dtype=bool)
    val_nonfloor_weights = np.zeros((num_nodes, NF), dtype=np.float32)
    test_nonfloor_weights = np.zeros((num_nodes, NF), dtype=np.float32)
    val_nonfloor_within_sse = np.zeros((num_nodes, NF), dtype=np.float32)
    test_nonfloor_within_sse = np.zeros((num_nodes, NF), dtype=np.float32)

    flat_labels = labels.reshape(-1)

    def within_sse(values: np.ndarray, pixel_mask: np.ndarray, means: np.ndarray) -> np.ndarray:
        active = pixel_mask.reshape(-1) & (flat_labels > 0)
        labs = flat_labels[active] - 1
        vals = values.reshape(-1)[active].astype(np.float64)
        sums_sq = np.bincount(labs, weights=vals * vals, minlength=num_nodes)
        counts = np.bincount(labs, minlength=num_nodes)
        result = sums_sq - counts * means.astype(np.float64) ** 2
        return np.maximum(result, 0.0).astype(np.float32)

    for fi in range(NF):
        values, counts = masked_region_means(
            maps_norm[..., fi], query_val_mask, labels, num_nodes
        )
        val_target[:, fi] = values
        val_mask[:, fi] = counts > 0
        val_weights[:, fi] = counts
        val_within_sse[:, fi] = within_sse(
            maps_norm[..., fi], query_val_mask, values
        )
        nonfloor_val_pixels = query_val_mask & (maps_norm[..., fi] > 1e-6)
        values, counts = masked_region_means(
            maps_norm[..., fi], nonfloor_val_pixels, labels, num_nodes
        )
        val_nonfloor_target[:, fi] = values
        val_nonfloor_mask[:, fi] = counts > 0
        val_nonfloor_weights[:, fi] = counts
        val_nonfloor_within_sse[:, fi] = within_sse(
            maps_norm[..., fi], nonfloor_val_pixels, values
        )
        values, counts = masked_region_means(
            maps_norm[..., fi], query_test_mask, labels, num_nodes
        )
        test_target[:, fi] = values
        test_mask[:, fi] = counts > 0
        test_weights[:, fi] = counts
        test_within_sse[:, fi] = within_sse(
            maps_norm[..., fi], query_test_mask, values
        )
        nonfloor_test_pixels = query_test_mask & (maps_norm[..., fi] > 1e-6)
        values, counts = masked_region_means(
            maps_norm[..., fi], nonfloor_test_pixels, labels, num_nodes
        )
        test_nonfloor_target[:, fi] = values
        test_nonfloor_mask[:, fi] = counts > 0
        test_nonfloor_weights[:, fi] = counts
        test_nonfloor_within_sse[:, fi] = within_sse(
            maps_norm[..., fi], nonfloor_test_pixels, values
        )

    return (
        val_target,
        val_mask,
        val_weights,
        val_within_sse,
        val_nonfloor_target,
        val_nonfloor_mask,
        val_nonfloor_weights,
        val_nonfloor_within_sse,
        test_target,
        test_mask,
        test_weights,
        test_within_sse,
        test_nonfloor_target,
        test_nonfloor_mask,
        test_nonfloor_weights,
        test_nonfloor_within_sse,
    )


def make_example(
    path: Path,
    auxiliary_path: Path,
    rates: list[float],
    target_index: int,
    mask_mode: str,
    auxiliary_mode: str,
    mask_seed: int,
    val_rate_percent: float,
    test_rate_percent: float,
) -> ControlledExample:
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"].astype(np.int32)
        building = data["building_mask"].astype(bool)
        edge_index = data["edge_index"].astype(np.int64)
        side = data["feature_matrix"][:, GEOM].astype(np.float32)
        city = scalar_text(data["city"])
        xml_idx = int(np.asarray(data["xml_idx"]).item())
        tx_idx = int(np.asarray(data["tx_idx"]).item())

    manifest_path, manifest, target_maps = load_manifest_maps(path)
    auxiliary_manifest_path = manifest_path
    auxiliary_maps = target_maps
    if auxiliary_mode == "shuffled":
        auxiliary_manifest_path, _, auxiliary_maps = load_manifest_maps(auxiliary_path)
    elif auxiliary_mode != "true":
        raise ValueError(auxiliary_mode)
    if target_maps.shape[:2] != building.shape or auxiliary_maps.shape != target_maps.shape:
        raise ValueError(f"Map/geometry shape mismatch for {path}")

    free = ~building
    sample_key = f"{city}|xml={xml_idx}|tx={tx_idx}"
    query_val_mask, query_test_mask = make_fixed_query_masks(
        free,
        mask_seed,
        sample_key,
        val_rate_percent,
        test_rate_percent,
    )
    context_masks = make_context_masks(
        free,
        query_val_mask,
        query_test_mask,
        rates,
        target_index,
        mask_mode,
        mask_seed,
        sample_key,
    )
    num_nodes = int(labels.max())
    sparse_values = np.zeros((num_nodes, NF), dtype=np.float32)
    observed_nodes = np.zeros((num_nodes, NF), dtype=np.float32)
    train_target = np.zeros((num_nodes, NF), dtype=np.float32)
    train_mask = np.zeros((num_nodes, NF), dtype=bool)

    for fi in range(NF):
        input_map = target_maps[..., fi]
        if auxiliary_mode == "shuffled" and fi != target_index:
            input_map = auxiliary_maps[..., fi]
        values, counts = masked_region_means(
            input_map, context_masks[..., fi], labels, num_nodes
        )
        sparse_values[:, fi] = values
        observed_nodes[:, fi] = counts > 0
        values, counts = masked_region_means(
            target_maps[..., fi], context_masks[..., fi], labels, num_nodes
        )
        train_target[:, fi] = values
        train_mask[:, fi] = counts > 0

    (
        val_target,
        val_mask,
        val_weights,
        val_within_sse,
        val_nonfloor_target,
        val_nonfloor_mask,
        val_nonfloor_weights,
        val_nonfloor_within_sse,
        test_target,
        test_mask,
        test_weights,
        test_within_sse,
        test_nonfloor_target,
        test_nonfloor_mask,
        test_nonfloor_weights,
        test_nonfloor_within_sse,
    ) = heldout_targets(target_maps, labels, query_val_mask, query_test_mask)
    x = np.concatenate([sparse_values, observed_nodes, side], axis=1).astype(np.float32)
    metadata = {
        "city": city,
        "xml_idx": xml_idx,
        "tx_idx": tx_idx,
        "manifest_path": str(manifest_path),
        "auxiliary_manifest_path": str(auxiliary_manifest_path),
        "rates_percent": rates,
        "target_index": int(target_index),
        "target_frequency_ghz": float(FREQS_GHZ[target_index]),
        "mask_mode": mask_mode,
        "auxiliary_mode": auxiliary_mode,
        "mask_seed": int(mask_seed),
        "fixed_val_rate_percent": float(val_rate_percent),
        "fixed_test_rate_percent": float(test_rate_percent),
        "free_pixel_count": int(free.sum()),
        "context_pixel_counts": [int(context_masks[..., fi].sum()) for fi in range(NF)],
        "observed_node_counts": [int(observed_nodes[:, fi].sum()) for fi in range(NF)],
        "val_pixel_count": int(query_val_mask.sum()),
        "test_pixel_count": int(query_test_mask.sum()),
        "num_nodes": int(num_nodes),
        "num_directed_edges": int(edge_index.shape[1]),
        "sample_key": sample_key,
        "tx": manifest["tx"],
    }
    return ControlledExample(
        path=path,
        name=path.stem,
        x=torch.tensor(x, dtype=torch.float32),
        train_target=torch.tensor(train_target, dtype=torch.float32),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_target=torch.tensor(val_target, dtype=torch.float32),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        val_pixel_weights=torch.tensor(val_weights, dtype=torch.float32),
        val_within_sse=torch.tensor(val_within_sse, dtype=torch.float32),
        val_nonfloor_target=torch.tensor(val_nonfloor_target, dtype=torch.float32),
        val_nonfloor_mask=torch.tensor(val_nonfloor_mask, dtype=torch.bool),
        val_nonfloor_pixel_weights=torch.tensor(
            val_nonfloor_weights, dtype=torch.float32
        ),
        val_nonfloor_within_sse=torch.tensor(
            val_nonfloor_within_sse, dtype=torch.float32
        ),
        test_target=torch.tensor(test_target, dtype=torch.float32),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        test_pixel_weights=torch.tensor(test_weights, dtype=torch.float32),
        test_within_sse=torch.tensor(test_within_sse, dtype=torch.float32),
        test_nonfloor_target=torch.tensor(test_nonfloor_target, dtype=torch.float32),
        test_nonfloor_mask=torch.tensor(test_nonfloor_mask, dtype=torch.bool),
        test_nonfloor_pixel_weights=torch.tensor(
            test_nonfloor_weights, dtype=torch.float32
        ),
        test_nonfloor_within_sse=torch.tensor(
            test_nonfloor_within_sse, dtype=torch.float32
        ),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        labels=labels,
        building=building,
        metadata=metadata,
    )


def load_controlled_examples(
    graph_root: Path,
    cities: list[str],
    max_graphs: int,
    rates: list[float],
    target_index: int,
    mask_mode: str,
    auxiliary_mode: str,
    mask_seed: int,
    selection_seed: int,
    val_rate_percent: float,
    test_rate_percent: float,
    shuffle_offset: int = 1,
) -> list[ControlledExample]:
    rates = validate_rates(rates)
    if not 0 <= target_index < NF:
        raise ValueError(target_index)
    paths = select_paths(graph_root, cities, max_graphs, selection_seed)
    if auxiliary_mode == "shuffled":
        auxiliary_pool = select_paths(graph_root, cities, 0, selection_seed)
        pairs = shuffled_pairs(paths, auxiliary_pool, shuffle_offset)
    else:
        pairs = {path: path for path in paths}
    examples = []
    for index, path in enumerate(paths, start=1):
        examples.append(
            make_example(
                path,
                pairs[path],
                rates,
                target_index,
                mask_mode,
                auxiliary_mode,
                mask_seed,
                val_rate_percent,
                test_rate_percent,
            )
        )
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(f"[controlled v6 load] {index}/{len(paths)} {path}", flush=True)
    return examples
