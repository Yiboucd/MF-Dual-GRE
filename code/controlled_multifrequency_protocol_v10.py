#!/usr/bin/env python3
"""Leakage-resistant controlled multi-frequency observation protocol.

The stated observation rate is the complete measurement budget. A fixed subset
of observed node-frequency anchors is reserved for checkpoint selection and is
removed from the training input. Dense ray-tracing pixels are used only for the
final evaluation on pixels that were never observed.
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
GEOM_FEATURE_NAMES = (
    "x",
    "y",
    "tx_distance",
    "building_proximity",
    "radio_depth",
)


def scalar_text(value: np.ndarray | str) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return str(value.item())
    return str(value)


def validate_geometry_feature_schema(feature_names: np.ndarray) -> tuple[str, ...]:
    """Reject graph files whose geometry slice could contain learned/data priors."""

    names = tuple(scalar_text(value) for value in np.asarray(feature_names).reshape(-1))
    selected = names[GEOM]
    if selected != GEOM_FEATURE_NAMES:
        raise ValueError(
            "Unexpected Stage-1 geometry feature schema: "
            f"expected {GEOM_FEATURE_NAMES}, got {selected}"
        )
    return selected


@dataclass
class ControlledExample:
    path: Path
    name: str
    x: torch.Tensor
    train_x: torch.Tensor
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


def graph_identity(path: Path) -> tuple[str, int, int]:
    with np.load(path, allow_pickle=False) as data:
        return (
            scalar_text(data["city"]),
            int(np.asarray(data["xml_idx"]).item()),
            int(np.asarray(data["tx_idx"]).item()),
        )


def shuffled_pairs(
    paths: list[Path],
    auxiliary_pool: list[Path],
    offset: int,
) -> dict[Path, Path]:
    """Pair each target with another TX from the same XML geometry.

    This destroys TX-specific cross-frequency correspondence while keeping the
    building mesh fixed, so the negative control is not confounded by a scene
    mismatch.
    """

    by_scene: dict[tuple[str, int], list[tuple[int, Path]]] = {}
    for path in auxiliary_pool:
        city, xml_idx, tx_idx = graph_identity(path)
        by_scene.setdefault((city, xml_idx), []).append((tx_idx, path))
    result: dict[Path, Path] = {}
    for path in paths:
        city, xml_idx, _ = graph_identity(path)
        ordered = [item[1] for item in sorted(by_scene[(city, xml_idx)])]
        if len(ordered) < 2:
            raise RuntimeError(
                f"Shuffled auxiliary control needs two TX examples for "
                f"{city} XML {xml_idx}"
            )
        index = ordered.index(path)
        shift = max(1, int(offset)) % len(ordered)
        if shift == 0:
            shift = 1
        paired = ordered[(index + shift) % len(ordered)]
        if paired == path:
            raise RuntimeError(f"Shuffled auxiliary paired an example with itself: {path}")
        result[path] = paired
    return result


def spatial_sample_order(
    free: np.ndarray,
    seed: int,
    sampling_mode: str,
) -> np.ndarray:
    """Return a deterministic ranking of free pixels for one sampling design."""

    valid = np.flatnonzero(free)
    if not valid.size:
        raise RuntimeError("Sampling mask contains no free pixels")
    rng = np.random.default_rng(seed)
    if sampling_mode == "random":
        return rng.permutation(valid)

    rows, cols = np.unravel_index(valid, free.shape)
    coordinates = np.column_stack([rows, cols]).astype(np.float64)
    jitter = rng.uniform(0.0, 1e-6, valid.size)

    if sampling_mode == "clustered":
        num_centers = max(2, min(12, int(round(np.sqrt(valid.size) / 24.0))))
        centers = coordinates[rng.choice(valid.size, size=num_centers, replace=False)]
        distance_sq = np.full(valid.size, np.inf, dtype=np.float64)
        for center in centers:
            delta = coordinates - center
            distance_sq = np.minimum(distance_sq, np.sum(delta * delta, axis=1))
        return valid[np.argsort(distance_sq + jitter)]

    if sampling_mode == "trajectory":
        # A deterministic polyline-like route. This is a trajectory stress test,
        # not a claim that the samples follow a mapped road network.
        waypoints = coordinates[rng.choice(valid.size, size=3, replace=False)]
        distance_sq = np.full(valid.size, np.inf, dtype=np.float64)
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            segment = end - start
            denominator = max(float(np.dot(segment, segment)), 1e-12)
            projection = np.clip(
                ((coordinates - start) @ segment) / denominator, 0.0, 1.0
            )
            nearest = start + projection[:, None] * segment
            delta = coordinates - nearest
            distance_sq = np.minimum(distance_sq, np.sum(delta * delta, axis=1))
        return valid[np.argsort(distance_sq + jitter)]

    raise ValueError(sampling_mode)


def make_fixed_test_mask(
    free: np.ndarray,
    test_mask_seed: int,
    sample_key: str,
    rate_percent: float,
) -> np.ndarray:
    """Select one deterministic holdout independent of observation-mask seeds."""

    if not 0.0 < rate_percent < 100.0:
        raise ValueError("Fixed-test rate must lie in (0, 100)")
    valid = np.flatnonzero(free)
    if not valid.size:
        raise RuntimeError("Fixed-test mask contains no free pixels")
    count = max(1, int(round(rate_percent / 100.0 * valid.size)))
    if count >= valid.size:
        raise RuntimeError("Fixed-test mask exhausts the free-space pixels")
    rng = np.random.default_rng(
        stable_seed(test_mask_seed, sample_key, "fixed-pixel-test-v11")
    )
    selected = rng.permutation(valid)[:count]
    flat = np.zeros(free.size, dtype=bool)
    flat[selected] = True
    return flat.reshape(free.shape)


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
    rates: list[float],
    target_index: int,
    mask_mode: str,
    mask_seed: int,
    sample_key: str,
    sampling_mode: str = "random",
    excluded_mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = free.shape
    valid = np.flatnonzero(free)
    if excluded_mask is not None:
        if excluded_mask.shape != free.shape:
            raise ValueError("Excluded-pixel mask and free-space mask must match")
        if np.any(excluded_mask & ~free):
            raise ValueError("Excluded-pixel mask contains non-free pixels")
        excluded_flat = excluded_mask.reshape(-1)
        available = valid[~excluded_flat[valid]]
    else:
        excluded_flat = None
        available = valid
    masks = np.zeros((h, w, NF), dtype=bool)
    target_seed = (
        stable_seed(mask_seed, sample_key, target_index, "target-context-v6")
        if sampling_mode == "random"
        else stable_seed(
            mask_seed, sample_key, target_index, sampling_mode, "target-context-v10"
        )
    )
    target_permutation = spatial_sample_order(
        free,
        target_seed,
        sampling_mode,
    )
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
            auxiliary_seed = (
                stable_seed(mask_seed, sample_key, fi, "aux-context-v6")
                if sampling_mode == "random"
                else stable_seed(
                    mask_seed, sample_key, fi, sampling_mode, "aux-context-v10"
                )
            )
            permutation = spatial_sample_order(
                free,
                auxiliary_seed,
                sampling_mode,
            )
        if excluded_flat is not None:
            permutation = permutation[~excluded_flat[permutation]]
        flat = np.zeros(h * w, dtype=bool)
        flat[permutation[:count]] = True
        masks[..., fi] = flat.reshape(h, w)
    return masks


def targets_from_pixel_masks(
    maps_norm: np.ndarray,
    labels: np.ndarray,
    pixel_masks: np.ndarray,
) -> tuple[
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
    target = np.zeros((num_nodes, NF), dtype=np.float32)
    mask = np.zeros((num_nodes, NF), dtype=bool)
    weights = np.zeros((num_nodes, NF), dtype=np.float32)
    within_sse_values = np.zeros((num_nodes, NF), dtype=np.float32)
    nonfloor_target = np.zeros((num_nodes, NF), dtype=np.float32)
    nonfloor_mask = np.zeros((num_nodes, NF), dtype=bool)
    nonfloor_weights = np.zeros((num_nodes, NF), dtype=np.float32)
    nonfloor_within_sse = np.zeros((num_nodes, NF), dtype=np.float32)

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
        pixel_mask = pixel_masks[..., fi]
        values, counts = masked_region_means(
            maps_norm[..., fi], pixel_mask, labels, num_nodes
        )
        target[:, fi] = values
        mask[:, fi] = counts > 0
        weights[:, fi] = counts
        within_sse_values[:, fi] = within_sse(
            maps_norm[..., fi], pixel_mask, values
        )
        nonfloor_pixels = pixel_mask & (maps_norm[..., fi] > 1e-6)
        values, counts = masked_region_means(
            maps_norm[..., fi], nonfloor_pixels, labels, num_nodes
        )
        nonfloor_target[:, fi] = values
        nonfloor_mask[:, fi] = counts > 0
        nonfloor_weights[:, fi] = counts
        nonfloor_within_sse[:, fi] = within_sse(
            maps_norm[..., fi], nonfloor_pixels, values
        )

    return (
        target,
        mask,
        weights,
        within_sse_values,
        nonfloor_target,
        nonfloor_mask,
        nonfloor_weights,
        nonfloor_within_sse,
    )


def split_validation_anchors(
    observed_nodes: np.ndarray,
    target_indices: list[int],
    fraction: float,
    mask_seed: int,
    sample_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve validation anchors from inside the stated measurement budget."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("Validation-anchor fraction must lie in (0, 1)")
    train_mask = observed_nodes.astype(bool).copy()
    val_mask = np.zeros_like(train_mask)
    for fi in target_indices:
        active = np.flatnonzero(observed_nodes[:, fi])
        if active.size < 2:
            raise RuntimeError(
                f"Need at least two observed anchors for {sample_key}, frequency {fi}"
            )
        rng = np.random.default_rng(
            stable_seed(mask_seed, sample_key, fi, "anchor-validation-v10")
        )
        count = min(active.size - 1, max(1, int(round(fraction * active.size))))
        selected = rng.permutation(active)[:count]
        val_mask[selected, fi] = True
        train_mask[selected, fi] = False
    return train_mask, val_mask


def hide_node_frequency_anchors(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove both value and observation-indicator channels for fixed anchors."""

    if x.ndim != 2 or x.shape[1] < 2 * NF or mask.shape != (x.shape[0], NF):
        raise ValueError("Anchor mask and feature matrix shapes are inconsistent")
    result = x.copy()
    result[:, :NF][mask] = 0.0
    result[:, NF : 2 * NF][mask] = 0.0
    return result


def query_input_mask(
    target_query_mask: np.ndarray,
    observed_nodes: np.ndarray,
    mask_mode: str,
) -> np.ndarray:
    """Return the input entries hidden for a target-frequency query.

    With aligned sampling, retaining an auxiliary observation at the queried
    node makes support/query validation easier than the all-band-excluded test
    task. Hide every observed frequency at that node while keeping the loss on
    the requested target frequency only.
    """

    if target_query_mask.shape != observed_nodes.shape:
        raise ValueError("Query and observation masks must have the same shape")
    if mask_mode == "independent":
        return target_query_mask.astype(bool).copy()
    if mask_mode != "aligned":
        raise ValueError(mask_mode)
    queried_nodes = target_query_mask.astype(bool).any(axis=1, keepdims=True)
    return queried_nodes & observed_nodes.astype(bool)


def make_example(
    path: Path,
    auxiliary_path: Path,
    rates: list[float],
    target_indices: list[int],
    mask_mode: str,
    auxiliary_mode: str,
    mask_seed: int,
    val_anchor_fraction: float,
    sampling_mode: str,
    measurement_noise_db: float,
    fixed_test_rate_percent: float,
    fixed_test_seed: int,
) -> ControlledExample:
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"].astype(np.int32)
        building = data["building_mask"].astype(bool)
        edge_index = data["edge_index"].astype(np.int64)
        geometry_feature_names = validate_geometry_feature_schema(data["feature_names"])
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
    target_index = target_indices[0]
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
        target_index,
        mask_mode,
        mask_seed,
        sample_key,
        sampling_mode,
        fixed_test_mask,
    )
    num_nodes = int(labels.max())
    sparse_values = np.zeros((num_nodes, NF), dtype=np.float32)
    observed_nodes = np.zeros((num_nodes, NF), dtype=np.float32)
    anchor_target = np.zeros((num_nodes, NF), dtype=np.float32)
    anchor_weights = np.zeros((num_nodes, NF), dtype=np.float32)
    anchor_within_sse = np.zeros((num_nodes, NF), dtype=np.float32)
    anchor_nonfloor_target = np.zeros((num_nodes, NF), dtype=np.float32)
    anchor_nonfloor_weights = np.zeros((num_nodes, NF), dtype=np.float32)
    anchor_nonfloor_within_sse = np.zeros((num_nodes, NF), dtype=np.float32)

    flat_labels = labels.reshape(-1)

    def within_sse(values: np.ndarray, pixel_mask: np.ndarray, means: np.ndarray) -> np.ndarray:
        active = pixel_mask.reshape(-1) & (flat_labels > 0)
        labs = flat_labels[active] - 1
        vals = values.reshape(-1)[active].astype(np.float64)
        sums_sq = np.bincount(labs, weights=vals * vals, minlength=num_nodes)
        counts = np.bincount(labs, minlength=num_nodes)
        return np.maximum(
            sums_sq - counts * means.astype(np.float64) ** 2, 0.0
        ).astype(np.float32)

    for fi in range(NF):
        input_map = target_maps[..., fi]
        if auxiliary_mode == "shuffled" and fi != target_index:
            input_map = auxiliary_maps[..., fi]
        measurement_map = input_map
        if measurement_noise_db > 0:
            measurement_map = input_map.copy()
            active = context_masks[..., fi]
            rng = np.random.default_rng(
                stable_seed(
                    mask_seed,
                    sample_key,
                    fi,
                    sampling_mode,
                    measurement_noise_db,
                    "measurement-noise-v10",
                )
            )
            measurement_map[active] = measurement_map[active] + rng.normal(
                0.0,
                measurement_noise_db / 40.0,
                int(active.sum()),
            )
        values, counts = masked_region_means(
            measurement_map, context_masks[..., fi], labels, num_nodes
        )
        sparse_values[:, fi] = values
        observed_nodes[:, fi] = counts > 0
        anchor_map = (
            measurement_map
            if auxiliary_mode == "true" or fi == target_index
            else target_maps[..., fi]
        )
        values, counts = masked_region_means(
            anchor_map, context_masks[..., fi], labels, num_nodes
        )
        anchor_target[:, fi] = values
        anchor_weights[:, fi] = counts
        anchor_within_sse[:, fi] = within_sse(
            anchor_map, context_masks[..., fi], values
        )
        nonfloor_pixels = context_masks[..., fi] & (target_maps[..., fi] > 1e-6)
        values, counts = masked_region_means(
            anchor_map, nonfloor_pixels, labels, num_nodes
        )
        anchor_nonfloor_target[:, fi] = values
        anchor_nonfloor_weights[:, fi] = counts
        anchor_nonfloor_within_sse[:, fi] = within_sse(
            anchor_map, nonfloor_pixels, values
        )

    train_mask, val_mask = split_validation_anchors(
        observed_nodes > 0,
        target_indices,
        val_anchor_fraction,
        mask_seed,
        sample_key,
    )
    val_target = anchor_target.copy()
    val_weights = anchor_weights * val_mask
    val_within_sse = anchor_within_sse * val_mask
    val_nonfloor_target = anchor_nonfloor_target.copy()
    val_nonfloor_mask = val_mask & (anchor_nonfloor_weights > 0)
    val_nonfloor_weights = anchor_nonfloor_weights * val_mask
    val_nonfloor_within_sse = anchor_nonfloor_within_sse * val_mask

    test_pixel_masks = (
        np.broadcast_to(
            free[..., None] & fixed_test_mask[..., None],
            context_masks.shape,
        )
        if fixed_test_mask is not None
        else free[..., None] & ~context_masks
    )

    (
        test_target,
        test_mask,
        test_weights,
        test_within_sse,
        test_nonfloor_target,
        test_nonfloor_mask,
        test_nonfloor_weights,
        test_nonfloor_within_sse,
    ) = targets_from_pixel_masks(
        target_maps,
        labels,
        test_pixel_masks,
    )
    x = np.concatenate([sparse_values, observed_nodes, side], axis=1).astype(np.float32)
    val_input_mask = query_input_mask(val_mask, observed_nodes > 0, mask_mode)
    train_x = hide_node_frequency_anchors(x, val_input_mask)
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
        "auxiliary_shuffle_policy": (
            "same_xml_different_tx" if auxiliary_mode == "shuffled" else "none"
        ),
        "sampling_mode": sampling_mode,
        "measurement_noise_db": float(measurement_noise_db),
        "fixed_test_rate_percent": float(fixed_test_rate_percent),
        "fixed_test_seed": int(fixed_test_seed),
        "test_pixel_policy": (
            "fixed_all_band_excluded_holdout_v11"
            if fixed_test_mask is not None
            else "all_unobserved_pixels_v10"
        ),
        "noisy_anchor_supervision": bool(measurement_noise_db > 0),
        "mask_seed": int(mask_seed),
        "validation_anchor_fraction": float(val_anchor_fraction),
        "free_pixel_count": int(free.sum()),
        "context_pixel_counts": [int(context_masks[..., fi].sum()) for fi in range(NF)],
        "observed_node_counts": [int(observed_nodes[:, fi].sum()) for fi in range(NF)],
        "validation_anchor_counts": val_mask.sum(axis=0).astype(int).tolist(),
        "validation_input_hidden_counts": (
            val_input_mask.sum(axis=0).astype(int).tolist()
        ),
        "aligned_query_policy": (
            "nodewise_all_observed_frequencies_hidden_v11"
            if mask_mode == "aligned"
            else "target_frequency_only_v11"
        ),
        "test_unobserved_pixel_counts": (
            test_pixel_masks.sum(axis=(0, 1)).astype(int).tolist()
        ),
        "num_nodes": int(num_nodes),
        "num_directed_edges": int(edge_index.shape[1]),
        "geometry_feature_names": list(geometry_feature_names),
        "graph_input_policy": "sparse_values_masks_plus_geometry_only_no_idw_no_dense_rss",
        "sample_key": sample_key,
        "tx": manifest["tx"],
    }
    return ControlledExample(
        path=path,
        name=path.stem,
        x=torch.tensor(x, dtype=torch.float32),
        train_x=torch.tensor(train_x, dtype=torch.float32),
        train_target=torch.tensor(anchor_target, dtype=torch.float32),
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
    target_indices: list[int],
    mask_mode: str,
    auxiliary_mode: str,
    mask_seed: int,
    selection_seed: int,
    val_anchor_fraction: float,
    shuffle_offset: int = 1,
    sampling_mode: str = "random",
    measurement_noise_db: float = 0.0,
    fixed_test_rate_percent: float = 0.0,
    fixed_test_seed: int = 9001,
) -> list[ControlledExample]:
    rates = validate_rates(rates)
    if sampling_mode not in {"random", "clustered", "trajectory"}:
        raise ValueError(sampling_mode)
    if measurement_noise_db < 0:
        raise ValueError("measurement_noise_db must be nonnegative")
    if fixed_test_rate_percent < 0 or fixed_test_rate_percent >= 100:
        raise ValueError("fixed_test_rate_percent must lie in [0, 100)")
    target_indices = sorted(set(int(index) for index in target_indices))
    if not target_indices or any(index < 0 or index >= NF for index in target_indices):
        raise ValueError(target_indices)
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
                target_indices,
                mask_mode,
                auxiliary_mode,
                mask_seed,
                val_anchor_fraction,
                sampling_mode,
                measurement_noise_db,
                fixed_test_rate_percent,
                fixed_test_seed,
            )
        )
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(f"[controlled v10 load] {index}/{len(paths)} {path}", flush=True)
    return examples
