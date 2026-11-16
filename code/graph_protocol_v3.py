#!/usr/bin/env python3
"""Pixel-disjoint validation and test targets for sparse graph completion."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from build_clean_superpixel_graphs import FREQS_GHZ, load_manifest, load_maps, rss_to_norm
from build_clean_superpixel_graphs_v2 import masked_region_means, stable_seed


def _scalar_text(value: np.ndarray | str) -> str:
    return str(value.item()) if isinstance(value, np.ndarray) and value.ndim == 0 else str(value)


def derive_pixel_disjoint_targets(
    data: np.lib.npyio.NpzFile,
    *,
    seed: int,
    val_fraction: float,
    exclude_training_queries: bool = False,
) -> dict[str, np.ndarray]:
    """Split free pixels into context, validation, and test sets with no overlap.

    Context pixels are the observations already stored in the graph. Validation
    pixels are sampled from their complement. Every remaining free pixel is a
    test pixel. Targets are means over each subset within each superpixel, so a
    partially observed superpixel can still be evaluated on its unseen pixels.
    """

    labels = data["labels"].astype(np.int32)
    building = data["building_mask"].astype(bool)
    context_masks = data["obs_train_pixel_masks"].astype(bool)
    training_query_masks = (
        data["val_pixel_masks"].astype(bool)
        if exclude_training_queries and "val_pixel_masks" in data
        else np.zeros_like(context_masks)
    )
    num_nodes = int(labels.max())
    manifest_path = Path(_scalar_text(data["manifest_path"]))
    manifest = load_manifest(manifest_path)
    maps_norm = rss_to_norm(load_maps(manifest))
    free = ~building
    nf = len(FREQS_GHZ)

    val_target = np.zeros((num_nodes, nf), dtype=np.float32)
    test_target = np.zeros((num_nodes, nf), dtype=np.float32)
    val_mask_nodes = np.zeros((num_nodes, nf), dtype=bool)
    test_mask_nodes = np.zeros((num_nodes, nf), dtype=bool)
    val_pixel_counts = np.zeros(nf, dtype=np.int32)
    test_pixel_counts = np.zeros(nf, dtype=np.int32)
    val_node_pixel_counts = np.zeros((num_nodes, nf), dtype=np.int32)
    test_node_pixel_counts = np.zeros((num_nodes, nf), dtype=np.int32)
    val_within_sse = np.zeros((num_nodes, nf), dtype=np.float32)
    test_within_sse = np.zeros((num_nodes, nf), dtype=np.float32)

    flat_labels = labels.reshape(-1)

    def within_region_sse(
        values: np.ndarray,
        pixel_mask: np.ndarray,
        means: np.ndarray,
    ) -> np.ndarray:
        active = pixel_mask.reshape(-1) & (flat_labels > 0)
        labs = flat_labels[active] - 1
        vals = values.reshape(-1)[active].astype(np.float64)
        sums_sq = np.bincount(labs, weights=vals * vals, minlength=num_nodes)
        counts = np.bincount(labs, minlength=num_nodes)
        result = sums_sq - counts * means.astype(np.float64) ** 2
        return np.maximum(result, 0.0).astype(np.float32)

    sample_key = f"{manifest['city']}|xml={manifest['xml_idx']}|tx={manifest['tx_idx']}"
    for fi in range(nf):
        context = context_masks[..., fi] & free
        training_query = training_query_masks[..., fi] & free
        candidates = np.flatnonzero(free & ~context & ~training_query)
        split_tag = "pixel-disjoint-v4-query-excluded" if exclude_training_queries else "pixel-disjoint-v3"
        rng = np.random.default_rng(stable_seed(seed, sample_key, fi, split_tag))
        candidates = rng.permutation(candidates)
        requested_val = max(1, int(round(val_fraction * int(context.sum()))))
        val_n = min(requested_val, max(0, candidates.size - 1))

        val_pixels = np.zeros(labels.size, dtype=bool)
        if val_n:
            val_pixels[candidates[:val_n]] = True
        val_pixels = val_pixels.reshape(labels.shape)
        test_pixels = free & ~context & ~training_query & ~val_pixels

        if (
            np.any(context & training_query)
            or np.any(context & val_pixels)
            or np.any(context & test_pixels)
            or np.any(training_query & val_pixels)
            or np.any(training_query & test_pixels)
            or np.any(val_pixels & test_pixels)
        ):
            raise RuntimeError(f"Pixel split overlap in {manifest_path} at frequency index {fi}")

        val_values, val_counts = masked_region_means(maps_norm[..., fi], val_pixels, labels, num_nodes)
        test_values, test_counts = masked_region_means(maps_norm[..., fi], test_pixels, labels, num_nodes)
        val_target[:, fi] = val_values
        test_target[:, fi] = test_values
        val_mask_nodes[:, fi] = val_counts > 0
        test_mask_nodes[:, fi] = test_counts > 0
        val_node_pixel_counts[:, fi] = val_counts
        test_node_pixel_counts[:, fi] = test_counts
        val_within_sse[:, fi] = within_region_sse(
            maps_norm[..., fi], val_pixels, val_values
        )
        test_within_sse[:, fi] = within_region_sse(
            maps_norm[..., fi], test_pixels, test_values
        )
        val_pixel_counts[fi] = int(val_pixels.sum())
        test_pixel_counts[fi] = int(test_pixels.sum())

    return {
        "val_target_vectors": val_target,
        "test_target_vectors": test_target,
        "val_mask": val_mask_nodes,
        "test_mask": test_mask_nodes,
        "val_pixel_counts": val_pixel_counts,
        "test_pixel_counts": test_pixel_counts,
        "val_node_pixel_counts": val_node_pixel_counts,
        "test_node_pixel_counts": test_node_pixel_counts,
        "val_within_sse": val_within_sse,
        "test_within_sse": test_within_sse,
        "target_schema": np.array(
            "pixel_disjoint_superpixel_targets_v4_query_excluded_exact_pixel_sse"
            if exclude_training_queries
            else "pixel_disjoint_superpixel_targets_v3_exact_pixel_sse"
        ),
    }


def load_or_build_pixel_disjoint_targets(
    graph_path: Path,
    data: np.lib.npyio.NpzFile,
    cache_root: Path,
    *,
    seed: int,
    val_fraction: float,
    exclude_training_queries: bool = False,
) -> dict[str, np.ndarray]:
    city = _scalar_text(data["city"])
    suffix = "targets_v4_query_excluded" if exclude_training_queries else "targets_v3"
    cache_path = cache_root / city / f"{graph_path.stem}_{suffix}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            required = {"val_within_sse", "test_within_sse"}
            if required.issubset(cached.files):
                return {key: cached[key] for key in cached.files}

    result = derive_pixel_disjoint_targets(
        data,
        seed=seed,
        val_fraction=val_fraction,
        exclude_training_queries=exclude_training_queries,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **result)
    os.replace(temp_path, cache_path)
    return result
