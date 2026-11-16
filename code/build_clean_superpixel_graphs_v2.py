#!/usr/bin/env python3
"""Leakage-free, geometry-masked superpixel graph construction for clean 2e8 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.segmentation import relabel_sequential, slic

from build_clean_superpixel_graphs import (
    FREQS_GHZ,
    bresenham,
    build_edges,
    centroids,
    find_manifests,
    load_manifest,
    load_maps,
    manifest_asset_path,
    region_means,
    rss_to_norm,
)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def load_geometry_mask(manifest: dict) -> tuple[np.ndarray, Path]:
    source = manifest_asset_path(manifest, manifest["source_coverage_dir"])
    for name in ("B_FOOTPRINT_GEOM_1024m_4m.npy", "B_MAP_GEOM_1024m_4m.npy"):
        path = source / name
        if path.exists():
            return np.load(path).astype(bool), path
    raise FileNotFoundError(f"No geometry footprint mask under {source}")


def build_geometry_superpixels(building: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    free = ~building
    h, w = building.shape
    building_distance = distance_transform_edt(free).astype(np.float32)
    scale = float(np.percentile(building_distance[free], 95)) if np.any(free) else 1.0
    building_distance = np.clip(building_distance / max(scale, 1.0), 0.0, 1.0)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    image = np.stack([building_distance, xx, yy], axis=-1)
    labels = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        mask=free,
        channel_axis=-1,
    ).astype(np.int32)
    labels[building] = 0
    labels, _, _ = relabel_sequential(labels)
    return labels.astype(np.int32)


def masked_region_means(
    values: np.ndarray,
    sample_mask: np.ndarray,
    labels: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat_labels = labels.reshape(-1)
    flat_mask = sample_mask.reshape(-1) & (flat_labels > 0)
    out = np.zeros(num_nodes, dtype=np.float32)
    counts = np.zeros(num_nodes, dtype=np.int32)
    if not np.any(flat_mask):
        return out, counts
    labs = flat_labels[flat_mask] - 1
    vals = values.reshape(-1)[flat_mask].astype(np.float64)
    sums = np.bincount(labs, weights=vals, minlength=num_nodes)
    counts = np.bincount(labs, minlength=num_nodes).astype(np.int32)
    used = counts > 0
    out[used] = (sums[used] / counts[used]).astype(np.float32)
    return out, counts


def idw_fill_masked(
    sparse_map: np.ndarray,
    obs_mask: np.ndarray,
    valid_mask: np.ndarray,
    neighbors: int,
    power: float,
    max_query_neighbor_pairs: int = 2_000_000,
) -> np.ndarray:
    out = np.zeros_like(sparse_map, dtype=np.float32)
    obs_y, obs_x = np.nonzero(obs_mask & valid_mask)
    if obs_y.size == 0:
        return out
    obs_coord = np.column_stack([obs_y, obs_x]).astype(np.float32)
    obs_values = sparse_map[obs_y, obs_x].astype(np.float32)
    query_y, query_x = np.nonzero(valid_mask)
    query = np.column_stack([query_y, query_x]).astype(np.float32)
    k = min(max(1, neighbors), obs_coord.shape[0])
    tree = cKDTree(obs_coord)
    chunk_size = max(1, min(query.shape[0], max_query_neighbor_pairs // k))
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        dist, idx = tree.query(query[start:stop], k=k)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        weights = 1.0 / (np.power(dist, power) + 1e-6)
        pred = np.sum(weights * obs_values[idx], axis=1) / np.sum(weights, axis=1)
        exact = dist[:, 0] < 1e-8
        if np.any(exact):
            pred[exact] = obs_values[idx[exact, 0]]
        out[query_y[start:stop], query_x[start:stop]] = pred.astype(np.float32)
    out[obs_y, obs_x] = obs_values
    return out


def ray_obstruction_depth(
    centers: np.ndarray,
    building: np.ndarray,
    tx_col: float,
    tx_row: float,
) -> np.ndarray:
    h, w = building.shape
    x0 = int(round(tx_col))
    y0 = int(round(tx_row))
    values = np.zeros((centers.shape[0], 1), dtype=np.float32)
    for node, center in enumerate(centers):
        x1, y1 = int(round(center[0])), int(round(center[1]))
        blocked = 0
        total = 0
        for x, y in bresenham(x0, y0, x1, y1):
            if 0 <= x < w and 0 <= y < h:
                blocked += int(building[y, x])
                total += 1
        values[node, 0] = blocked / max(total, 1)
    return values


def build_one_graph(manifest_path: Path, out_path: Path, args: argparse.Namespace) -> None:
    manifest = load_manifest(manifest_path)
    maps_db = load_maps(manifest)
    building, building_path = load_geometry_mask(manifest)
    if building.shape != maps_db.shape[:2]:
        raise ValueError(f"Mask/map shape mismatch: {building.shape} vs {maps_db.shape[:2]}")
    free = ~building
    labels = build_geometry_superpixels(building, args.superpixels, args.compactness)
    num_nodes = int(labels.max())
    if num_nodes < 10:
        raise RuntimeError(f"Too few superpixels for {manifest_path}: {num_nodes}")

    centers = centroids(labels, num_nodes)
    edge_index = build_edges(labels, centers, building, args.dist_th)
    gt_norm = region_means(rss_to_norm(maps_db), labels, num_nodes, nonzero=False)

    h, w = building.shape
    x_vec = centers[:, 0:1] / w
    y_vec = centers[:, 1:2] / h
    tx_col = w / 2.0 + float(manifest["tx"][0]) / 4.0
    tx_row = h / 2.0 + float(manifest["tx"][1]) / 4.0
    txd = np.sqrt((centers[:, 0:1] - tx_col) ** 2 + (centers[:, 1:2] - tx_row) ** 2)
    txd = txd / math.sqrt(h * h + w * w)

    dist_to_building = distance_transform_edt(free).astype(np.float32)
    building_proximity_map = np.exp(-dist_to_building / args.building_context_scale).astype(np.float32)
    building_proximity = region_means(building_proximity_map, labels, num_nodes, nonzero=False)
    radio_depth = ray_obstruction_depth(centers, building, tx_col, tx_row)

    nf = len(FREQS_GHZ)
    sparse_vectors = np.zeros((num_nodes, nf), dtype=np.float32)
    obs_node_masks = np.zeros((num_nodes, nf), dtype=np.float32)
    idw_vectors = np.zeros((num_nodes, nf), dtype=np.float32)
    sparse_maps = np.zeros((h, w, nf), dtype=np.float32)
    obs_pixel_masks = np.zeros((h, w, nf), dtype=bool)
    val_pixel_masks = np.zeros((h, w, nf), dtype=bool)
    val_target_vectors = np.zeros((num_nodes, nf), dtype=np.float32)
    train_mask = np.zeros((num_nodes, nf), dtype=bool)
    val_mask = np.zeros((num_nodes, nf), dtype=bool)
    test_mask = np.zeros((num_nodes, nf), dtype=bool)
    requested_counts = np.zeros(nf, dtype=np.int32)
    validation_counts = np.zeros(nf, dtype=np.int32)
    observed_node_counts = np.zeros(nf, dtype=np.int32)

    valid_pixels = np.flatnonzero(free)
    flat_labels = labels.reshape(-1)
    sample_key = f"{manifest['city']}|xml={manifest['xml_idx']}|tx={manifest['tx_idx']}"
    all_nodes = np.arange(num_nodes)
    for fi, freq in enumerate(FREQS_GHZ):
        known_n = min(valid_pixels.size, max(1, int(round(args.known_percent / 100.0 * valid_pixels.size))))
        sample_rng = np.random.default_rng(stable_seed(args.seed, sample_key, fi, "pixels"))
        permutation = sample_rng.permutation(valid_pixels)
        known = permutation[:known_n]
        requested_counts[fi] = known.size

        pixel_mask = np.zeros(h * w, dtype=bool)
        pixel_mask[known] = True
        pixel_mask = pixel_mask.reshape(h, w)
        sparse = np.zeros((h, w), dtype=np.float32)
        norm_map = rss_to_norm(maps_db[..., fi])
        sparse[pixel_mask] = norm_map[pixel_mask]
        sparse_maps[..., fi] = sparse
        obs_pixel_masks[..., fi] = pixel_mask

        node_values, node_counts = masked_region_means(sparse, pixel_mask, labels, num_nodes)
        node_observed = node_counts > 0
        sparse_vectors[:, fi] = node_values
        obs_node_masks[:, fi] = node_observed.astype(np.float32)
        train_mask[node_observed, fi] = True
        observed_node_counts[fi] = int(node_observed.sum())

        # Validation measurements are a separate sparse pixel set and never enter the input.
        remaining = permutation[known_n:]
        remaining_labels = flat_labels[remaining]
        if args.query_node_policy == "unobserved":
            remaining = remaining[(remaining_labels > 0) & (~node_observed[remaining_labels - 1])]
        else:
            remaining = remaining[remaining_labels > 0]
        val_n = min(remaining.size, max(1, int(round(args.val_fraction * known_n))))
        val_known = remaining[:val_n]
        validation_counts[fi] = val_known.size
        val_pixel_mask = np.zeros(h * w, dtype=bool)
        val_pixel_mask[val_known] = True
        val_pixel_mask = val_pixel_mask.reshape(h, w)
        val_pixel_masks[..., fi] = val_pixel_mask
        val_values, val_node_counts = masked_region_means(norm_map, val_pixel_mask, labels, num_nodes)
        val_nodes = val_node_counts > 0
        val_target_vectors[:, fi] = val_values
        val_mask[val_nodes, fi] = True
        test_nodes = all_nodes[~node_observed & ~val_nodes]
        test_mask[test_nodes, fi] = True

        idw_img = idw_fill_masked(sparse, pixel_mask, free, args.idw_neighbors, args.idw_power)
        idw_vectors[:, fi] = region_means(idw_img, labels, num_nodes, nonzero=False).squeeze(1)

    feature_matrix = np.concatenate(
        [
            sparse_vectors,
            obs_node_masks,
            x_vec.astype(np.float32),
            y_vec.astype(np.float32),
            txd.astype(np.float32),
            building_proximity.astype(np.float32),
            radio_depth.astype(np.float32),
            idw_vectors,
        ],
        axis=1,
    ).astype(np.float32)
    feature_names = (
        [f"sparse_{freq:g}G" for freq in FREQS_GHZ]
        + [f"obs_mask_{freq:g}G" for freq in FREQS_GHZ]
        + ["x", "y", "tx_distance", "building_proximity", "radio_depth"]
        + [f"idw_{freq:g}G" for freq in FREQS_GHZ]
    )

    floor_all = np.all(maps_db <= -119.95, axis=-1)
    mask_dice = 2.0 * np.count_nonzero(building & floor_all) / max(
        np.count_nonzero(building) + np.count_nonzero(floor_all), 1
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        feature_matrix=feature_matrix,
        feature_names=np.array(feature_names),
        matrix_gt=gt_norm.astype(np.float32),
        train_target_vectors=sparse_vectors.astype(np.float32),
        val_target_vectors=val_target_vectors.astype(np.float32),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        edge_index=edge_index,
        idw_vectors=idw_vectors.astype(np.float32),
        sparse_train_maps=sparse_maps,
        obs_train_pixel_masks=obs_pixel_masks,
        val_pixel_masks=val_pixel_masks,
        building_mask=building,
        labels=labels.astype(np.int32),
        centroids=centers.astype(np.float32),
        freqs_ghz=np.array(FREQS_GHZ, dtype=np.float32),
        manifest_path=str(manifest_path),
        geometry_mask_path=str(building_path),
        city=manifest["city"],
        xml_idx=int(manifest["xml_idx"]),
        tx_idx=int(manifest["tx_idx"]),
        tx=np.array(manifest["tx"], dtype=np.float32),
        known_percent=float(args.known_percent),
        actual_pixel_percent=requested_counts.astype(np.float32) / valid_pixels.size * 100.0,
        actual_val_pixel_percent=validation_counts.astype(np.float32) / valid_pixels.size * 100.0,
        observed_node_counts=observed_node_counts,
        superpixels=int(num_nodes),
        graph_schema="clean_superpixel_graph_v2_geometry_no_gt",
        sampling_seed=int(args.seed),
        idw_neighbors=int(args.idw_neighbors),
        idw_power=float(args.idw_power),
        mask_floor_dice=float(mask_dice),
        query_node_policy=np.array(args.query_node_policy),
    )


def output_path(out_root: Path, manifest_path: Path, args: argparse.Namespace) -> Path:
    manifest = load_manifest(manifest_path)
    city = manifest["city"]
    xml_part = manifest_path.parent.parent.name
    tx_part = manifest_path.parent.name
    name = f"{city}_{xml_part}_{tx_part}_sp{args.superpixels}_kp{args.known_percent:g}_v2.npz"
    return out_root / city / f"sp{args.superpixels}_kp{args.known_percent:g}" / name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cities", default="Davis,Woodland,Vacaville")
    parser.add_argument("--max-graphs", type=int, default=1160)
    parser.add_argument("--superpixels", type=int, default=2000)
    parser.add_argument("--known-percent", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--dist-th", type=float, default=30.0)
    parser.add_argument("--compactness", type=float, default=12.0)
    parser.add_argument("--building-context-scale", type=float, default=12.0)
    parser.add_argument("--idw-neighbors", type=int, default=64)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--query-node-policy", choices=["unobserved", "any_pixel"], default="unobserved")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()

    random.seed(args.seed)
    cities = [city.strip() for city in args.cities.split(",") if city.strip()]
    manifests = find_manifests(args.data_root, cities, completed_only=True)
    random.Random(args.seed).shuffle(manifests)
    if args.max_graphs > 0:
        manifests = manifests[: args.max_graphs]
    print(
        f"[build v2] manifests={len(manifests)} cities={cities} sp={args.superpixels} "
        f"kp={args.known_percent} mask=geometry superpixels=geometry-only",
        flush=True,
    )
    for index, manifest_path in enumerate(manifests, start=1):
        out_path = output_path(args.out_root, manifest_path, args)
        if args.skip_existing and out_path.exists():
            print(f"[{index}/{len(manifests)}] skip {out_path}", flush=True)
            continue
        print(f"[{index}/{len(manifests)}] build {manifest_path}", flush=True)
        build_one_graph(manifest_path, out_path, args)
        print(f"  -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
