#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from skimage.segmentation import relabel_sequential, slic


FREQS_GHZ = [2.4, 3.65, 4.9, 5.8, 24.0, 28.0, 39.0]


def rss_to_norm(arr_db: np.ndarray) -> np.ndarray:
    return np.clip((arr_db + 120.0) / 40.0, 0.0, 1.0).astype(np.float32)


def norm_to_rss(arr_norm: np.ndarray) -> np.ndarray:
    return arr_norm * 40.0 - 120.0


def load_manifest(path: Path) -> dict:
    """Read a manifest and bind portable paths to its dataset root.

    Release examples use <root>/<city>/<export>/<transmitter>/manifest.json.
    Absolute paths in original or newly generated manifests remain supported.
    """
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("release_path_base") == "dataset_root":
        manifest["_dataset_root"] = str(path.parents[3])
    return manifest


def manifest_asset_path(manifest: dict, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() and manifest.get("release_path_base") == "dataset_root":
        if "_dataset_root" not in manifest:
            raise ValueError("Use load_manifest(path) to bind release dataset paths")
        path = Path(manifest["_dataset_root"]) / path
    return path


def find_manifests(root: Path, cities: list[str], completed_only: bool = True) -> list[Path]:
    paths: list[Path] = []
    for city in cities:
        paths.extend(sorted((root / city).glob("xml_*/*/manifest.json")))
    if not completed_only:
        return paths
    out = []
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "completed_at" in data and len(data.get("records", [])) >= len(FREQS_GHZ):
            out.append(path)
    return out


def load_maps(manifest: dict) -> np.ndarray:
    by_freq = {round(float(r["freq_ghz"]), 2): manifest_asset_path(manifest, r["path"]) for r in manifest["records"]}
    maps = []
    for freq in FREQS_GHZ:
        maps.append(np.load(by_freq[round(freq, 2)]).astype(np.float32))
    return np.stack(maps, axis=-1)


def choose_source_mask(manifest: dict) -> np.ndarray | None:
    source = manifest_asset_path(manifest, manifest["source_coverage_dir"])
    for name in ("B_FOOTPRINT_GEOM_1024m_4m.npy", "B_MAP_GEOM_1024m_4m.npy"):
        path = source / name
        if path.exists():
            return np.load(path).astype(bool)
    return None


def infer_building_mask(maps_db: np.ndarray, manifest: dict, policy: str) -> np.ndarray:
    source = choose_source_mask(manifest)
    if policy == "source" and source is not None:
        return source
    floor_all = np.all(maps_db <= -119.95, axis=-1)
    if policy == "derived_all_floor":
        return floor_all
    if policy == "source_or_derived":
        return source if source is not None else floor_all
    raise ValueError(f"Unknown mask policy: {policy}")


def build_superpixels(maps_db: np.ndarray, building: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    valid = ~building
    mean_map = np.nanmean(rss_to_norm(maps_db), axis=-1)
    h, w = mean_map.shape
    yy, xx = np.meshgrid(
        np.linspace(0, 1, h, dtype=np.float32),
        np.linspace(0, 1, w, dtype=np.float32),
        indexing="ij",
    )
    image = np.stack([mean_map, xx, yy], axis=-1)
    labels = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        mask=valid,
        channel_axis=-1,
    ).astype(np.int32)
    labels[building] = 0
    labels, _, _ = relabel_sequential(labels)
    return labels.astype(np.int32)


def region_means(values: np.ndarray, labels: np.ndarray, num_nodes: int, nonzero: bool = False) -> np.ndarray:
    flat_l = labels.reshape(-1)
    valid = flat_l > 0
    lab = flat_l[valid] - 1
    if values.ndim == 2:
        vals = values.reshape(-1)[valid].astype(np.float64)
        if nonzero:
            keep = vals != 0
            lab = lab[keep]
            vals = vals[keep]
        sums = np.bincount(lab, weights=vals, minlength=num_nodes)
        counts = np.bincount(lab, minlength=num_nodes)
        out = np.zeros(num_nodes, dtype=np.float32)
        ok = counts > 0
        out[ok] = sums[ok] / counts[ok]
        return out[:, None]

    c = values.shape[-1]
    out = np.zeros((num_nodes, c), dtype=np.float32)
    for fi in range(c):
        vals = values[..., fi].reshape(-1)[valid].astype(np.float64)
        lab_f = lab
        if nonzero:
            keep = vals != 0
            lab_f = lab[keep]
            vals = vals[keep]
        sums = np.bincount(lab_f, weights=vals, minlength=num_nodes)
        counts = np.bincount(lab_f, minlength=num_nodes)
        ok = counts > 0
        out[ok, fi] = sums[ok] / counts[ok]
    return out


def centroids(labels: np.ndarray, num_nodes: int) -> np.ndarray:
    rows, cols = np.nonzero(labels > 0)
    lab = labels[rows, cols] - 1
    sum_r = np.bincount(lab, weights=rows, minlength=num_nodes)
    sum_c = np.bincount(lab, weights=cols, minlength=num_nodes)
    counts = np.bincount(lab, minlength=num_nodes).clip(min=1)
    return np.stack([sum_c / counts, sum_r / counts], axis=1).astype(np.float32)


def bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def has_los(c0: np.ndarray, c1: np.ndarray, building: np.ndarray) -> bool:
    h, w = building.shape
    x0, y0 = int(round(c0[0])), int(round(c0[1]))
    x1, y1 = int(round(c1[0])), int(round(c1[1]))
    for x, y in bresenham(x0, y0, x1, y1):
        if 0 <= x < w and 0 <= y < h and building[y, x]:
            return False
    return True


def build_edges(labels: np.ndarray, centers: np.ndarray, building: np.ndarray, dist_th: float) -> np.ndarray:
    num_nodes = centers.shape[0]
    edges: set[tuple[int, int]] = set((i, i) for i in range(num_nodes))

    # Boundary adjacency between touching superpixels.
    right_a = labels[:, :-1]
    right_b = labels[:, 1:]
    mask = (right_a > 0) & (right_b > 0) & (right_a != right_b)
    for a, b in zip(right_a[mask].ravel(), right_b[mask].ravel()):
        i, j = int(a) - 1, int(b) - 1
        edges.add((i, j))
        edges.add((j, i))
    down_a = labels[:-1, :]
    down_b = labels[1:, :]
    mask = (down_a > 0) & (down_b > 0) & (down_a != down_b)
    for a, b in zip(down_a[mask].ravel(), down_b[mask].ravel()):
        i, j = int(a) - 1, int(b) - 1
        edges.add((i, j))
        edges.add((j, i))

    # LOS edges within distance threshold, matching the MATLAB spirit.
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=dist_th)
    for i, j in pairs:
        if has_los(centers[i], centers[j], building):
            edges.add((i, j))
            edges.add((j, i))

    edge_arr = np.array(sorted(edges), dtype=np.int64).T
    return edge_arr


def idw_fill_sparse(sparse_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    obs = sparse_map != 0
    if not np.any(obs):
        return np.zeros_like(sparse_map, dtype=np.float32)
    h, w = sparse_map.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    ox = xx[obs].reshape(1, -1)
    oy = yy[obs].reshape(1, -1)
    ov = sparse_map[obs].reshape(1, -1)
    out = np.zeros((h, w), dtype=np.float32)
    for r0 in range(0, h, 16):
        r1 = min(h, r0 + 16)
        qx = xx[r0:r1].reshape(-1, 1)
        qy = yy[r0:r1].reshape(-1, 1)
        dist2 = (qx - ox) ** 2 + (qy - oy) ** 2
        weights = 1.0 / (dist2 + 1e-6)
        pred = (weights * ov).sum(axis=1) / weights.sum(axis=1)
        out[r0:r1] = pred.reshape(r1 - r0, w)
    out[~valid_mask] = 0
    out[obs] = sparse_map[obs]
    return out


def build_one_graph(manifest_path: Path, out_path: Path, args: argparse.Namespace) -> None:
    manifest = load_manifest(manifest_path)
    maps_db = load_maps(manifest)
    building = infer_building_mask(maps_db, manifest, args.mask_policy)
    free = ~building
    labels = build_superpixels(maps_db, building, args.superpixels, args.compactness)
    num_nodes = int(labels.max())
    if num_nodes < 10:
        raise RuntimeError(f"Too few superpixels for {manifest_path}: {num_nodes}")

    centers = centroids(labels, num_nodes)
    edge_index = build_edges(labels, centers, building, args.dist_th)

    gt_norm = region_means(rss_to_norm(maps_db), labels, num_nodes, nonzero=False)
    building_vec = region_means(building.astype(np.float32), labels, num_nodes, nonzero=False)
    x_vec = centers[:, 0:1] / maps_db.shape[1]
    y_vec = centers[:, 1:2] / maps_db.shape[0]
    tx_col = 128.0 + float(manifest["tx"][0]) / 4.0
    tx_row = 128.0 + float(manifest["tx"][1]) / 4.0
    txd = np.sqrt((centers[:, 0:1] - tx_col) ** 2 + (centers[:, 1:2] - tx_row) ** 2)
    txd = txd / math.sqrt(maps_db.shape[0] ** 2 + maps_db.shape[1] ** 2)

    rng = np.random.default_rng(args.seed + abs(hash(str(manifest_path))) % (2**31))
    sparse_train_vectors = np.zeros((num_nodes, len(FREQS_GHZ)), dtype=np.float32)
    obs_train_mask_vectors = np.zeros((num_nodes, len(FREQS_GHZ)), dtype=np.float32)
    idw_vectors = np.zeros((num_nodes, len(FREQS_GHZ)), dtype=np.float32)
    train_mask = np.zeros((num_nodes, len(FREQS_GHZ)), dtype=bool)
    val_mask = np.zeros((num_nodes, len(FREQS_GHZ)), dtype=bool)
    test_mask = np.zeros((num_nodes, len(FREQS_GHZ)), dtype=bool)

    valid_pixels = np.flatnonzero(free)
    flat_labels = labels.reshape(-1)
    for fi in range(len(FREQS_GHZ)):
        known_n = max(1, int(round(args.known_percent / 100.0 * valid_pixels.size)))
        known = rng.choice(valid_pixels, size=known_n, replace=False)
        obs_nodes = np.unique(flat_labels[known])
        obs_nodes = obs_nodes[obs_nodes > 0] - 1
        rng.shuffle(obs_nodes)
        split = int(round(args.train_ratio * obs_nodes.size))
        train_nodes = obs_nodes[:split]
        val_nodes = obs_nodes[split:]
        all_nodes = np.arange(num_nodes)
        observed_node_mask = np.zeros(num_nodes, dtype=bool)
        observed_node_mask[obs_nodes] = True
        train_mask[train_nodes, fi] = True
        val_mask[val_nodes, fi] = True
        test_mask[all_nodes[~observed_node_mask], fi] = True

        # Train sparse image contains only pixel observations whose superpixel is in train_nodes.
        train_node_set = np.zeros(num_nodes + 1, dtype=bool)
        train_node_set[train_nodes + 1] = True
        keep_pix = known[train_node_set[flat_labels[known]]]
        sparse_img = np.zeros(maps_db.shape[:2], dtype=np.float32)
        sparse_img.reshape(-1)[keep_pix] = rss_to_norm(maps_db[..., fi]).reshape(-1)[keep_pix]
        sparse_train_vectors[:, fi] = region_means(sparse_img, labels, num_nodes, nonzero=True).squeeze(1)
        obs_train_mask_vectors[:, fi] = sparse_train_vectors[:, fi] != 0
        idw_img = idw_fill_sparse(sparse_img, free)
        idw_vectors[:, fi] = region_means(idw_img, labels, num_nodes, nonzero=False).squeeze(1)

    feature_matrix = np.concatenate(
        [
            sparse_train_vectors,
            obs_train_mask_vectors,
            x_vec.astype(np.float32),
            y_vec.astype(np.float32),
            txd.astype(np.float32),
            building_vec.astype(np.float32),
            idw_vectors,
        ],
        axis=1,
    ).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        feature_matrix=feature_matrix,
        matrix_gt=gt_norm.astype(np.float32),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        edge_index=edge_index,
        idw_vectors=idw_vectors.astype(np.float32),
        labels=labels.astype(np.int32),
        centroids=centers.astype(np.float32),
        freqs_ghz=np.array(FREQS_GHZ, dtype=np.float32),
        manifest_path=str(manifest_path),
        city=manifest["city"],
        xml_idx=int(manifest["xml_idx"]),
        tx_idx=int(manifest["tx_idx"]),
        tx=np.array(manifest["tx"], dtype=np.float32),
        known_percent=float(args.known_percent),
        superpixels=int(num_nodes),
    )


def output_path(out_root: Path, manifest_path: Path, args: argparse.Namespace) -> Path:
    manifest = load_manifest(manifest_path)
    city = manifest["city"]
    xml_part = manifest_path.parent.parent.name
    tx_part = manifest_path.parent.name
    name = f"{city}_{xml_part}_{tx_part}_sp{args.superpixels}_kp{args.known_percent:g}.npz"
    return out_root / city / f"sp{args.superpixels}_kp{args.known_percent:g}" / name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cities", default="Davis,Woodland")
    parser.add_argument("--max-graphs", type=int, default=16)
    parser.add_argument("--superpixels", type=int, default=2000)
    parser.add_argument("--known-percent", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dist-th", type=float, default=30.0)
    parser.add_argument("--compactness", type=float, default=12.0)
    parser.add_argument("--mask-policy", choices=["source", "source_or_derived", "derived_all_floor"], default="derived_all_floor")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()

    random.seed(args.seed)
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    manifests = find_manifests(args.data_root, cities, completed_only=True)
    random.shuffle(manifests)
    if args.max_graphs > 0:
        manifests = manifests[: args.max_graphs]
    print(f"[build] manifests={len(manifests)} cities={cities} sp={args.superpixels} kp={args.known_percent}", flush=True)
    for i, manifest_path in enumerate(manifests, start=1):
        out_path = output_path(args.out_root, manifest_path, args)
        if args.skip_existing and out_path.exists():
            print(f"[{i}/{len(manifests)}] skip {out_path}", flush=True)
            continue
        print(f"[{i}/{len(manifests)}] build {manifest_path}", flush=True)
        build_one_graph(manifest_path, out_path, args)
        print(f"  -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
