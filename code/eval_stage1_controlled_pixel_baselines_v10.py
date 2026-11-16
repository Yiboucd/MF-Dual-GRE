#!/usr/bin/env python3
"""Native-grid baselines for the controlled cross-frequency protocol.

Prediction is performed directly from raw pixel observations. Superpixel labels
are read only to report the same regional metric as Stage 1; they are never used
to construct an interpolation neighborhood or prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from build_clean_superpixel_graphs import FREQS_GHZ
from build_clean_superpixel_graphs_v2 import masked_region_means, stable_seed
from controlled_multifrequency_protocol_v10 import (
    NF,
    city_of,
    load_manifest_maps,
    make_context_masks,
    make_fixed_test_mask,
    scalar_text,
    select_paths,
    shuffled_pairs,
    validate_rates,
)


DB_SCALE = 40.0
DB_FLOOR = -120.0


def split_context_calibration(
    context_mask: np.ndarray,
    fraction: float,
    mask_seed: int,
    sample_key: str,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split only already-budgeted target observations for model selection."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    active = np.flatnonzero(context_mask)
    if active.size < 3:
        raise RuntimeError(f"Too few target observations for {sample_key}")
    rng = np.random.default_rng(
        stable_seed(mask_seed, sample_key, target_index, "pixel-cal-v10")
    )
    count = min(active.size - 2, max(1, int(round(fraction * active.size))))
    calibration_indices = rng.permutation(active)[:count]
    calibration = np.zeros(context_mask.size, dtype=bool)
    calibration[calibration_indices] = True
    calibration = calibration.reshape(context_mask.shape)
    fit = context_mask & ~calibration
    return fit, calibration


def norm_to_db(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float64) * DB_SCALE + DB_FLOOR


def query_idw(
    observed_coords: np.ndarray,
    observed_values: np.ndarray,
    query_coords: np.ndarray,
    neighbors: int,
    power: float,
) -> np.ndarray:
    """Interpolate query pixels from their nearest raw-pixel observations."""
    if observed_coords.shape[0] == 0:
        raise ValueError("IDW requires at least one observed pixel")
    if query_coords.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    k = min(max(1, int(neighbors)), observed_coords.shape[0])
    tree = cKDTree(observed_coords.astype(np.float64))
    distances, indices = tree.query(query_coords.astype(np.float64), k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    exact = distances[:, 0] <= 1e-12
    weights = 1.0 / np.maximum(distances, 1e-6) ** float(power)
    predictions = np.sum(weights * observed_values[indices], axis=1) / np.sum(
        weights, axis=1
    )
    if np.any(exact):
        predictions[exact] = observed_values[indices[exact, 0]]
    return predictions.astype(np.float64)


def query_ordinary_kriging(
    observed_coords: np.ndarray,
    observed_values: np.ndarray,
    query_coords: np.ndarray,
    neighbors: int,
    range_pixels: float,
    nugget: float,
    batch_size: int,
) -> np.ndarray:
    """Local ordinary kriging with a fixed exponential covariance model."""
    if observed_coords.shape[0] == 0:
        raise ValueError("Ordinary kriging requires at least one observed pixel")
    if query_coords.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    k = min(max(2, int(neighbors)), observed_coords.shape[0])
    if k == 1:
        return np.full(query_coords.shape[0], observed_values[0], dtype=np.float64)

    tree = cKDTree(observed_coords.astype(np.float64))
    query_distances, neighbor_indices = tree.query(
        query_coords.astype(np.float64), k=k
    )
    output = np.empty(query_coords.shape[0], dtype=np.float64)
    range_pixels = max(float(range_pixels), 1e-3)

    for start in range(0, query_coords.shape[0], max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), query_coords.shape[0])
        indices = neighbor_indices[start:stop]
        local_coords = observed_coords[indices].astype(np.float64)
        pair_delta = local_coords[:, :, None, :] - local_coords[:, None, :, :]
        pair_distance = np.sqrt(np.sum(pair_delta * pair_delta, axis=-1))
        covariance = np.exp(-pair_distance / range_pixels)
        covariance[:, np.arange(k), np.arange(k)] += float(nugget)

        system = np.zeros((stop - start, k + 1, k + 1), dtype=np.float64)
        system[:, :k, :k] = covariance
        system[:, :k, k] = 1.0
        system[:, k, :k] = 1.0
        rhs = np.zeros((stop - start, k + 1), dtype=np.float64)
        rhs[:, :k] = np.exp(-query_distances[start:stop] / range_pixels)
        rhs[:, k] = 1.0
        try:
            weights = np.linalg.solve(system, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            weights = (np.linalg.pinv(system, rcond=1e-8) @ rhs[..., None])[..., 0]
        output[start:stop] = np.sum(
            weights[:, :k] * observed_values[indices], axis=1
        )

    exact = query_distances[:, 0] <= 1e-12
    if np.any(exact):
        output[exact] = observed_values[neighbor_indices[exact, 0]]
    return output


def tensor_svd_completion(
    maps_db: np.ndarray,
    context_masks: np.ndarray,
    free: np.ndarray,
    neighbors: int,
    power: float,
    rank: int,
    iterations: int,
) -> np.ndarray:
    """Complete the raw pixel-by-frequency matrix with hard observations.

    Each active frequency is initialized independently by native-grid IDW.
    Alternating truncated SVD then models only cross-frequency structure; no
    superpixel labels or graph edges enter the prediction.
    """

    free_coords = np.argwhere(free)
    observed = context_masks[free]
    active_frequencies = np.flatnonzero(observed.any(axis=0))
    if active_frequencies.size == 0:
        raise ValueError("Tensor-SVD requires at least one active frequency")

    values = maps_db[free][:, active_frequencies].astype(np.float64)
    observed = observed[:, active_frequencies]
    estimate = np.empty_like(values)
    for local_index, frequency_index in enumerate(active_frequencies):
        frequency_mask = context_masks[..., frequency_index]
        observed_coords = np.argwhere(frequency_mask)
        observed_values = maps_db[..., frequency_index][frequency_mask]
        estimate[:, local_index] = query_idw(
            observed_coords,
            observed_values,
            free_coords,
            neighbors,
            power,
        )
        estimate[observed[:, local_index], local_index] = values[
            observed[:, local_index], local_index
        ]

    if active_frequencies.size > 1:
        effective_rank = min(
            max(1, int(rank)),
            max(1, active_frequencies.size - 1),
        )
        for _ in range(max(1, int(iterations))):
            column_mean = estimate.mean(axis=0, keepdims=True)
            centered = estimate - column_mean
            left, singular_values, right = np.linalg.svd(
                centered, full_matrices=False
            )
            reconstructed = (
                left[:, :effective_rank] * singular_values[:effective_rank]
            ) @ right[:effective_rank]
            estimate = reconstructed + column_mean
            estimate[observed] = values[observed]

    completed = np.full(maps_db.shape, np.nan, dtype=np.float64)
    for local_index, frequency_index in enumerate(active_frequencies):
        channel = completed[..., frequency_index]
        channel[free] = estimate[:, local_index]
    return completed


def fit_affine(source: np.ndarray, target: np.ndarray, ridge: float) -> tuple[float, float]:
    if source.size < 2 or float(np.std(source)) < 1e-8:
        return 1.0, float(np.mean(target - source))
    design = np.column_stack([source, np.ones_like(source)])
    regularizer = np.diag([float(ridge), 0.0])
    coefficients = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ target,
    )
    return float(coefficients[0]), float(coefficients[1])


def affine_auxiliary_prediction(
    maps_db: np.ndarray,
    context_masks: np.ndarray,
    target_index: int,
    target_query_coords: np.ndarray,
    query_coords: np.ndarray,
    target_values: np.ndarray,
    neighbors: int,
    power: float,
    ridge: float,
) -> tuple[np.ndarray | None, dict[str, object]]:
    transferred = []
    calibration_errors = []
    coefficients: dict[str, list[float]] = {}
    for fi in range(NF):
        if fi == target_index or not np.any(context_masks[..., fi]):
            continue
        observed_coords = np.argwhere(context_masks[..., fi])
        observed_values = maps_db[..., fi][context_masks[..., fi]]
        auxiliary_at_target = query_idw(
            observed_coords,
            observed_values,
            target_query_coords,
            neighbors,
            power,
        )
        slope, intercept = fit_affine(auxiliary_at_target, target_values, ridge)
        auxiliary_at_query = query_idw(
            observed_coords,
            observed_values,
            query_coords,
            neighbors,
            power,
        )
        transferred.append(slope * auxiliary_at_query + intercept)
        residual = slope * auxiliary_at_target + intercept - target_values
        calibration_errors.append(float(np.mean(residual * residual)))
        coefficients[f"{FREQS_GHZ[fi]:g}"] = [slope, intercept]

    if not transferred:
        return None, {"coefficients": {}, "auxiliary_weights": {}}
    errors = np.asarray(calibration_errors, dtype=np.float64)
    weights = 1.0 / np.maximum(errors, 1e-4)
    weights /= weights.sum()
    stacked = np.stack(transferred, axis=0)
    prediction = np.sum(weights[:, None] * stacked, axis=0)
    active_frequencies = list(coefficients)
    return prediction, {
        "coefficients": coefficients,
        "auxiliary_weights": {
            frequency: float(weight)
            for frequency, weight in zip(active_frequencies, weights)
        },
    }


def regional_sse(
    prediction: np.ndarray,
    target: np.ndarray,
    query_mask: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, int]:
    num_nodes = int(labels.max())
    prediction_map = np.zeros(labels.shape, dtype=np.float32)
    target_map = np.zeros(labels.shape, dtype=np.float32)
    prediction_map[query_mask] = prediction.astype(np.float32)
    target_map[query_mask] = target.astype(np.float32)
    prediction_mean, counts = masked_region_means(
        prediction_map, query_mask, labels, num_nodes
    )
    target_mean, _ = masked_region_means(target_map, query_mask, labels, num_nodes)
    active = counts > 0
    error = prediction_mean[active].astype(np.float64) - target_mean[active].astype(
        np.float64
    )
    return float(np.sum(error * error)), int(np.sum(active))


def add_score(
    accumulator: dict[str, object],
    prediction: np.ndarray,
    target: np.ndarray,
    query_mask: np.ndarray,
    labels: np.ndarray,
    frequency_index: int,
) -> None:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    accumulator["pixel_sse"][frequency_index] += float(np.sum(error * error))
    accumulator["pixel_count"][frequency_index] += int(error.size)

    true_path = target.astype(np.float64) > DB_FLOOR + 1e-4
    predicted_path = prediction.astype(np.float64) > -119.5
    nonfloor_error = error[true_path]
    accumulator["nonfloor_pixel_sse"][frequency_index] += float(
        np.sum(nonfloor_error * nonfloor_error)
    )
    accumulator["nonfloor_pixel_count"][frequency_index] += int(
        nonfloor_error.size
    )
    accumulator["path_tp"][frequency_index] += int(
        np.sum(predicted_path & true_path)
    )
    accumulator["path_tn"][frequency_index] += int(
        np.sum(~predicted_path & ~true_path)
    )
    accumulator["path_fp"][frequency_index] += int(
        np.sum(predicted_path & ~true_path)
    )
    accumulator["path_fn"][frequency_index] += int(
        np.sum(~predicted_path & true_path)
    )

    score, count = regional_sse(prediction, target, query_mask, labels)
    accumulator["regional_sse"][frequency_index] += score
    accumulator["regional_count"][frequency_index] += count


def summarize(accumulator: dict[str, object], target_indices: list[int]) -> dict[str, object]:
    pixel_sse = accumulator["pixel_sse"]
    pixel_count = accumulator["pixel_count"]
    regional_sse_values = accumulator["regional_sse"]
    regional_count = accumulator["regional_count"]
    nonfloor_pixel_sse = accumulator["nonfloor_pixel_sse"]
    nonfloor_pixel_count = accumulator["nonfloor_pixel_count"]

    def path_scores(indices: list[int]) -> dict[str, float]:
        tp = float(sum(accumulator["path_tp"][fi] for fi in indices))
        tn = float(sum(accumulator["path_tn"][fi] for fi in indices))
        fp = float(sum(accumulator["path_fp"][fi] for fi in indices))
        fn = float(sum(accumulator["path_fn"][fi] for fi in indices))
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

    def per_frequency(sse: np.ndarray, counts: np.ndarray) -> dict[str, float]:
        return {
            f"{FREQS_GHZ[fi]:g}": float(math.sqrt(sse[fi] / max(counts[fi], 1)))
            for fi in target_indices
        }

    return {
        "pixel_rmse_db": float(
            math.sqrt(
                sum(pixel_sse[fi] for fi in target_indices)
                / max(sum(pixel_count[fi] for fi in target_indices), 1)
            )
        ),
        "regional_rmse_db": float(
            math.sqrt(
                sum(regional_sse_values[fi] for fi in target_indices)
                / max(sum(regional_count[fi] for fi in target_indices), 1)
            )
        ),
        "nonfloor_pixel_rmse_db": float(
            math.sqrt(
                sum(nonfloor_pixel_sse[fi] for fi in target_indices)
                / max(sum(nonfloor_pixel_count[fi] for fi in target_indices), 1)
            )
        ),
        "pixel_rmse_db_per_target_frequency": per_frequency(
            pixel_sse, pixel_count
        ),
        "regional_rmse_db_per_target_frequency": per_frequency(
            regional_sse_values, regional_count
        ),
        "nonfloor_pixel_rmse_db_per_target_frequency": per_frequency(
            nonfloor_pixel_sse, nonfloor_pixel_count
        ),
        "path_existence": path_scores(target_indices),
        "path_existence_per_target_frequency": {
            f"{FREQS_GHZ[fi]:g}": path_scores([fi]) for fi in target_indices
        },
        "pixel_counts_per_target_frequency": {
            f"{FREQS_GHZ[fi]:g}": int(pixel_count[fi]) for fi in target_indices
        },
        "nonfloor_pixel_counts_per_target_frequency": {
            f"{FREQS_GHZ[fi]:g}": int(nonfloor_pixel_count[fi])
            for fi in target_indices
        },
    }


def new_accumulator() -> dict[str, object]:
    return {
        "pixel_sse": np.zeros(NF, dtype=np.float64),
        "pixel_count": np.zeros(NF, dtype=np.int64),
        "regional_sse": np.zeros(NF, dtype=np.float64),
        "regional_count": np.zeros(NF, dtype=np.int64),
        "nonfloor_pixel_sse": np.zeros(NF, dtype=np.float64),
        "nonfloor_pixel_count": np.zeros(NF, dtype=np.int64),
        "path_tp": np.zeros(NF, dtype=np.int64),
        "path_tn": np.zeros(NF, dtype=np.int64),
        "path_fp": np.zeros(NF, dtype=np.int64),
        "path_fn": np.zeros(NF, dtype=np.int64),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    rates = validate_rates(args.rates)
    cities = [item.strip() for item in args.cities.split(",") if item.strip()]
    paths = select_paths(args.graph_root, cities, args.max_graphs, args.selection_seed)
    if args.auxiliary_mode == "shuffled":
        auxiliary_pool = select_paths(args.graph_root, cities, 0, args.selection_seed)
        pairs = shuffled_pairs(paths, auxiliary_pool, args.shuffle_offset)
    else:
        pairs = {path: path for path in paths}

    labels_by_key = {
        "idw": "2D-IDW",
        "kriging": "Ordinary Kriging",
        "affine": "Affine-transfer IDW",
        "tensor": "Tensor-SVD",
    }
    methods = [labels_by_key[key] for key in args.methods]
    stratum_names = (
        "all",
        "same_frequency_context",
        "no_same_frequency_context",
        "auxiliary_only_context",
        "no_context_any_frequency",
    )
    accumulators = {
        method: {name: new_accumulator() for name in stratum_names}
        for method in methods
    }
    elapsed = {method: 0.0 for method in methods}
    selected_alphas: list[float] = []
    calibration_examples: list[dict[str, object]] = []

    for example_index, path in enumerate(paths, start=1):
        with np.load(path, allow_pickle=False) as data:
            labels = data["labels"].astype(np.int32)
            building = data["building_mask"].astype(bool)
            city = scalar_text(data["city"])
            xml_idx = int(np.asarray(data["xml_idx"]).item())
            tx_idx = int(np.asarray(data["tx_idx"]).item())
        _, _, target_maps_norm = load_manifest_maps(path)
        auxiliary_maps_norm = target_maps_norm
        if args.auxiliary_mode == "shuffled":
            _, _, auxiliary_maps_norm = load_manifest_maps(pairs[path])
        target_maps_db = norm_to_db(target_maps_norm)

        free = ~building
        sample_key = f"{city}|xml={xml_idx}|tx={tx_idx}"
        fixed_test_mask = (
            make_fixed_test_mask(
                free,
                args.fixed_test_seed,
                sample_key,
                args.fixed_test_rate_percent,
            )
            if args.fixed_test_rate_percent > 0
            else None
        )
        for target_index in args.target_indices:
            maps_db = target_maps_db.copy()
            if args.auxiliary_mode == "shuffled":
                for fi in range(NF):
                    if fi != target_index:
                        maps_db[..., fi] = norm_to_db(auxiliary_maps_norm[..., fi])
            context_masks = make_context_masks(
                free,
                rates,
                target_index,
                args.mask_mode,
                args.mask_seed,
                sample_key,
                args.sampling_mode,
                fixed_test_mask,
            )
            if args.measurement_noise_db > 0:
                for fi in range(NF):
                    active = context_masks[..., fi]
                    rng = np.random.default_rng(
                        stable_seed(
                            args.mask_seed,
                            sample_key,
                            fi,
                            args.sampling_mode,
                            args.measurement_noise_db,
                            "measurement-noise-v10",
                        )
                    )
                    maps_db[..., fi][active] += rng.normal(
                        0.0, args.measurement_noise_db, int(active.sum())
                    )
            test_mask = (
                free & fixed_test_mask
                if fixed_test_mask is not None
                else free & ~context_masks[..., target_index]
            )
            num_nodes = int(labels.max())
            node_context = np.zeros((num_nodes, NF), dtype=bool)
            for fi in range(NF):
                active_labels = labels[context_masks[..., fi]]
                active_labels = active_labels[active_labels > 0] - 1
                if active_labels.size:
                    node_context[:, fi] = (
                        np.bincount(active_labels, minlength=num_nodes) > 0
                    )
            test_node_indices = labels[test_mask] - 1
            if np.any(test_node_indices < 0):
                raise RuntimeError(
                    f"Test mask contains a non-region pixel for {sample_key}"
                )
            same_frequency_context = node_context[
                test_node_indices, target_index
            ]
            any_frequency_context = node_context[test_node_indices].any(axis=1)
            stratum_selectors = {
                "all": np.ones(test_node_indices.shape[0], dtype=bool),
                "same_frequency_context": same_frequency_context,
                "no_same_frequency_context": ~same_frequency_context,
                "auxiliary_only_context": (
                    ~same_frequency_context & any_frequency_context
                ),
                "no_context_any_frequency": ~any_frequency_context,
            }
            stratum_query_masks = {}
            for name, selector in stratum_selectors.items():
                query_mask = np.zeros_like(test_mask)
                query_mask[test_mask] = selector
                stratum_query_masks[name] = query_mask

            def add_all_strata(method: str, prediction: np.ndarray) -> None:
                for name, selector in stratum_selectors.items():
                    add_score(
                        accumulators[method][name],
                        prediction[selector],
                        test_target[selector],
                        stratum_query_masks[name],
                        labels,
                        target_index,
                    )

            target_context = context_masks[..., target_index]
            if not np.any(target_context):
                raise RuntimeError(
                    f"Target-only baselines need target observations for {sample_key}"
                )
            observed_coords = np.argwhere(target_context)
            observed_values = maps_db[..., target_index][target_context]
            test_coords = np.argwhere(test_mask)
            test_target = target_maps_db[..., target_index][test_mask]

            idw_started = time.perf_counter()
            idw_test = None
            if "idw" in args.methods or "affine" in args.methods:
                idw_test = query_idw(
                    observed_coords,
                    observed_values,
                    test_coords,
                    args.idw_neighbors,
                    args.idw_power,
                )
            idw_seconds = time.perf_counter() - idw_started
            if "idw" in args.methods:
                elapsed["2D-IDW"] += idw_seconds
                add_all_strata("2D-IDW", idw_test)

            if "kriging" in args.methods:
                started = time.perf_counter()
                kriging_test = query_ordinary_kriging(
                    observed_coords,
                    observed_values,
                    test_coords,
                    args.kriging_neighbors,
                    args.kriging_range,
                    args.kriging_nugget,
                    args.kriging_batch_size,
                )
                elapsed["Ordinary Kriging"] += time.perf_counter() - started
                add_all_strata("Ordinary Kriging", kriging_test)

            if "affine" in args.methods:
                started = time.perf_counter()
                fit_mask, calibration_mask = split_context_calibration(
                    target_context,
                    args.calibration_fraction,
                    args.mask_seed,
                    sample_key,
                    target_index,
                )
                fit_coords = np.argwhere(fit_mask)
                fit_values = maps_db[..., target_index][fit_mask]
                calibration_coords = np.argwhere(calibration_mask)
                calibration_target = maps_db[..., target_index][calibration_mask]
                idw_calibration = query_idw(
                    fit_coords,
                    fit_values,
                    calibration_coords,
                    args.idw_neighbors,
                    args.idw_power,
                )
                auxiliary_calibration, _ = affine_auxiliary_prediction(
                    maps_db,
                    context_masks,
                    target_index,
                    fit_coords,
                    calibration_coords,
                    fit_values,
                    args.idw_neighbors,
                    args.idw_power,
                    args.affine_ridge,
                )
                auxiliary_test, calibration = affine_auxiliary_prediction(
                    maps_db,
                    context_masks,
                    target_index,
                    observed_coords,
                    test_coords,
                    observed_values,
                    args.idw_neighbors,
                    args.idw_power,
                    args.affine_ridge,
                )
                if auxiliary_calibration is None or auxiliary_test is None:
                    affine_test = idw_test
                    best_alpha = 0.0
                else:
                    candidate_alphas = np.linspace(0.0, 1.0, args.alpha_steps)
                    validation_errors = [
                        np.mean(
                            (
                                (1.0 - alpha) * idw_calibration
                                + alpha * auxiliary_calibration
                                - calibration_target
                            )
                            ** 2
                        )
                        for alpha in candidate_alphas
                    ]
                    best_alpha = float(
                        candidate_alphas[int(np.argmin(validation_errors))]
                    )
                    affine_test = (
                        (1.0 - best_alpha) * idw_test + best_alpha * auxiliary_test
                    )
                elapsed["Affine-transfer IDW"] += idw_seconds + (
                    time.perf_counter() - started
                )
                selected_alphas.append(best_alpha)
                if len(calibration_examples) < 10:
                    calibration_examples.append(
                        {
                            "sample_key": sample_key,
                            "target_frequency_ghz": float(FREQS_GHZ[target_index]),
                            "selected_auxiliary_mix": best_alpha,
                            **calibration,
                        }
                    )
                add_all_strata("Affine-transfer IDW", affine_test)

            if "tensor" in args.methods:
                started = time.perf_counter()
                completed = tensor_svd_completion(
                    maps_db,
                    context_masks,
                    free,
                    args.idw_neighbors,
                    args.idw_power,
                    args.tensor_rank,
                    args.tensor_iterations,
                )
                tensor_test = completed[..., target_index][test_mask]
                if not np.all(np.isfinite(tensor_test)):
                    raise RuntimeError(
                        f"Tensor-SVD did not reconstruct target band for {sample_key}"
                    )
                elapsed["Tensor-SVD"] += time.perf_counter() - started
                add_all_strata("Tensor-SVD", tensor_test)

        if example_index == 1 or example_index % 25 == 0 or example_index == len(paths):
            print(f"[controlled pixel v6] {example_index}/{len(paths)} {path}", flush=True)

    results = {
        method: {
            **summarize(accumulators[method]["all"], args.target_indices),
            "strata": {
                name: summarize(accumulators[method][name], args.target_indices)
                for name in stratum_names
            },
            "elapsed_seconds": float(elapsed[method]),
            "mean_seconds_per_map_target": float(
                elapsed[method] / max(len(paths) * len(args.target_indices), 1)
            ),
        }
        for method in methods
    }
    return {
        "protocol": (
            "native_pixel_budget_internal_fixed_pixel_holdout_v11"
            if args.fixed_test_rate_percent > 0
            else "native_pixel_budget_internal_calibration_v10"
        ),
        "prediction_grid": "native_pixel_grid",
        "uses_graph_for_prediction": False,
        "uses_superpixels_for_prediction": False,
        "regional_metric_note": (
            "labels are used only after native-grid prediction; regional RMSE "
            "is the unweighted error over active region-frequency means"
        ),
        "tensor_svd_note": (
            "native-grid per-frequency IDW initialization followed by iterative "
            "low-rank completion of the free-pixel-by-frequency matrix; observed "
            "entries are restored after every iteration"
        ),
        "path_existence_definition": {
            "ground_truth": "path gain above the -120 dB no-path floor",
            "prediction_threshold_db": -119.5,
        },
        "strata_definition": {
            "same_frequency_context": "test pixels whose region contains a target-band context sample",
            "auxiliary_only_context": "test pixels whose region contains only auxiliary-band context",
            "no_context_any_frequency": "test pixels whose region has no context sample at any frequency",
        },
        "rates_percent": rates,
        "total_rate_budget_percent": float(sum(rates)),
        "target_indices": args.target_indices,
        "target_frequencies_ghz": [
            float(FREQS_GHZ[fi]) for fi in args.target_indices
        ],
        "mask_mode": args.mask_mode,
        "auxiliary_mode": args.auxiliary_mode,
        "auxiliary_shuffle_policy": (
            "same_xml_different_tx" if args.auxiliary_mode == "shuffled" else "none"
        ),
        "sampling_mode": args.sampling_mode,
        "measurement_noise_db": args.measurement_noise_db,
        "mask_seed": args.mask_seed,
        "selection_seed": args.selection_seed,
        "fixed_test_rate_percent": args.fixed_test_rate_percent,
        "fixed_test_seed": args.fixed_test_seed,
        "test_pixel_policy": (
            "fixed_all_band_excluded_holdout_v11"
            if args.fixed_test_rate_percent > 0
            else "all_unobserved_pixels_v10"
        ),
        "calibration_fraction_within_observation_budget": args.calibration_fraction,
        "test_definition": (
            "deterministic free-space pixel holdout excluded from every carrier observation pool"
            if args.fixed_test_rate_percent > 0
            else "all free-space pixels absent from the target-band observation mask"
        ),
        "num_examples": len(paths),
        "selected_auxiliary_mix_mean": (
            float(np.mean(selected_alphas)) if selected_alphas else None
        ),
        "calibration_examples": calibration_examples,
        "results": results,
        "args": vars(args),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs=NF, required=True)
    parser.add_argument("--target-indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--methods",
        choices=["idw", "kriging", "affine", "tensor"],
        nargs="+",
        default=["idw", "kriging", "affine", "tensor"],
    )
    parser.add_argument("--mask-mode", choices=["independent", "aligned"], required=True)
    parser.add_argument("--auxiliary-mode", choices=["true", "shuffled"], required=True)
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "clustered", "trajectory"],
        default="random",
    )
    parser.add_argument("--measurement-noise-db", type=float, default=0.0)
    parser.add_argument("--fixed-test-rate-percent", type=float, default=0.0)
    parser.add_argument("--fixed-test-seed", type=int, default=9001)
    parser.add_argument("--cities", default="Davis,Woodland,Vacaville")
    parser.add_argument("--max-graphs", type=int, default=480)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--idw-neighbors", type=int, default=32)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--kriging-neighbors", type=int, default=16)
    parser.add_argument("--kriging-range", type=float, default=64.0)
    parser.add_argument("--kriging-nugget", type=float, default=1e-4)
    parser.add_argument("--kriging-batch-size", type=int, default=512)
    parser.add_argument("--affine-ridge", type=float, default=1e-3)
    parser.add_argument("--alpha-steps", type=int, default=11)
    parser.add_argument("--tensor-rank", type=int, default=3)
    parser.add_argument("--tensor-iterations", type=int, default=10)
    args = parser.parse_args()
    args.target_indices = [int(fi) for fi in args.target_indices]
    if any(fi < 0 or fi >= NF for fi in args.target_indices):
        parser.error(f"target indices must lie in [0, {NF - 1}]")
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("calibration fraction must lie in (0, 1)")
    if args.measurement_noise_db < 0:
        parser.error("measurement noise must be nonnegative")
    if args.fixed_test_rate_percent < 0 or args.fixed_test_rate_percent >= 100:
        parser.error("fixed-test rate must lie in [0, 100)")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(args)
    serializable = json.loads(json.dumps(metrics, default=str))
    (args.out_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    print(json.dumps(serializable, indent=2), flush=True)


if __name__ == "__main__":
    main()
