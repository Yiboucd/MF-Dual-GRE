#!/usr/bin/env python3
"""Shared dense-pixel metrics for clean multi-frequency radiomap experiments."""

from __future__ import annotations

import numpy as np
import torch


class PixelMetricAccumulator:
    """Accumulate exact per-band metrics without storing full prediction maps."""

    def __init__(
        self,
        num_frequencies: int,
        db_span: float = 40.0,
        floor_db: float = -120.0,
        path_threshold_db: float = -119.5,
    ) -> None:
        self.num_frequencies = int(num_frequencies)
        self.db_span = float(db_span)
        self.floor_db = float(floor_db)
        self.path_threshold_norm = (
            float(path_threshold_db) - self.floor_db
        ) / self.db_span
        shape = (self.num_frequencies,)
        self.all_sse = np.zeros(shape, dtype=np.float64)
        self.unseen_sse = np.zeros(shape, dtype=np.float64)
        self.nonfloor_unseen_sse = np.zeros(shape, dtype=np.float64)
        self.all_count = np.zeros(shape, dtype=np.float64)
        self.unseen_count = np.zeros(shape, dtype=np.float64)
        self.nonfloor_unseen_count = np.zeros(shape, dtype=np.float64)
        self.path_tp = np.zeros(shape, dtype=np.float64)
        self.path_tn = np.zeros(shape, dtype=np.float64)
        self.path_fp = np.zeros(shape, dtype=np.float64)
        self.path_fn = np.zeros(shape, dtype=np.float64)

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        observation_mask: torch.Tensor | None,
        evaluation_mask: torch.Tensor | None = None,
    ) -> None:
        if valid.ndim == 4:
            valid = valid[:, 0]
        if valid.ndim != 3:
            raise ValueError(f"Expected BxHxW valid mask, got {tuple(valid.shape)}")
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        if prediction.shape[1] != self.num_frequencies:
            raise ValueError(tuple(prediction.shape))
        if evaluation_mask is not None and evaluation_mask.shape != prediction.shape:
            raise ValueError(
                f"Evaluation-mask mismatch: {tuple(evaluation_mask.shape)} vs "
                f"{tuple(prediction.shape)}"
            )

        prediction = prediction.clamp(0.0, 1.0)
        error_db = (prediction - target) * self.db_span
        for frequency in range(self.num_frequencies):
            if evaluation_mask is not None:
                unseen = valid & evaluation_mask[:, frequency].bool()
            else:
                unseen = valid
            if evaluation_mask is None and observation_mask is not None:
                unseen = unseen & (~observation_mask[:, frequency].bool())
            nonfloor = unseen & (target[:, frequency] > 1e-6)
            error = error_db[:, frequency]
            self.all_sse[frequency] += float(error[valid].square().sum().cpu())
            self.unseen_sse[frequency] += float(error[unseen].square().sum().cpu())
            self.nonfloor_unseen_sse[frequency] += float(
                error[nonfloor].square().sum().cpu()
            )
            self.all_count[frequency] += int(valid.sum().cpu())
            self.unseen_count[frequency] += int(unseen.sum().cpu())
            self.nonfloor_unseen_count[frequency] += int(nonfloor.sum().cpu())

            target_path = target[:, frequency] > 1e-6
            predicted_path = prediction[:, frequency] > self.path_threshold_norm
            self.path_tp[frequency] += int(
                (unseen & target_path & predicted_path).sum().cpu()
            )
            self.path_tn[frequency] += int(
                (unseen & ~target_path & ~predicted_path).sum().cpu()
            )
            self.path_fp[frequency] += int(
                (unseen & ~target_path & predicted_path).sum().cpu()
            )
            self.path_fn[frequency] += int(
                (unseen & target_path & ~predicted_path).sum().cpu()
            )

    @staticmethod
    def _rmse(sse: np.ndarray, count: np.ndarray) -> tuple[float, list[float]]:
        per_frequency = np.sqrt(sse / np.maximum(count, 1.0))
        aggregate = float(np.sqrt(sse.sum() / max(float(count.sum()), 1.0)))
        return aggregate, per_frequency.tolist()

    @staticmethod
    def _path_scores(
        tp: np.ndarray,
        tn: np.ndarray,
        fp: np.ndarray,
        fn: np.ndarray,
    ) -> dict[str, float]:
        tp_value = float(tp.sum())
        tn_value = float(tn.sum())
        fp_value = float(fp.sum())
        fn_value = float(fn.sum())
        precision = tp_value / max(tp_value + fp_value, 1.0)
        recall = tp_value / max(tp_value + fn_value, 1.0)
        specificity = tn_value / max(tn_value + fp_value, 1.0)
        return {
            "accuracy": (tp_value + tn_value)
            / max(tp_value + tn_value + fp_value + fn_value, 1.0),
            "balanced_accuracy": 0.5 * (recall + specificity),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "true_path_pixels": tp_value + fn_value,
            "true_floor_pixels": tn_value + fp_value,
        }

    def finalize(self) -> dict[str, object]:
        all_rmse, all_per_frequency = self._rmse(self.all_sse, self.all_count)
        unseen_rmse, unseen_per_frequency = self._rmse(
            self.unseen_sse, self.unseen_count
        )
        nonfloor_rmse, nonfloor_per_frequency = self._rmse(
            self.nonfloor_unseen_sse, self.nonfloor_unseen_count
        )
        path_per_frequency = [
            self._path_scores(
                self.path_tp[index : index + 1],
                self.path_tn[index : index + 1],
                self.path_fp[index : index + 1],
                self.path_fn[index : index + 1],
            )
            for index in range(self.num_frequencies)
        ]
        floor_count = self.unseen_count - self.nonfloor_unseen_count
        return {
            "rmse_db": all_rmse,
            "rmse_db_per_freq": all_per_frequency,
            "macro_rmse_db": float(np.mean(all_per_frequency)),
            "unseen_rmse_db": unseen_rmse,
            "unseen_rmse_db_per_freq": unseen_per_frequency,
            "unseen_macro_rmse_db": float(np.mean(unseen_per_frequency)),
            "unseen_nonfloor_rmse_db": nonfloor_rmse,
            "unseen_nonfloor_rmse_db_per_freq": nonfloor_per_frequency,
            "unseen_nonfloor_macro_rmse_db": float(
                np.mean(nonfloor_per_frequency)
            ),
            "unseen_counts_per_freq": self.unseen_count.astype(int).tolist(),
            "unseen_nonfloor_counts_per_freq": self.nonfloor_unseen_count.astype(
                int
            ).tolist(),
            "unseen_floor_fraction_per_freq": (
                floor_count / np.maximum(self.unseen_count, 1.0)
            ).tolist(),
            "path_existence": self._path_scores(
                self.path_tp, self.path_tn, self.path_fp, self.path_fn
            ),
            "path_existence_per_freq": path_per_frequency,
            "path_existence_macro_f1": float(
                np.mean([scores["f1"] for scores in path_per_frequency])
            ),
            "path_existence_macro_balanced_accuracy": float(
                np.mean(
                    [scores["balanced_accuracy"] for scores in path_per_frequency]
                )
            ),
            "path_existence_definition": {
                "ground_truth": "normalized path gain > 0, i.e. above the -120 dB no-path floor",
                "prediction_threshold_db": self.floor_db
                + self.path_threshold_norm * self.db_span,
            },
        }
