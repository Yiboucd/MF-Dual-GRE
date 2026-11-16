#!/usr/bin/env python3
"""Fine-resolution AE refinement with local cross-frequency contrastive loss."""

from __future__ import annotations

import argparse
import json
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
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset

from radiomap_metrics_v9 import PixelMetricAccumulator
from build_clean_superpixel_graphs import load_manifest, load_maps


FREQS_GHZ = [2.4, 3.65, 4.9, 5.8, 24.0, 28.0, 39.0]
NF = len(FREQS_GHZ)
CONTROLLED_CACHE_PROTOCOLS = {
    "budget_internal_stage1_projection_v10",
    "budget_internal_stage1_projection_fixed_holdout_v11",
    "budget_internal_stage1_projection_xml_inductive_fixed_holdout_v11",
}
FIXED_CONTROLLED_CACHE_PROTOCOLS = {
    "budget_internal_stage1_projection_fixed_holdout_v11",
    "budget_internal_stage1_projection_xml_inductive_fixed_holdout_v11",
}


def jsonable_argument(value: object) -> object:
    """Convert argparse values, including lists of paths, to JSON-safe values."""

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return [jsonable_argument(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable_argument(item) for key, item in value.items()}
    return value


def entry_identity(entry: dict[str, object]) -> tuple[str, int, int]:
    return str(entry["city"]), int(entry["xml_idx"]), int(entry["tx_idx"])


def verify_evaluation_partition(
    reference_entries: list[dict[str, object]],
    evaluation_entries: list[dict[str, object]],
    evaluation_root: Path,
) -> None:
    reference = [entry_identity(entry) for entry in reference_entries]
    candidate = [entry_identity(entry) for entry in evaluation_entries]
    if candidate != reference:
        raise ValueError(
            f"Sample/order mismatch for evaluation cache {evaluation_root}; "
            "all mask evaluations must use the identical held-out maps"
        )


def validate_controlled_cache(
    coarse_root: Path,
    entries: list[dict[str, object]],
) -> dict[str, object]:
    metadata_path = coarse_root / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Missing controlled-cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol") not in CONTROLLED_CACHE_PROTOCOLS:
        raise ValueError(
            f"Refusing legacy or unmatched Stage-1 cache {coarse_root}: "
            f"protocol={metadata.get('protocol')}"
        )
    if int(metadata.get("num_examples", -1)) != len(entries):
        raise ValueError(
            f"Cache index/metadata count mismatch: {len(entries)} vs "
            f"{metadata.get('num_examples')}"
        )
    if metadata.get("source_stage1_protocol") not in {
        "budget_internal_anchor_validation_transductive_v10",
        "budget_internal_fixed_pixel_holdout_transductive_v11",
        "budget_internal_fixed_pixel_holdout_xml_inductive_v11",
    }:
        raise ValueError("Cache was not produced by the corrected masked-anchor Stage-1 protocol")
    stage1_split_mode = str(metadata.get("stage1_split_mode", "transductive"))
    expected_cache_protocol = (
        "budget_internal_stage1_projection_xml_inductive_fixed_holdout_v11"
        if stage1_split_mode == "xml_inductive"
        else "budget_internal_stage1_projection_fixed_holdout_v11"
    )
    expected_source_protocol = (
        "budget_internal_fixed_pixel_holdout_xml_inductive_v11"
        if stage1_split_mode == "xml_inductive"
        else "budget_internal_fixed_pixel_holdout_transductive_v11"
    )
    if metadata.get("protocol") in FIXED_CONTROLLED_CACHE_PROTOCOLS:
        if metadata.get("protocol") != expected_cache_protocol:
            raise ValueError("Cache protocol and Stage-1 operating point disagree")
        if metadata.get("source_stage1_protocol") != expected_source_protocol:
            raise ValueError("Cache and checkpoint split protocols disagree")
    if stage1_split_mode == "xml_inductive":
        groups = metadata.get("source_stage1_split_groups", {})
        train_groups = set(groups.get("train", []))
        val_groups = set(groups.get("val", []))
        test_groups = set(groups.get("test", []))
        if not train_groups or not val_groups or not test_groups:
            raise ValueError("XML-inductive cache lacks complete Stage-1 groups")
        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise ValueError("XML-inductive cache has overlapping Stage-1 groups")
    elif stage1_split_mode != "transductive":
        raise ValueError(f"Unsupported Stage-1 operating point: {stage1_split_mode}")
    if not entries:
        raise ValueError(f"Empty Stage-2 cache: {coarse_root}")
    with np.load(str(entries[0]["cache_path"]), allow_pickle=False) as sample:
        missing = {"coarse", "target", "side", "valid", "labels", "sparse", "obs_mask"} - set(sample.files)
        if metadata.get("protocol") in FIXED_CONTROLLED_CACHE_PROTOCOLS:
            missing |= {"evaluation_mask"} - set(sample.files)
            if "evaluation_mask" in sample.files:
                evaluation_mask = sample["evaluation_mask"].astype(bool)
                observation_mask = sample["obs_mask"].astype(bool)
                if evaluation_mask.shape != observation_mask.shape:
                    raise ValueError("Fixed evaluation and observation masks have different shapes")
                if np.any(evaluation_mask & observation_mask):
                    raise ValueError("Fixed evaluation pixels overlap sparse observations")
    if missing:
        raise ValueError(f"Controlled cache sample is missing fields: {sorted(missing)}")
    if metadata.get("protocol") in FIXED_CONTROLLED_CACHE_PROTOCOLS:
        if float(metadata.get("fixed_test_rate_percent", 0.0)) <= 0:
            raise ValueError("v11 cache does not declare a positive fixed holdout rate")
        if "fixed_test_seed" not in metadata:
            raise ValueError("v11 cache does not declare its fixed holdout seed")
    source_model = metadata.get("source_model_key")
    if source_model == "idw":
        if metadata.get("prediction_grid") != "native_pixel_grid":
            raise ValueError("IDW Stage-1 prior must be generated on the native pixel grid")
        if metadata.get("idw_injected") is not True:
            raise ValueError("IDW Stage-1 prior lacks explicit provenance")
    elif source_model in {"mfgat", "sf_gat_all"}:
        if metadata.get("prediction_grid") != "superpixel_projection":
            raise ValueError("GAT Stage-1 prior must be projected from superpixels")
        if metadata.get("idw_injected") is not False:
            raise ValueError("GAT Stage-1 prior unexpectedly declares IDW injection")
    else:
        raise ValueError(f"Unsupported Stage-1 cache source: {source_model}")
    return metadata


def controlled_cache_split(
    coarse_root: Path,
    max_examples: int,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[str]],
    dict[str, object],
]:
    entries = json.loads((coarse_root / "index.json").read_text(encoding="utf-8"))
    metadata = validate_controlled_cache(coarse_root, entries)
    if max_examples > 0:
        subset_rng = random.Random(split_seed + 991)
        subset_rng.shuffle(entries)
        entries = entries[:max_examples]
    train_entries, val_entries, test_entries, split_groups = grouped_split(
        entries, split_seed, train_ratio, val_ratio
    )
    source_split_mode = str(metadata.get("stage1_split_mode", "transductive"))
    if source_split_mode == "xml_inductive":
        source_groups = metadata.get("source_stage1_split_groups", {})
        if split_groups != source_groups:
            raise ValueError(
                "Stage-2 XML groups do not exactly match the XML-inductive "
                "Stage-1 checkpoint"
            )
    elif source_split_mode != "transductive":
        raise ValueError(f"Unsupported Stage-1 operating point: {source_split_mode}")
    if not train_entries or not val_entries or not test_entries:
        raise RuntimeError("XML-grouped split produced an empty partition")
    return train_entries, val_entries, test_entries, split_groups, metadata


def norm_map(arr_db: np.ndarray) -> np.ndarray:
    return np.clip((arr_db.astype(np.float32) + 120.0) / 40.0, 0.0, 1.0)


def nodes_to_pixel(node_values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    padded = np.zeros((node_values.shape[0] + 1, node_values.shape[1]), dtype=np.float32)
    padded[1:] = node_values
    return padded[labels].transpose(2, 0, 1).astype(np.float32)


def load_target_maps(manifest_path: str) -> np.ndarray:
    manifest = load_manifest(Path(manifest_path))
    return norm_map(load_maps(manifest)).transpose(2, 0, 1).astype(np.float32)


def grouped_split(entries: list[dict[str, object]], seed: int, train_ratio: float, val_ratio: float):
    train_groups: set[tuple[str, int]] = set()
    val_groups: set[tuple[str, int]] = set()
    test_groups: set[tuple[str, int]] = set()
    for city in sorted({str(entry["city"]) for entry in entries}):
        groups = sorted({(str(entry["city"]), int(entry["xml_idx"])) for entry in entries if entry["city"] == city})
        random.Random(seed + sum(ord(char) for char in city)).shuffle(groups)
        n = len(groups)
        n_train = max(1, int(round(train_ratio * n)))
        n_val = max(1, int(round(val_ratio * n)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train_groups.update(groups[:n_train])
        val_groups.update(groups[n_train : n_train + n_val])
        test_groups.update(groups[n_train + n_val :])
    select = lambda groups: [entry for entry in entries if (str(entry["city"]), int(entry["xml_idx"])) in groups]
    split_groups = {
        "train": [f"{city}:xml_{idx}" for city, idx in sorted(train_groups)],
        "val": [f"{city}:xml_{idx}" for city, idx in sorted(val_groups)],
        "test": [f"{city}:xml_{idx}" for city, idx in sorted(test_groups)],
    }
    return select(train_groups), select(val_groups), select(test_groups), split_groups


class CoarseRefinementDataset(Dataset):
    def __init__(self, entries: list[dict[str, object]], edge_width: float, include_sparse_inputs: bool = False):
        self.entries = entries
        self.edge_width = edge_width
        self.include_sparse_inputs = include_sparse_inputs

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        cache = np.load(str(entry["cache_path"]), allow_pickle=False)
        coarse = cache["coarse"].astype(np.float32)
        labels = cache["labels"].astype(np.int64)
        valid = cache["valid"].astype(bool)
        side = cache["side"].astype(np.float32)
        target = cache["target"].astype(np.float32)
        distance = distance_transform_edt(valid).astype(np.float32)
        edge = valid & (distance <= self.edge_width)
        item = {
            "coarse": torch.tensor(coarse, dtype=torch.float32),
            "side": torch.tensor(side, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "valid": torch.tensor(valid[None], dtype=torch.bool),
            "edge": torch.tensor(edge[None], dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "name": Path(str(entry["graph_path"])).stem,
        }
        if "evaluation_mask" in cache:
            item["evaluation_mask"] = torch.tensor(
                cache["evaluation_mask"].astype(bool), dtype=torch.bool
            )
        if self.include_sparse_inputs:
            if "sparse" in cache and "obs_mask" in cache:
                sparse = cache["sparse"].astype(np.float32)
                obs_mask = cache["obs_mask"].astype(np.float32)
            else:
                graph = np.load(str(entry["graph_path"]), allow_pickle=False)
                sparse = graph["sparse_train_maps"].transpose(2, 0, 1).astype(np.float32)
                obs_mask = graph["obs_train_pixel_masks"].transpose(2, 0, 1).astype(np.float32)
            item["sparse"] = torch.tensor(sparse, dtype=torch.float32)
            item["obs_mask"] = torch.tensor(obs_mask, dtype=torch.float32)
        return item


def collate(batch):
    output = {
        "coarse": torch.stack([item["coarse"] for item in batch]),
        "side": torch.stack([item["side"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "valid": torch.stack([item["valid"] for item in batch]),
        "edge": torch.stack([item["edge"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "name": [item["name"] for item in batch],
    }
    if "sparse" in batch[0]:
        output["sparse"] = torch.stack([item["sparse"] for item in batch])
        output["obs_mask"] = torch.stack([item["obs_mask"] for item in batch])
    if "evaluation_mask" in batch[0]:
        output["evaluation_mask"] = torch.stack(
            [item["evaluation_mask"] for item in batch]
        )
    return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = 8 if out_channels >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedFrequencyRefiner(nn.Module):
    def __init__(
        self,
        side_channels: int,
        width: int,
        projection_dim: int,
        include_sparse_inputs: bool = False,
        frequency_mixer: bool = False,
        frequency_mixer_heads: int = 4,
    ):
        super().__init__()
        self.include_sparse_inputs = include_sparse_inputs
        self.frequency_mixer = frequency_mixer
        input_channels = 1 + side_channels + 1 + (2 if include_sparse_inputs else 0)
        self.enc1 = ConvBlock(input_channels, width)
        self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(width * 2, width * 2)
        self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.bottleneck = ConvBlock(width * 4, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(width * 2, width)
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.residual_scale = nn.Parameter(torch.full((NF,), 0.10))
        self.projector = nn.Sequential(
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, projection_dim),
        )
        if frequency_mixer:
            latent_channels = width * 4
            if latent_channels % frequency_mixer_heads:
                raise ValueError("Latent channels must be divisible by frequency mixer heads")
            self.frequency_norm = nn.LayerNorm(latent_channels)
            self.frequency_attention = nn.MultiheadAttention(
                latent_channels,
                frequency_mixer_heads,
                batch_first=True,
            )
            self.frequency_mix_scale = nn.Parameter(torch.tensor(0.10))
        log_freq = np.log(np.asarray(FREQS_GHZ, dtype=np.float32))
        log_freq = (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min())
        self.register_buffer("frequency_code", torch.tensor(log_freq, dtype=torch.float32))

    def encode(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        latent = self.bottleneck(self.down2(e2))
        return e1, e2, latent

    def decode(self, e1: torch.Tensor, e2: torch.Tensor, latent: torch.Tensor):
        u2 = self.up2(latent)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        residual = self.head(self.dec1(torch.cat([u1, e1], dim=1)))
        return residual

    def mix_frequencies(self, latent: torch.Tensor) -> torch.Tensor:
        batch, frequencies, channels, height, width = latent.shape
        tokens = latent.permute(0, 3, 4, 1, 2).reshape(-1, frequencies, channels)
        normalized = self.frequency_norm(tokens)
        mixed, _ = self.frequency_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + torch.tanh(self.frequency_mix_scale) * mixed
        return (
            tokens.reshape(batch, height, width, frequencies, channels)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    def forward(
        self,
        coarse: torch.Tensor,
        side: torch.Tensor,
        sparse: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
    ):
        batch, frequencies, h, w = coarse.shape
        side_rep = side[:, None].expand(batch, frequencies, side.shape[1], h, w)
        freq = self.frequency_code[None, :, None, None, None].expand(batch, frequencies, 1, h, w)
        inputs = [coarse[:, :, None]]
        if self.include_sparse_inputs:
            if sparse is None or obs_mask is None:
                raise ValueError("Sparse maps and observation masks are required for sparse-aware refinement")
            inputs.extend([sparse[:, :, None], obs_mask[:, :, None]])
        inputs.extend([side_rep, freq])
        x = torch.cat(inputs, dim=2)
        e1, e2, latent_flat = self.encode(x.reshape(batch * frequencies, x.shape[2], h, w))
        latent = latent_flat.reshape(
            batch,
            frequencies,
            latent_flat.shape[1],
            latent_flat.shape[2],
            latent_flat.shape[3],
        )
        if self.frequency_mixer:
            latent = self.mix_frequencies(latent)
        residual = self.decode(
            e1,
            e2,
            latent.reshape(
                batch * frequencies,
                latent.shape[2],
                latent.shape[3],
                latent.shape[4],
            ),
        )
        residual = torch.tanh(residual).reshape(batch, frequencies, h, w)
        pred = coarse + self.residual_scale[None, :, None, None] * residual
        return pred, latent


def local_contrastive_loss(
    model: SharedFrequencyRefiner,
    latent: torch.Tensor,
    labels: torch.Tensor,
    max_regions: int,
    temperature: float,
    center_frequency: bool = False,
) -> torch.Tensor:
    batch, frequencies, channels, hz, wz = latent.shape
    labels_small = F.interpolate(labels[:, None].float(), size=(hz, wz), mode="nearest")[:, 0].long()
    region_embeddings = []
    for bi in range(batch):
        flat_labels = labels_small[bi].reshape(-1)
        unique = torch.unique(flat_labels)
        unique = unique[unique > 0]
        if unique.numel() > max_regions:
            unique = unique[torch.randperm(unique.numel(), device=unique.device)[:max_regions]]
        if not unique.numel():
            continue
        region_masks = flat_labels[None, :] == unique[:, None]
        counts = region_masks.sum(dim=1).clamp_min(1).to(latent.dtype)
        latent_flat = latent[bi].reshape(frequencies, channels, -1)
        pooled = torch.einsum("fcp,rp->rfc", latent_flat, region_masks.to(latent.dtype))
        pooled = pooled / counts[:, None, None]
        projected = model.projector(pooled)
        if not center_frequency:
            projected = F.normalize(projected, dim=-1, eps=1e-6)
        region_embeddings.append(projected)
    if not region_embeddings:
        return latent.new_tensor(0.0)
    embeddings = torch.cat(region_embeddings, dim=0)
    if center_frequency:
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        embeddings = F.normalize(embeddings, dim=-1, eps=1e-6)
    regions = embeddings.shape[0]
    flat = embeddings.reshape(regions * frequencies, -1)
    logits = flat @ flat.t() / temperature
    eye = torch.eye(flat.shape[0], dtype=torch.bool, device=flat.device)
    logits = logits.masked_fill(eye, -1e9)
    region_ids = torch.arange(regions, device=flat.device).repeat_interleave(frequencies)
    positive = (region_ids[:, None] == region_ids[None, :]) & (~eye)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * positive).sum(dim=1).div(positive.sum(dim=1).clamp_min(1)).mean()


def reconstruction_loss(pred, target, valid, edge, edge_weight: float):
    mask = valid.expand_as(target)
    weights = (1.0 + edge_weight * edge).expand_as(target)
    squared = (pred - target).square() * weights
    return squared[mask].sum() / weights[mask].sum().clamp_min(1.0)


def gradient_loss(pred, target, valid):
    dx_mask = valid[:, :, :, 1:] & valid[:, :, :, :-1]
    dy_mask = valid[:, :, 1:, :] & valid[:, :, :-1, :]
    dx_mask = dx_mask.expand(-1, NF, -1, -1)
    dy_mask = dy_mask.expand(-1, NF, -1, -1)
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1])
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) - (target[:, :, 1:, :] - target[:, :, :-1, :])
    return 0.5 * (dx[dx_mask].square().mean() + dy[dy_mask].square().mean())


def apply_observation_consistency(pred, sparse, obs_mask, enabled: bool):
    if not enabled:
        return pred
    if sparse is None or obs_mask is None:
        raise ValueError("Hard observation consistency requires sparse inputs and masks")
    return torch.where(obs_mask.bool(), sparse, pred)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    hard_observation_consistency: bool = False,
    zero_side_input: bool = False,
):
    model.eval()
    prediction_metrics = PixelMetricAccumulator(NF)
    coarse_metrics = PixelMetricAccumulator(NF)
    for batch in loader:
        coarse = batch["coarse"].to(device, non_blocking=True)
        side = batch["side"].to(device, non_blocking=True)
        if zero_side_input:
            side = torch.zeros_like(side)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        sparse = batch["sparse"].to(device, non_blocking=True) if "sparse" in batch else None
        obs_mask = batch["obs_mask"].to(device, non_blocking=True) if "obs_mask" in batch else None
        evaluation_mask = (
            batch["evaluation_mask"].to(device, non_blocking=True)
            if "evaluation_mask" in batch
            else None
        )
        pred, _ = model(coarse, side, sparse, obs_mask)
        pred = apply_observation_consistency(pred, sparse, obs_mask, hard_observation_consistency)
        prediction_metrics.update(pred, target, valid, obs_mask, evaluation_mask)
        coarse_metrics.update(coarse, target, valid, obs_mask, evaluation_mask)
    result = prediction_metrics.finalize()
    coarse_result = coarse_metrics.finalize()
    result.update(
        {
            "coarse_rmse_db": coarse_result["rmse_db"],
            "coarse_rmse_db_per_freq": coarse_result["rmse_db_per_freq"],
            "unseen_coarse_rmse_db": coarse_result["unseen_rmse_db"],
            "unseen_coarse_rmse_db_per_freq": coarse_result[
                "unseen_rmse_db_per_freq"
            ],
            "coarse_diagnostics": coarse_result,
        }
    )
    return result


@torch.no_grad()
def benchmark(
    model: nn.Module,
    dataset,
    device: torch.device,
    hard_observation_consistency: bool = False,
    zero_side_input: bool = False,
) -> float:
    """Measure one-map Stage-2 latency with the same inference path as evaluation."""

    example = collate([dataset[0]])
    coarse = example["coarse"].to(device)
    side = example["side"].to(device)
    if zero_side_input:
        side = torch.zeros_like(side)
    sparse = example["sparse"].to(device) if "sparse" in example else None
    obs_mask = example["obs_mask"].to(device) if "obs_mask" in example else None
    model.eval()

    def run_once() -> None:
        prediction, _ = model(coarse, side, sparse, obs_mask)
        apply_observation_consistency(
            prediction,
            sparse,
            obs_mask,
            hard_observation_consistency,
        )

    for _ in range(3):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    repeats = 10
    for _ in range(repeats):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - started) / repeats


@torch.no_grad()
def make_figure(
    model,
    dataset,
    device,
    out_path: Path,
    freq_idx: int,
    hard_observation_consistency: bool = False,
    zero_side_input: bool = False,
):
    example = dataset[0]
    coarse = example["coarse"][None].to(device)
    side = example["side"][None].to(device)
    if zero_side_input:
        side = torch.zeros_like(side)
    sparse = example["sparse"][None].to(device) if "sparse" in example else None
    obs_mask = example["obs_mask"][None].to(device) if "obs_mask" in example else None
    pred, _ = model(coarse, side, sparse, obs_mask)
    pred = apply_observation_consistency(pred, sparse, obs_mask, hard_observation_consistency)
    valid = example["valid"][0].numpy()
    coarse_db = example["coarse"][freq_idx].numpy() * 40.0 - 120.0
    pred_db = pred[0, freq_idx].clamp(0.0, 1.0).cpu().numpy() * 40.0 - 120.0
    target_db = example["target"][freq_idx].numpy() * 40.0 - 120.0
    for array in [coarse_db, pred_db, target_db]:
        array[~valid] = np.nan
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    ecmap = plt.get_cmap("magma").copy()
    ecmap.set_bad("white")
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), constrained_layout=True)
    im = axes[0, 0].imshow(coarse_db, cmap=cmap, vmin=-120, vmax=-80)
    axes[0, 0].set_title("Stage-1 coarse")
    axes[0, 1].imshow(pred_db, cmap=cmap, vmin=-120, vmax=-80)
    axes[0, 1].set_title("Stage-2 refined")
    axes[0, 2].imshow(target_db, cmap=cmap, vmin=-120, vmax=-80)
    axes[0, 2].set_title("Ground truth")
    if "sparse" in example:
        sparse_db = example["sparse"][freq_idx].numpy() * 40.0 - 120.0
        observed = example["obs_mask"][freq_idx].numpy().astype(bool) & valid
        sparse_db[~observed] = np.nan
        axes[1, 0].imshow(sparse_db, cmap=cmap, vmin=-120, vmax=-80)
        axes[1, 0].set_title("Raw sparse observations")
    else:
        axes[1, 0].axis("off")
    eim = axes[1, 1].imshow(np.abs(coarse_db - target_db), cmap=ecmap, vmin=0, vmax=8)
    axes[1, 1].set_title("Coarse absolute error")
    axes[1, 2].imshow(np.abs(pred_db - target_db), cmap=ecmap, vmin=0, vmax=8)
    axes[1, 2].set_title("Refined absolute error")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.01, label="Path gain (dB)")
    fig.colorbar(eim, ax=axes[1, 1:], fraction=0.035, pad=0.01, label="error (dB)")
    fig.suptitle(f"{example['name']} | {FREQS_GHZ[freq_idx]:.2f} GHz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-root", type=Path, required=True)
    parser.add_argument("--evaluation-coarse-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional deterministic subset size for smoke tests; zero uses all entries.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-weight", type=float, default=1e-4)
    parser.add_argument("--clip-temp", type=float, default=0.07)
    parser.add_argument("--clip-warmup-epochs", type=int, default=5)
    parser.add_argument("--clip-center-frequency", action="store_true")
    parser.add_argument("--include-sparse-inputs", action="store_true")
    parser.add_argument("--hard-observation-consistency", action="store_true")
    parser.add_argument("--frequency-mixer", action="store_true")
    parser.add_argument(
        "--zero-side-input",
        action="store_true",
        help="Zero all geometry/TX side channels for the matched no-geometry ablation.",
    )
    parser.add_argument("--frequency-mixer-heads", type=int, default=4)
    parser.add_argument("--contrastive-regions", type=int, default=64)
    parser.add_argument("--edge-width", type=float, default=2.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=0.02)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--figure-freq-index", type=int, default=1)
    parser.add_argument(
        "--selection-metric",
        choices=["unseen_rmse_db", "rmse_db"],
        default="unseen_rmse_db",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_entries, val_entries, test_entries, split_groups, cache_metadata = controlled_cache_split(
        args.coarse_root,
        args.max_examples,
        args.split_seed,
        args.train_ratio,
        args.val_ratio,
    )
    train_set = CoarseRefinementDataset(train_entries, args.edge_width, args.include_sparse_inputs)
    val_set = CoarseRefinementDataset(val_entries, args.edge_width, args.include_sparse_inputs)
    test_set = CoarseRefinementDataset(test_entries, args.edge_width, args.include_sparse_inputs)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    side_channels = train_set[0]["side"].shape[0]
    model = SharedFrequencyRefiner(
        side_channels,
        args.width,
        args.projection_dim,
        args.include_sparse_inputs,
        args.frequency_mixer,
        args.frequency_mixer_heads,
    ).to(device)
    if args.init_checkpoint is not None:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    initial_val = evaluate(
        model,
        val_loader,
        device,
        args.hard_observation_consistency,
        args.zero_side_input,
    )
    best_val = initial_val[args.selection_metric]
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(best_state, args.out_dir / "best_model.pt")
    history = []
    stale = 0
    print(
        f"[load stage2 v2] train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"device={device} clip={args.clip_weight} center_frequency={args.clip_center_frequency} "
        f"sparse_inputs={args.include_sparse_inputs} hard_consistency={args.hard_observation_consistency} "
        f"frequency_mixer={args.frequency_mixer} "
        f"init={args.init_checkpoint} initial={initial_val}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, mse_values, clip_values, grad_values = [], [], [], []
        start = time.time()
        active_clip = args.clip_weight if epoch > args.clip_warmup_epochs else 0.0
        for batch in train_loader:
            coarse = batch["coarse"].to(device, non_blocking=True)
            side = batch["side"].to(device, non_blocking=True)
            if args.zero_side_input:
                side = torch.zeros_like(side)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            edge = batch["edge"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            sparse = batch["sparse"].to(device, non_blocking=True) if "sparse" in batch else None
            obs_mask = batch["obs_mask"].to(device, non_blocking=True) if "obs_mask" in batch else None
            pred, latent = model(coarse, side, sparse, obs_mask)
            pred = apply_observation_consistency(
                pred,
                sparse,
                obs_mask,
                args.hard_observation_consistency,
            )
            mse = reconstruction_loss(pred, target, valid, edge, args.edge_weight)
            grad = gradient_loss(pred, target, valid) if args.gradient_weight > 0 else pred.new_tensor(0.0)
            clip = (
                local_contrastive_loss(
                    model,
                    latent,
                    labels,
                    args.contrastive_regions,
                    args.clip_temp,
                    args.clip_center_frequency,
                )
                if active_clip > 0
                else pred.new_tensor(0.0)
            )
            loss = mse + args.gradient_weight * grad + active_clip * clip
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mse_values.append(float(mse.detach().cpu()))
            grad_values.append(float(grad.detach().cpu()))
            clip_values.append(float(clip.detach().cpu()))
        val = evaluate(
            model,
            val_loader,
            device,
            args.hard_observation_consistency,
            args.zero_side_input,
        )
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "mse": float(np.mean(mse_values)),
            "gradient": float(np.mean(grad_values)),
            "clip": float(np.mean(clip_values)),
            "active_clip_weight": active_clip,
            "val": val,
            "seconds": time.time() - start,
        }
        history.append(record)
        if val[args.selection_metric] < best_val:
            best_val = val[args.selection_metric]
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, args.out_dir / "best_model.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:03d}] loss={record['loss']:.6f} mse={record['mse']:.6f} "
                f"clip={record['clip']:.4f} val={val[args.selection_metric]:.3f} "
                f"coarse={val['coarse_rmse_db']:.3f} sec={record['seconds']:.1f}",
                flush=True,
            )
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.patience > 0 and stale >= args.patience:
            print(f"[early stop] epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    val = evaluate(
        model,
        val_loader,
        device,
        args.hard_observation_consistency,
        args.zero_side_input,
    )
    test = evaluate(
        model,
        test_loader,
        device,
        args.hard_observation_consistency,
        args.zero_side_input,
    )
    mask_evaluations = []
    for evaluation_root in args.evaluation_coarse_roots:
        _, _, evaluation_test_entries, evaluation_split_groups, evaluation_metadata = controlled_cache_split(
            evaluation_root,
            args.max_examples,
            args.split_seed,
            args.train_ratio,
            args.val_ratio,
        )
        if evaluation_split_groups != split_groups:
            raise ValueError(f"XML split mismatch for evaluation cache {evaluation_root}")
        verify_evaluation_partition(test_entries, evaluation_test_entries, evaluation_root)
        evaluation_test_set = CoarseRefinementDataset(
            evaluation_test_entries, args.edge_width, args.include_sparse_inputs
        )
        evaluation_loader = DataLoader(
            evaluation_test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate,
        )
        mask_evaluations.append(
            {
                "coarse_root": str(evaluation_root),
                "mask_seed": evaluation_metadata["mask_seed"],
                "test": evaluate(
                    model,
                    evaluation_loader,
                    device,
                    args.hard_observation_consistency,
                    args.zero_side_input,
                ),
            }
        )
    inference_ms = benchmark(
        model,
        test_set,
        device,
        args.hard_observation_consistency,
        args.zero_side_input,
    )
    result = {
        "args": {key: jsonable_argument(value) for key, value in vars(args).items()},
        "protocol": (
            "fully_xml_inductive_two_stage_fixed_holdout_v11"
            if cache_metadata.get("protocol")
            == "budget_internal_stage1_projection_xml_inductive_fixed_holdout_v11"
            else "inductive_xml_grouped_matched_stage1_input_fixed_holdout_v11"
            if cache_metadata.get("protocol")
            == "budget_internal_stage1_projection_fixed_holdout_v11"
            else "inductive_xml_grouped_matched_stage1_input_v10"
        ),
        "primary_metric": args.selection_metric,
        "method": "MFGAT two-stage refiner",
        "stage1_source_model": cache_metadata.get("source_model_key"),
        "stage1_operating_point": cache_metadata.get(
            "stage1_split_mode", "transductive"
        ),
        "stage2_frequency_coupling": (
            "latent_cross_frequency_attention"
            if args.frequency_mixer
            else "frequency_separable_shared_weights"
        ),
        "common_input": "frozen Stage-1 coarse prior + sparse maps + masks + geometry",
        "architecture": "shared-frequency residual U-Net with optional latent frequency attention",
        "components": {
            "frequency_mixer": bool(args.frequency_mixer),
            "hard_observation_consistency": bool(args.hard_observation_consistency),
            "clip_weight": float(args.clip_weight),
            "zero_side_input": bool(args.zero_side_input),
        },
        "num_train": len(train_set),
        "num_val": len(val_set),
        "num_test": len(test_set),
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "cache_provenance": cache_metadata,
        "mask_evaluations": mask_evaluations,
        "split_groups": split_groups,
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "inference_ms_per_map": inference_ms,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "best_epoch": best_epoch,
        "best_val_selection_metric_db": best_val,
        "val": val,
        "test": test,
        "freqs_ghz": FREQS_GHZ,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(
        model,
        test_set,
        device,
        args.out_dir / "prediction_example.png",
        args.figure_freq_index,
        args.hard_observation_consistency,
        args.zero_side_input,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
