#!/usr/bin/env python3
"""Stage-2 MF refiner v12: deeper shared-frequency residual U-Net with early
cross-frequency fusion, latent frequency attention, cosine LR schedule and
flip/rotation augmentation.

Protocol, caches, metrics and the output schema are identical to
train_stage2_refiner_v10.py (which is imported and left untouched). Only the
network and the optimisation recipe differ, so v10 validators and aggregators
keep working on the produced metrics.json.

Optional: a second (non-learned, e.g. IDW) prior cache can be added as an
extra input channel with --aux-coarse-root / --aux-evaluation-coarse-roots.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_stage2_complete_cnn_baselines_v10 import ldpl_prior
from train_stage2_refiner_v10 import (
    local_contrastive_loss,
    FREQS_GHZ,
    NF,
    CoarseRefinementDataset,
    ConvBlock,
    apply_observation_consistency,
    benchmark,
    collate,
    controlled_cache_split,
    evaluate,
    gradient_loss,
    jsonable_argument,
    make_figure,
    reconstruction_loss,
    verify_evaluation_partition,
)


# ----------------------------------------------------------------------------- data
class AuxCoarseDataset(CoarseRefinementDataset):
    """Adds an auxiliary prior (e.g. IDW cache) as item['aux'] (NF, H, W)."""

    def __init__(self, entries, edge_width, include_sparse_inputs, aux_root: Path | None):
        super().__init__(entries, edge_width, include_sparse_inputs)
        self.aux_lookup: dict[str, str] = {}
        if aux_root is not None:
            aux_entries = json.loads((aux_root / "index.json").read_text(encoding="utf-8"))
            self.aux_lookup = {
                Path(str(entry["graph_path"])).stem: str(entry["cache_path"]) for entry in aux_entries
            }
            missing = [
                Path(str(entry["graph_path"])).stem
                for entry in entries
                if Path(str(entry["graph_path"])).stem not in self.aux_lookup
            ]
            if missing:
                raise ValueError(f"aux cache {aux_root} misses {len(missing)} entries; first {missing[:2]}")

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        if self.aux_lookup:
            aux = np.load(self.aux_lookup[item["name"]], allow_pickle=False)["coarse"].astype(np.float32)
            item["aux"] = torch.tensor(aux, dtype=torch.float32)
        return item


def collate_v12(batch):
    output = collate(batch)
    if "aux" in batch[0]:
        output["aux"] = torch.stack([item["aux"] for item in batch])
    return output


SPATIAL_KEYS = ("coarse", "side", "target", "valid", "edge", "labels", "sparse", "obs_mask", "aux", "evaluation_mask")


def augment_batch(batch: dict, rng: random.Random) -> dict:
    """Random horizontal/vertical flip and 90-degree rotation applied jointly to
    every spatial tensor (all are [B, C, H, W] or [B, H, W] for labels)."""

    flip_h = rng.random() < 0.5
    flip_v = rng.random() < 0.5
    rot = rng.random() < 0.5
    if not (flip_h or flip_v or rot):
        return batch
    out = dict(batch)
    for key in SPATIAL_KEYS:
        if key not in batch:
            continue
        tensor = batch[key]
        if flip_h:
            tensor = torch.flip(tensor, dims=(-1,))
        if flip_v:
            tensor = torch.flip(tensor, dims=(-2,))
        if rot:
            tensor = torch.rot90(tensor, 1, dims=(-2, -1))
        out[key] = tensor.contiguous()
    return out


# ----------------------------------------------------------------------------- model
class MFRefinerV12(nn.Module):
    def __init__(
        self,
        side_channels: int,
        width: int,
        depth: int,
        channel_cap: int,
        include_sparse_inputs: bool,
        cross_frequency_input: bool,
        frequency_mixer: bool,
        frequency_mixer_heads: int,
        aux_input: bool,
        projection_dim: int = 64,
        use_skips: bool = True,
        physics_prior: bool = False,
    ):
        super().__init__()
        self.include_sparse_inputs = include_sparse_inputs
        self.cross_frequency_input = cross_frequency_input
        self.frequency_mixer = frequency_mixer
        self.aux_input = aux_input
        self.use_skips = use_skips
        self.physics_prior = physics_prior
        self.depth = depth
        per_band = 1 + side_channels + 1  # own coarse, side, frequency code
        if include_sparse_inputs:
            per_band += 2
        if aux_input:
            per_band += 1
        if physics_prior:
            per_band += 1
        cross = 0
        if cross_frequency_input:
            cross = NF  # all-band coarse priors
            if include_sparse_inputs:
                cross += 2 * NF
            if aux_input:
                cross += NF
            if physics_prior:
                cross += NF
        in_channels = per_band + cross
        chans = [min(width * (2**i), channel_cap) for i in range(depth + 1)]
        self.chans = chans
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        prev = in_channels
        for i in range(depth):
            self.enc.append(ConvBlock(prev, chans[i]))
            self.down.append(nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1))
            prev = chans[i + 1]
        self.bottleneck = ConvBlock(chans[depth], chans[depth])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 4, stride=2, padding=1))
            self.dec.append(ConvBlock((2 if use_skips else 1) * chans[i], chans[i]))
        self.head = nn.Conv2d(chans[0], 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.residual_scale = nn.Parameter(torch.full((NF,), 0.10))
        latent_channels = chans[depth]
        self.projector = nn.Sequential(
            nn.Linear(latent_channels, latent_channels // 2), nn.GELU(), nn.Linear(latent_channels // 2, projection_dim)
        )
        if frequency_mixer:
            if latent_channels % frequency_mixer_heads:
                raise ValueError("Latent channels must be divisible by frequency mixer heads")
            self.frequency_norm = nn.LayerNorm(latent_channels)
            self.frequency_attention = nn.MultiheadAttention(latent_channels, frequency_mixer_heads, batch_first=True)
            self.frequency_mix_scale = nn.Parameter(torch.tensor(0.10))
        log_freq = np.log(np.asarray(FREQS_GHZ, dtype=np.float32))
        log_freq = (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min())
        self.register_buffer("frequency_code", torch.tensor(log_freq, dtype=torch.float32))

    def mix_frequencies(self, latent: torch.Tensor) -> torch.Tensor:
        batch, frequencies, channels, height, width = latent.shape
        tokens = latent.permute(0, 3, 4, 1, 2).reshape(-1, frequencies, channels)
        normalized = self.frequency_norm(tokens)
        mixed, _ = self.frequency_attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + torch.tanh(self.frequency_mix_scale) * mixed
        return tokens.reshape(batch, height, width, frequencies, channels).permute(0, 3, 4, 1, 2).contiguous()

    def forward(self, coarse, side, sparse=None, obs_mask=None, aux=None):
        batch, frequencies, h, w = coarse.shape
        side_rep = side[:, None].expand(batch, frequencies, side.shape[1], h, w)
        freq = self.frequency_code[None, :, None, None, None].expand(batch, frequencies, 1, h, w)
        inputs = [coarse[:, :, None]]
        if self.include_sparse_inputs:
            if sparse is None or obs_mask is None:
                raise ValueError("Sparse maps and observation masks are required")
            inputs.extend([sparse[:, :, None], obs_mask[:, :, None]])
        if self.aux_input:
            if aux is None:
                raise ValueError("aux prior required")
            inputs.append(aux[:, :, None])
        phys = ldpl_prior(side) if self.physics_prior else None
        if phys is not None:
            inputs.append(phys[:, :, None])
        inputs.extend([side_rep, freq])
        if self.cross_frequency_input:
            shared = [coarse]
            if self.include_sparse_inputs:
                shared.extend([sparse, obs_mask])
            if self.aux_input:
                shared.append(aux)
            if phys is not None:
                shared.append(phys)
            shared_t = torch.cat(shared, dim=1)  # [B, C_shared, H, W]
            inputs.append(shared_t[:, None].expand(batch, frequencies, shared_t.shape[1], h, w))
        x = torch.cat(inputs, dim=2).reshape(batch * frequencies, -1, h, w)
        skips = []
        for enc, down in zip(self.enc, self.down):
            x = enc(x)
            skips.append(x)
            x = down(x)
        latent_flat = self.bottleneck(x)
        latent = latent_flat.reshape(batch, frequencies, latent_flat.shape[1], latent_flat.shape[2], latent_flat.shape[3])
        if self.frequency_mixer:
            latent = self.mix_frequencies(latent)
        x = latent.reshape(batch * frequencies, latent.shape[2], latent.shape[3], latent.shape[4])
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1) if self.use_skips else x)
        residual = torch.tanh(self.head(x)).reshape(batch, frequencies, h, w)
        pred = coarse + self.residual_scale[None, :, None, None] * residual
        return pred, latent


class AuxAdapter(nn.Module):
    """Lets the v10 evaluate()/benchmark()/make_figure() helpers (which call
    model(coarse, side, sparse, obs_mask)) work with an aux tensor that we
    stash on the module before each call."""

    def __init__(self, core: MFRefinerV12):
        super().__init__()
        self.core = core
        self._aux: torch.Tensor | None = None

    def set_aux(self, aux):
        self._aux = aux

    def forward(self, coarse, side, sparse=None, obs_mask=None):
        aux = self._aux
        if aux is not None and aux.shape[0] != coarse.shape[0]:
            aux = aux[: coarse.shape[0]]
        return self.core(coarse, side, sparse, obs_mask, aux)


@torch.no_grad()
def evaluate_v12(model: AuxAdapter, loader, device, hard_consistency: bool):
    """Same as v10 evaluate() but feeds aux per batch."""
    from radiomap_metrics_v9 import PixelMetricAccumulator

    model.eval()
    prediction_metrics = PixelMetricAccumulator(NF)
    coarse_metrics = PixelMetricAccumulator(NF)
    for batch in loader:
        coarse = batch["coarse"].to(device, non_blocking=True)
        side = batch["side"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        sparse = batch["sparse"].to(device, non_blocking=True) if "sparse" in batch else None
        obs_mask = batch["obs_mask"].to(device, non_blocking=True) if "obs_mask" in batch else None
        evaluation_mask = batch["evaluation_mask"].to(device, non_blocking=True) if "evaluation_mask" in batch else None
        model.set_aux(batch["aux"].to(device, non_blocking=True) if "aux" in batch else None)
        pred, _ = model(coarse, side, sparse, obs_mask)
        pred = apply_observation_consistency(pred, sparse, obs_mask, hard_consistency)
        prediction_metrics.update(pred, target, valid, obs_mask, evaluation_mask)
        coarse_metrics.update(coarse, target, valid, obs_mask, evaluation_mask)
    result = prediction_metrics.finalize()
    coarse_result = coarse_metrics.finalize()
    result.update(
        {
            "coarse_rmse_db": coarse_result["rmse_db"],
            "coarse_rmse_db_per_freq": coarse_result["rmse_db_per_freq"],
            "unseen_coarse_rmse_db": coarse_result["unseen_rmse_db"],
            "unseen_coarse_rmse_db_per_freq": coarse_result["unseen_rmse_db_per_freq"],
            "coarse_diagnostics": coarse_result,
        }
    )
    return result


def cosine_lr(step: int, total: int, warmup: int, base: float, minimum: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (base - minimum) * (1.0 + np.cos(np.pi * progress))


# ----------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-root", type=Path, required=True)
    parser.add_argument("--evaluation-coarse-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--aux-coarse-root", type=Path, default=None)
    parser.add_argument("--aux-evaluation-coarse-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--channel-cap", type=int, default=192)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lr-schedule", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--include-sparse-inputs", action="store_true")
    parser.add_argument("--hard-observation-consistency", action="store_true")
    parser.add_argument("--frequency-mixer", action="store_true")
    parser.add_argument("--cross-frequency-input", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-skips", action="store_true", help="AE mode: decoder without skip connections")
    parser.add_argument("--physics-prior", action="store_true", help="add a per-band LDPL physics channel")
    parser.add_argument("--clip-weight", type=float, default=0.0)
    parser.add_argument("--clip-temp", type=float, default=0.07)
    parser.add_argument("--clip-warmup-epochs", type=int, default=5)
    parser.add_argument("--contrastive-regions", type=int, default=64)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast for the forward pass")
    parser.add_argument("--frequency-mixer-heads", type=int, default=4)
    parser.add_argument("--edge-width", type=float, default=2.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=0.02)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--figure-freq-index", type=int, default=1)
    parser.add_argument("--selection-metric", choices=["unseen_rmse_db", "rmse_db"], default="unseen_rmse_db")
    args = parser.parse_args()

    if args.aux_coarse_root is not None and len(args.aux_evaluation_coarse_roots) != len(args.evaluation_coarse_roots):
        raise ValueError("--aux-evaluation-coarse-roots must align with --evaluation-coarse-roots")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    aug_rng = random.Random(args.seed + 17)

    train_entries, val_entries, test_entries, split_groups, cache_metadata = controlled_cache_split(
        args.coarse_root, args.max_examples, args.split_seed, args.train_ratio, args.val_ratio
    )
    aux_root = args.aux_coarse_root
    train_set = AuxCoarseDataset(train_entries, args.edge_width, args.include_sparse_inputs, aux_root)
    val_set = AuxCoarseDataset(val_entries, args.edge_width, args.include_sparse_inputs, aux_root)
    test_set = AuxCoarseDataset(test_entries, args.edge_width, args.include_sparse_inputs, aux_root)
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True, collate_fn=collate_v12)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    side_channels = train_set[0]["side"].shape[0]
    core = MFRefinerV12(
        side_channels,
        args.width,
        args.depth,
        args.channel_cap,
        args.include_sparse_inputs,
        args.cross_frequency_input,
        args.frequency_mixer,
        args.frequency_mixer_heads,
        aux_root is not None,
        args.projection_dim,
        use_skips=not args.no_skips,
        physics_prior=args.physics_prior,
    )
    model = AuxAdapter(core).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    initial_val = evaluate_v12(model, val_loader, device, args.hard_observation_consistency)
    best_val = initial_val[args.selection_metric]
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, args.out_dir / "best_model.pt")
    history = []
    stale = 0
    num_params = int(sum(p.numel() for p in model.parameters()))
    print(
        f"[load stage2 v12] train={len(train_set)} val={len(val_set)} test={len(test_set)} device={device} "
        f"params={num_params} depth={args.depth} chans={core.chans} cross_input={args.cross_frequency_input} "
        f"mixer={args.frequency_mixer} aux={aux_root is not None} aug={not args.no_augment} amp={args.amp} "
        f"initial_val={initial_val[args.selection_metric]:.3f} coarse={initial_val['unseen_coarse_rmse_db']:.3f}",
        flush=True,
    )
    autocast_dtype = torch.bfloat16
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.lr_schedule == "cosine":
            lr = cosine_lr(epoch - 1, args.epochs, args.warmup_epochs, args.lr, args.min_lr)
        else:
            lr = args.lr
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses, mse_values, grad_values = [], [], []
        start = time.time()
        for batch in train_loader:
            if not args.no_augment:
                batch = augment_batch(batch, aug_rng)
            coarse = batch["coarse"].to(device, non_blocking=True)
            side = batch["side"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            edge = batch["edge"].to(device, non_blocking=True)
            sparse = batch["sparse"].to(device, non_blocking=True) if "sparse" in batch else None
            obs_mask = batch["obs_mask"].to(device, non_blocking=True) if "obs_mask" in batch else None
            model.set_aux(batch["aux"].to(device, non_blocking=True) if "aux" in batch else None)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=args.amp and device.type == "cuda"):
                pred, latent = model(coarse, side, sparse, obs_mask)
            pred = pred.float()
            pred = apply_observation_consistency(pred, sparse, obs_mask, args.hard_observation_consistency)
            mse = reconstruction_loss(pred, target, valid, edge, args.edge_weight)
            grad = gradient_loss(pred, target, valid) if args.gradient_weight > 0 else pred.new_tensor(0.0)
            active_clip = args.clip_weight if epoch > args.clip_warmup_epochs else 0.0
            clip = (
                local_contrastive_loss(
                    model.core, latent.float(), batch["labels"].to(device, non_blocking=True),
                    args.contrastive_regions, args.clip_temp,
                )
                if active_clip > 0
                else pred.new_tensor(0.0)
            )
            loss = mse + args.gradient_weight * grad + active_clip * clip
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mse_values.append(float(mse.detach().cpu()))
            grad_values.append(float(grad.detach().cpu()))
        val = evaluate_v12(model, val_loader, device, args.hard_observation_consistency)
        record = {
            "epoch": epoch,
            "lr": lr,
            "loss": float(np.mean(losses)),
            "mse": float(np.mean(mse_values)),
            "gradient": float(np.mean(grad_values)),
            "val": val,
            "seconds": time.time() - start,
        }
        history.append(record)
        if val[args.selection_metric] < best_val:
            best_val = val[args.selection_metric]
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, args.out_dir / "best_model.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:03d}] lr={lr:.2e} loss={record['loss']:.6f} mse={record['mse']:.6f} "
                f"val={val[args.selection_metric]:.3f} best={best_val:.3f}@{best_epoch} "
                f"coarse={val['unseen_coarse_rmse_db']:.3f} sec={record['seconds']:.1f}",
                flush=True,
            )
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.patience > 0 and stale >= args.patience:
            print(f"[early stop] epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    val = evaluate_v12(model, val_loader, device, args.hard_observation_consistency)
    test = evaluate_v12(model, test_loader, device, args.hard_observation_consistency)
    mask_evaluations = []
    for index, evaluation_root in enumerate(args.evaluation_coarse_roots):
        _, _, evaluation_test_entries, evaluation_split_groups, evaluation_metadata = controlled_cache_split(
            evaluation_root, args.max_examples, args.split_seed, args.train_ratio, args.val_ratio
        )
        if evaluation_split_groups != split_groups:
            raise ValueError(f"XML split mismatch for evaluation cache {evaluation_root}")
        verify_evaluation_partition(test_entries, evaluation_test_entries, evaluation_root)
        eval_aux = args.aux_evaluation_coarse_roots[index] if aux_root is not None else None
        evaluation_test_set = AuxCoarseDataset(evaluation_test_entries, args.edge_width, args.include_sparse_inputs, eval_aux)
        evaluation_loader = DataLoader(evaluation_test_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        mask_evaluations.append(
            {
                "coarse_root": str(evaluation_root),
                "mask_seed": evaluation_metadata["mask_seed"],
                "test": evaluate_v12(model, evaluation_loader, device, args.hard_observation_consistency),
            }
        )
    # benchmark/figure helpers call model(coarse, side, sparse, obs_mask); stash aux of example 0
    if aux_root is not None:
        model.set_aux(test_set[0]["aux"][None].to(device))
    else:
        model.set_aux(None)
    inference_ms = benchmark(model, test_set, device, args.hard_observation_consistency, False)
    result = {
        "args": {key: jsonable_argument(value) for key, value in vars(args).items()},
        "protocol": (
            "fully_xml_inductive_two_stage_fixed_holdout_v11"
            if cache_metadata.get("protocol") == "budget_internal_stage1_projection_xml_inductive_fixed_holdout_v11"
            else "inductive_xml_grouped_matched_stage1_input_fixed_holdout_v11"
            if cache_metadata.get("protocol") == "budget_internal_stage1_projection_fixed_holdout_v11"
            else "inductive_xml_grouped_matched_stage1_input_v10"
        ),
        "primary_metric": args.selection_metric,
        "method": "MFGAT two-stage refiner v12",
        "stage1_source_model": cache_metadata.get("source_model_key"),
        "stage1_operating_point": cache_metadata.get("stage1_split_mode", "transductive"),
        "stage2_frequency_coupling": (
            ("early_input_fusion+" if args.cross_frequency_input else "")
            + ("latent_cross_frequency_attention" if args.frequency_mixer else "frequency_separable_shared_weights")
        ),
        "common_input": "frozen Stage-1 coarse prior + sparse maps + masks + geometry"
        + (" + auxiliary prior" if aux_root is not None else ""),
        "architecture": f"shared-frequency residual U-Net v12 depth={args.depth} chans={core.chans}",
        "components": {
            "frequency_mixer": bool(args.frequency_mixer),
            "cross_frequency_input": bool(args.cross_frequency_input),
            "hard_observation_consistency": bool(args.hard_observation_consistency),
            "aux_prior": aux_root is not None,
            "augment": not args.no_augment,
            "lr_schedule": args.lr_schedule,
            "amp": bool(args.amp),
            "clip_weight": float(args.clip_weight),
            "ae_no_skips": bool(args.no_skips),
            "physics_prior": bool(args.physics_prior),
            "zero_side_input": False,
        },
        "num_train": len(train_set),
        "num_val": len(val_set),
        "num_test": len(test_set),
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "cache_provenance": cache_metadata,
        "mask_evaluations": mask_evaluations,
        "split_groups": split_groups,
        "num_parameters": num_params,
        "inference_ms_per_map": inference_ms,
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "best_epoch": best_epoch,
        "best_val_selection_metric_db": best_val,
        "val": val,
        "test": test,
        "freqs_ghz": FREQS_GHZ,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        make_figure(model, test_set, device, args.out_dir / "prediction_example.png", args.figure_freq_index, args.hard_observation_consistency, False)
    except Exception as exc:  # figure is cosmetic
        print(f"[figure skipped] {exc}", flush=True)
    summary = {k: result[k] for k in ("num_parameters", "inference_ms_per_map", "peak_cuda_memory_mb", "best_epoch", "best_val_selection_metric_db")}
    summary["test_unseen_rmse_db"] = test["unseen_rmse_db"]
    summary["mask_eval_unseen_rmse_db"] = [m["test"]["unseen_rmse_db"] for m in mask_evaluations]
    print("[summary] " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
