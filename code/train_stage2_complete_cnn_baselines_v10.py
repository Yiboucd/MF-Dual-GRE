#!/usr/bin/env python3
"""Train fully supervised dense radio-map baselines under one protocol.

These are complete-pipeline baselines. They receive raw sparse measurements,
their binary observation masks, and geometry/TX side information. They never
receive IDW, superpixel values, or a frozen MFGAT Stage-1 prediction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from torch.utils.data import DataLoader

from radiomap_metrics_v9 import PixelMetricAccumulator

from train_stage2_refiner_v10 import (
    FREQS_GHZ,
    NF,
    CoarseRefinementDataset,
    FIXED_CONTROLLED_CACHE_PROTOCOLS,
    apply_observation_consistency,
    collate,
    grouped_split,
    controlled_cache_split,
    jsonable_argument,
    validate_controlled_cache,
    verify_evaluation_partition,
)


def normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = normalization_groups(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FullUNet(nn.Module):
    """Four-level U-Net used by the U-Net and GAN baselines."""

    def __init__(self, in_channels: int, out_channels: int, width: int):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, width)
        self.enc2 = DoubleConv(width, 2 * width)
        self.enc3 = DoubleConv(2 * width, 4 * width)
        self.enc4 = DoubleConv(4 * width, 8 * width)
        self.center = DoubleConv(8 * width, 16 * width)
        self.up4 = nn.ConvTranspose2d(16 * width, 8 * width, 2, stride=2)
        self.dec4 = DoubleConv(16 * width, 8 * width)
        self.up3 = nn.ConvTranspose2d(8 * width, 4 * width, 2, stride=2)
        self.dec3 = DoubleConv(8 * width, 4 * width)
        self.up2 = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2)
        self.dec2 = DoubleConv(4 * width, 2 * width)
        self.up1 = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.dec1 = DoubleConv(2 * width, width)
        self.head = nn.Conv2d(width, out_channels, 1)

    @staticmethod
    def merge(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        center = self.center(F.max_pool2d(e4, 2))
        d4 = self.dec4(self.merge(self.up4(center), e4))
        d3 = self.dec3(self.merge(self.up3(d4), e3))
        d2 = self.dec2(self.merge(self.up2(d3), e2))
        d1 = self.dec1(self.merge(self.up1(d2), e1))
        return self.head(d1)


class ConvAutoencoder(nn.Module):
    """Dense convolutional autoencoder without U-Net skip connections."""

    def __init__(self, in_channels: int, out_channels: int, width: int):
        super().__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, width),
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
            DoubleConv(2 * width, 2 * width),
            nn.Conv2d(2 * width, 4 * width, 4, stride=2, padding=1),
            DoubleConv(4 * width, 4 * width),
            nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1),
            DoubleConv(8 * width, 8 * width),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8 * width, 4 * width, 4, stride=2, padding=1),
            DoubleConv(4 * width, 4 * width),
            nn.ConvTranspose2d(4 * width, 2 * width, 4, stride=2, padding=1),
            DoubleConv(2 * width, 2 * width),
            nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1),
            DoubleConv(width, width),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class PatchDiscriminator(nn.Module):
    def __init__(self, condition_channels: int, target_channels: int, width: int):
        super().__init__()
        in_channels = condition_channels + target_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
            nn.InstanceNorm2d(2 * width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * width, 4 * width, 4, stride=2, padding=1),
            nn.InstanceNorm2d(4 * width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * width, 1, 3, padding=1),
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([condition, target], dim=1))


def frequency_codes(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = torch.log(torch.tensor(FREQS_GHZ, device=device, dtype=dtype))
    return (values - values.min()) / (values.max() - values.min())


def ldpl_prior(side: torch.Tensor) -> torch.Tensor:
    """Deterministic frequency-aware log-distance prior in normalized dB."""

    batch, _, height, width = side.shape
    distance_pixels = side[:, 3].clamp_min(1.0 / math.sqrt(height**2 + width**2))
    distance_m = (distance_pixels * math.sqrt(height**2 + width**2) * 4.0).clamp_min(1.0)
    frequencies = torch.tensor(FREQS_GHZ, device=side.device, dtype=side.dtype)
    path_gain_db = -(
        32.4
        + 20.0 * torch.log10(frequencies)[None, :, None, None]
        + 30.0 * torch.log10(distance_m)[:, None]
    )
    prior = ((path_gain_db + 120.0) / 40.0).clamp(0.0, 1.0)
    building = side[:, 0:1] > 0.5
    return torch.where(building, torch.zeros_like(prior), prior)


def raw_condition(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, ...]:
    sparse = batch["sparse"].to(device, non_blocking=True)
    obs_mask = batch["obs_mask"].to(device, non_blocking=True)
    side = batch["side"].to(device, non_blocking=True)
    condition = torch.cat([sparse, obs_mask, side], dim=1)
    return condition, sparse, obs_mask, side


class DirectDenseModel(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(
        self,
        condition: torch.Tensor,
        sparse: torch.Tensor,
        obs_mask: torch.Tensor,
        side: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del sparse, obs_mask, side
        return torch.sigmoid(self.core(condition)), None


class ResidualPhysicsUNet(nn.Module):
    """RMEGAN generator: U-Net residual around a deterministic LDPL prior."""

    def __init__(self, raw_channels: int, width: int):
        super().__init__()
        self.core = FullUNet(raw_channels + NF, NF, width)
        self.scale = nn.Parameter(torch.full((NF,), 0.15))

    def forward(
        self,
        condition: torch.Tensor,
        sparse: torch.Tensor,
        obs_mask: torch.Tensor,
        side: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del sparse, obs_mask
        prior = ldpl_prior(side)
        residual = torch.tanh(self.core(torch.cat([condition, prior], dim=1)))
        prediction = prior + self.scale[None, :, None, None] * residual
        return prediction.clamp(0.0, 1.0), prior


class Geo2SigMap(nn.Module):
    """Multi-band adaptation of the published cascaded U-Net architecture.

    U-Net-Iso receives building geometry plus a 3GPP/LDPL-style isotropic prior.
    U-Net-Dir receives building, the first prediction, sparse target-band
    measurements, and an explicit missingness mask. Shared weights are used
    across bands and frequency enters through the deterministic prior.
    """

    def __init__(self, width: int):
        super().__init__()
        self.unet_iso = FullUNet(2, 1, width)
        self.unet_dir = FullUNet(4, 1, width)

    def forward(
        self,
        condition: torch.Tensor,
        sparse: torch.Tensor,
        obs_mask: torch.Tensor,
        side: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del condition
        batch, frequencies, height, width = sparse.shape
        prior = ldpl_prior(side)
        building = side[:, 0:1, :, :]
        building_per_band = building[:, None].expand(batch, frequencies, 1, height, width)
        iso_input = torch.cat(
            [building_per_band, prior[:, :, None]], dim=2
        ).reshape(batch * frequencies, 2, height, width)
        isotropic = torch.sigmoid(self.unet_iso(iso_input)).reshape(
            batch, frequencies, height, width
        )
        directional_input = torch.cat(
            [
                building_per_band,
                isotropic[:, :, None],
                sparse[:, :, None],
                obs_mask[:, :, None],
            ],
            dim=2,
        ).reshape(batch * frequencies, 4, height, width)
        prediction = torch.sigmoid(self.unet_dir(directional_input)).reshape(
            batch, frequencies, height, width
        )
        return prediction, isotropic


def build_generator(model_name: str, raw_channels: int, width: int) -> nn.Module:
    if model_name == "ae":
        return DirectDenseModel(ConvAutoencoder(raw_channels, NF, width))
    if model_name in {"unet", "cgan"}:
        return DirectDenseModel(FullUNet(raw_channels, NF, width))
    if model_name == "rmegan":
        return ResidualPhysicsUNet(raw_channels, width)
    if model_name == "geo2sigmap":
        return Geo2SigMap(width)
    raise ValueError(f"Unknown model {model_name}")


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    mask = valid.expand_as(target)
    if kind == "l1":
        return F.l1_loss(prediction[mask], target[mask])
    if kind == "huber":
        return F.smooth_l1_loss(prediction[mask], target[mask], beta=0.05)
    return F.mse_loss(prediction[mask], target[mask])


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    dx_mask = valid[:, :, :, 1:] & valid[:, :, :, :-1]
    dy_mask = valid[:, :, 1:, :] & valid[:, :, :-1, :]
    dx_mask = dx_mask.expand(-1, NF, -1, -1)
    dy_mask = dy_mask.expand(-1, NF, -1, -1)
    dx = (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]) - (
        target[:, :, :, 1:] - target[:, :, :, :-1]
    )
    dy = (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]) - (
        target[:, :, 1:, :] - target[:, :, :-1, :]
    )
    return 0.5 * (dx[dx_mask].abs().mean() + dy[dy_mask].abs().mean())


@torch.no_grad()
def evaluate(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    hard_consistency: bool,
) -> dict[str, object]:
    generator.eval()
    accumulator = PixelMetricAccumulator(NF)
    for batch in loader:
        condition, sparse, obs_mask, side = raw_condition(batch, device)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)[:, 0]
        evaluation_mask = (
            batch["evaluation_mask"].to(device, non_blocking=True)
            if "evaluation_mask" in batch
            else None
        )
        prediction, _ = generator(condition, sparse, obs_mask, side)
        prediction = apply_observation_consistency(
            prediction.clamp(0.0, 1.0), sparse, obs_mask, hard_consistency
        )
        accumulator.update(prediction, target, valid, obs_mask, evaluation_mask)
    return accumulator.finalize()


@torch.no_grad()
def benchmark(generator: nn.Module, dataset, device: torch.device) -> float:
    example = collate([dataset[0]])
    condition, sparse, obs_mask, side = raw_condition(example, device)
    generator.eval()
    for _ in range(3):
        generator(condition, sparse, obs_mask, side)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    repeats = 10
    for _ in range(repeats):
        generator(condition, sparse, obs_mask, side)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - started) / repeats


@torch.no_grad()
def make_figure(
    generator: nn.Module,
    dataset,
    device: torch.device,
    out_path: Path,
    frequency_index: int,
    hard_consistency: bool,
) -> None:
    example = collate([dataset[0]])
    condition, sparse, obs_mask, side = raw_condition(example, device)
    prediction, intermediate = generator(condition, sparse, obs_mask, side)
    prediction = apply_observation_consistency(
        prediction.clamp(0.0, 1.0), sparse, obs_mask, hard_consistency
    )[0, frequency_index].cpu().numpy()
    target = example["target"][0, frequency_index].numpy()
    valid = example["valid"][0, 0].numpy().astype(bool)
    observed = example["obs_mask"][0, frequency_index].numpy().astype(bool)
    sparse_map = example["sparse"][0, frequency_index].numpy()
    sparse_db = np.full(valid.shape, np.nan, dtype=np.float32)
    sparse_db[observed & valid] = sparse_map[observed & valid] * 40.0 - 120.0
    panels = [(sparse_db, "Sparse observations")]
    if intermediate is not None:
        middle = intermediate[0, frequency_index].detach().cpu().numpy() * 40.0 - 120.0
        middle[~valid] = np.nan
        panels.append((middle, "Physics/coarse branch"))
    prediction_db = prediction * 40.0 - 120.0
    target_db = target * 40.0 - 120.0
    prediction_db[~valid] = np.nan
    target_db[~valid] = np.nan
    panels.extend([(prediction_db, "Prediction"), (target_db, "Ground truth")])
    error = np.abs(prediction_db - target_db)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("white")
    figure, axes = plt.subplots(1, len(panels) + 1, figsize=(3.4 * (len(panels) + 1), 3.5), constrained_layout=True)
    image = None
    for axis, (array, title) in zip(axes[:-1], panels):
        image = axis.imshow(array, cmap=cmap, vmin=-120, vmax=-80)
        axis.set_title(title)
    error_image = axes[-1].imshow(error, cmap=error_cmap, vmin=0, vmax=8)
    axes[-1].set_title("Absolute error")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes[:-1], fraction=0.025, pad=0.01, label="Path gain (dB)")
    figure.colorbar(error_image, ax=axes[-1], fraction=0.05, pad=0.01, label="error (dB)")
    figure.suptitle(f"Complete supervised pipeline | {FREQS_GHZ[frequency_index]:.2f} GHz")
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-root", type=Path, required=True)
    parser.add_argument("--evaluation-coarse-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["unet", "geo2sigmap", "ae", "cgan", "rmegan"],
        required=True,
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "l1", "huber"], default="mse")
    parser.add_argument("--gradient-weight", type=float, default=0.02)
    parser.add_argument("--iso-loss-weight", type=float, default=0.25)
    parser.add_argument("--lambda-pixel", type=float, default=100.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--figure-freq-index", type=int, default=1)
    parser.add_argument("--no-hard-observation-consistency", action="store_true")
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
    if not train_entries or not val_entries or not test_entries:
        raise RuntimeError("XML-grouped split produced an empty partition")
    train_set = CoarseRefinementDataset(train_entries, 2.0, include_sparse_inputs=True)
    val_set = CoarseRefinementDataset(val_entries, 2.0, include_sparse_inputs=True)
    test_set = CoarseRefinementDataset(test_entries, 2.0, include_sparse_inputs=True)
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    side_channels = train_set[0]["side"].shape[0]
    raw_channels = 2 * NF + side_channels
    generator = build_generator(args.model, raw_channels, args.width).to(device)
    discriminator = None
    optimizer_d = None
    if args.model in {"cgan", "rmegan"}:
        condition_channels = raw_channels + (NF if args.model == "rmegan" else 0)
        discriminator = PatchDiscriminator(condition_channels, NF, args.width).to(device)
        optimizer_d = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.lr,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay,
        )
    optimizer_g = torch.optim.AdamW(
        generator.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )
    adversarial = nn.BCEWithLogitsLoss()
    hard_consistency = not args.no_hard_observation_consistency
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    initial = evaluate(generator, val_loader, device, hard_consistency)
    best_value = float(initial["unseen_rmse_db"])
    best_epoch = 0
    best_state = copy.deepcopy(generator.state_dict())
    torch.save(best_state, args.out_dir / "best_generator.pt")
    history: list[dict[str, object]] = []
    stale = 0
    print(
        f"[complete supervised v8] model={args.model} train={len(train_set)} "
        f"val={len(val_set)} test={len(test_set)} raw_channels={raw_channels} "
        f"hard_consistency={hard_consistency} device={device} initial={initial}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        generator.train()
        if discriminator is not None:
            discriminator.train()
        generator_losses: list[float] = []
        discriminator_losses: list[float] = []
        started = time.time()
        for batch in train_loader:
            condition, sparse, obs_mask, side = raw_condition(batch, device)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            discriminator_condition = condition
            if args.model == "rmegan":
                discriminator_condition = torch.cat([condition, ldpl_prior(side)], dim=1)

            if discriminator is not None and optimizer_d is not None:
                with torch.no_grad():
                    fake, _ = generator(condition, sparse, obs_mask, side)
                    fake = apply_observation_consistency(
                        fake.clamp(0.0, 1.0), sparse, obs_mask, hard_consistency
                    )
                real_logits = discriminator(discriminator_condition, target)
                fake_logits = discriminator(discriminator_condition, fake)
                discriminator_loss = 0.5 * (
                    adversarial(real_logits, torch.ones_like(real_logits))
                    + adversarial(fake_logits, torch.zeros_like(fake_logits))
                )
                optimizer_d.zero_grad(set_to_none=True)
                discriminator_loss.backward()
                optimizer_d.step()
                discriminator_losses.append(float(discriminator_loss.detach().cpu()))

            prediction, intermediate = generator(condition, sparse, obs_mask, side)
            prediction = apply_observation_consistency(
                prediction.clamp(0.0, 1.0), sparse, obs_mask, hard_consistency
            )
            pixel = reconstruction_loss(prediction, target, valid, args.loss)
            gradients = gradient_loss(prediction, target, valid)
            generator_loss = pixel + args.gradient_weight * gradients
            if args.model == "geo2sigmap" and intermediate is not None:
                generator_loss = generator_loss + args.iso_loss_weight * reconstruction_loss(
                    intermediate, target, valid, args.loss
                )
            if discriminator is not None:
                logits = discriminator(discriminator_condition, prediction)
                generator_loss = args.lambda_pixel * generator_loss + adversarial(
                    logits, torch.ones_like(logits)
                )
            optimizer_g.zero_grad(set_to_none=True)
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            optimizer_g.step()
            generator_losses.append(float(generator_loss.detach().cpu()))

        scheduler.step()
        validation = evaluate(generator, val_loader, device, hard_consistency)
        record = {
            "epoch": epoch,
            "generator_loss": float(np.mean(generator_losses)),
            "discriminator_loss": (
                float(np.mean(discriminator_losses)) if discriminator_losses else 0.0
            ),
            "val": validation,
            "seconds": time.time() - started,
        }
        history.append(record)
        if validation["unseen_rmse_db"] < best_value:
            best_value = float(validation["unseen_rmse_db"])
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(generator.state_dict())
            torch.save(best_state, args.out_dir / "best_generator.pt")
        else:
            stale += 1
        (args.out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:03d}] g={record['generator_loss']:.5f} "
                f"d={record['discriminator_loss']:.5f} "
                f"val_unseen={validation['unseen_rmse_db']:.3f} "
                f"sec={record['seconds']:.1f}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            print(f"[early stop] epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    generator.load_state_dict(best_state)
    validation = evaluate(generator, val_loader, device, hard_consistency)
    test = evaluate(generator, test_loader, device, hard_consistency)
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
        evaluation_set = CoarseRefinementDataset(
            evaluation_test_entries, 2.0, include_sparse_inputs=True
        )
        evaluation_loader = DataLoader(
            evaluation_set,
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
                "test": evaluate(generator, evaluation_loader, device, hard_consistency),
            }
        )
    inference_ms = benchmark(generator, test_set, device)
    result = {
        "protocol": (
            "complete_pipeline_supervised_xml_grouped_fixed_holdout_v11"
            if cache_metadata.get("protocol") in FIXED_CONTROLLED_CACHE_PROTOCOLS
            else "complete_pipeline_supervised_xml_grouped_v10"
        ),
        "primary_metric": "unseen_rmse_db",
        "method": args.model,
        "input": "raw sparse maps + binary masks + same-XML geometry/TX context",
        "forbidden_inputs": "no IDW, no superpixel values, no MFGAT Stage-1 prediction",
        "geo2sigmap_adaptation": (
            "shared-band cascaded U-Net-Iso/U-Net-Dir with deterministic frequency-aware LDPL prior"
            if args.model == "geo2sigmap"
            else None
        ),
        "hard_observation_consistency": hard_consistency,
        "num_train": len(train_set),
        "num_val": len(val_set),
        "num_test": len(test_set),
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "stage1_operating_point": cache_metadata.get(
            "stage1_split_mode", "transductive"
        ),
        "cache_provenance": cache_metadata,
        "mask_evaluations": mask_evaluations,
        "split_groups": split_groups,
        "num_parameters": sum(parameter.numel() for parameter in generator.parameters()),
        "inference_ms_per_map": inference_ms,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "best_epoch": best_epoch,
        "best_val_unseen_rmse_db": best_value,
        "val": validation,
        "test": test,
        "freqs_ghz": FREQS_GHZ,
        "args": {
            key: jsonable_argument(value)
            for key, value in vars(args).items()
        },
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    make_figure(
        generator,
        test_set,
        device,
        args.out_dir / "prediction_example.png",
        args.figure_freq_index,
        hard_consistency,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
