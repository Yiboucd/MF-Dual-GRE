#!/usr/bin/env python3
"""Load released checkpoints with recorded architecture arguments on the CPU.

This release adapter calls the project model constructors without changing
their architecture, prediction rule, or observation-consistency settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def read_config_and_weights(checkpoint: Path, config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = str(config["checkpoint_sha256"])
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint.name}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return config, state


def load_phase1(checkpoint: Path, config_path: Path):
    from train_stage1_controlled_v10 import build_model

    config, state = read_config_and_weights(checkpoint, config_path)
    model, _, _ = build_model(argparse.Namespace(**config["args"]))
    model.load_state_dict(state, strict=True)
    return model.eval(), config


def load_phase2(checkpoint: Path, config_path: Path):
    from train_stage2_refiner_v12 import AuxAdapter, MFRefinerV12

    config, state = read_config_and_weights(checkpoint, config_path)
    args = config["args"]
    core = MFRefinerV12(
        side_channels=int(config["side_channels"]),
        width=int(args["width"]),
        depth=int(args["depth"]),
        channel_cap=int(args["channel_cap"]),
        include_sparse_inputs=bool(args["include_sparse_inputs"]),
        cross_frequency_input=bool(args["cross_frequency_input"]),
        frequency_mixer=bool(args["frequency_mixer"]),
        frequency_mixer_heads=int(args["frequency_mixer_heads"]),
        aux_input=args.get("aux_coarse_root") is not None,
        projection_dim=int(args["projection_dim"]),
        use_skips=not bool(args.get("no_skips", False)),
        physics_prior=bool(args.get("physics_prior", False)),
    )
    model = AuxAdapter(core)
    model.load_state_dict(state, strict=True)
    return model.eval(), config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    loader = load_phase1 if args.phase == 1 else load_phase2
    model, config = loader(args.checkpoint, args.config)
    print(json.dumps({
        "phase": args.phase,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "strict_state_dict_load": True,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": "cpu",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
