#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np

try:
    import tensorflow as tf

    for _gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(_gpu, True)
        except Exception:
            pass
except Exception:
    tf = None

from sionna.rt import PlanarArray, Transmitter, load_scene


CITY_CONFIGS: dict[str, dict[str, Any]] = {
    "Davis": {
        "folder_date": "Davis_2025-05-06_18-27_c38754",
        "coverage_prefix": "Davis_new",
        "start": 1,
        "end": 19,
    },
    "Woodland": {
        "folder_date": "Woodland_2025-05-06_18-39_c1c506",
        "coverage_prefix": "Woodland_new",
        "start": 1,
        "end": 18,
    },
    "Vacaville": {
        "folder_date": "Vacaville_2025-05-20_14-05_5d1dd4",
        "coverage_prefix": "Vacaville_new_new",
        "start": 1,
        "end": 21,
    },
    "SAC": {
        "folder_date": "SAC_2025-07-11_23-02_3c69da",
        "coverage_prefix": "SAC",
        "start": 1,
        "end": 53,
    },
    "HK": {
        "folder_date": "HK_2025-08-22_12-08_c095dc",
        "coverage_prefix": "HK",
        "start": 1,
        "end": 53,
    },
}


FNAME_PAT = re.compile(
    r"^txx_(?P<x>-?\d+(?:\.\d+)?)_y_(?P<y>-?\d+(?:\.\d+)?)_h(?P<h>\d+)_"
    r"f(?P<fg>\d+\.\d+)G_tilt(?P<tilt>-?\d+)_"
    r"(?P<kind>with_building|free_space)\.npy$"
)


def parse_freqs(text: str) -> list[float]:
    freqs = []
    for token in text.split(","):
        token = token.strip()
        if token:
            freqs.append(float(token))
    if not freqs:
        raise ValueError("No frequencies supplied")
    return freqs


def read_uuid(info_path: Path) -> str | None:
    try:
        for line in info_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("uuid:"):
                return line.split(":", 1)[1].strip().lower()
    except OSError:
        return None
    return None


def extract_tx_combos(folder: Path, prefer_freq_tag: str | None) -> list[tuple[float, float, int, int]]:
    preferred: set[tuple[float, float, int, int]] = set()
    fallback: set[tuple[float, float, int, int]] = set()
    for path in folder.glob("*.npy"):
        match = FNAME_PAT.match(path.name)
        if not match or match.group("kind") != "with_building":
            continue
        combo = (
            float(match.group("x")),
            float(match.group("y")),
            int(match.group("h")),
            int(match.group("tilt")),
        )
        fallback.add(combo)
        freq_tag = f"f{float(match.group('fg')):.2f}G"
        if prefer_freq_tag is None or freq_tag == prefer_freq_tag:
            preferred.add(combo)
    return sorted(preferred or fallback)


def discover_tasks(
    base_dir: Path,
    cities: list[str],
    prefer_freq_tag: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for city in cities:
        cfg = CITY_CONFIGS[city]
        source_root = base_dir / cfg["folder_date"]
        coverage_root = source_root / "coverage_map"
        xml_root = source_root / "Bl_xml_files"
        for xml_idx in range(int(cfg["start"]), int(cfg["end"]) + 1):
            coverage_dir = coverage_root / f"{cfg['coverage_prefix']}_{xml_idx}"
            info_path = coverage_dir / "info.txt"
            if not coverage_dir.is_dir() or not info_path.exists():
                warnings.append({"city": city, "xml_idx": xml_idx, "warning": "missing_coverage_or_info"})
                continue
            uuid = read_uuid(info_path)
            if not uuid:
                warnings.append({"city": city, "xml_idx": xml_idx, "warning": "missing_uuid"})
                continue
            xml_path = xml_root / uuid / f"{uuid}.xml"
            if not xml_path.exists():
                warnings.append({"city": city, "xml_idx": xml_idx, "uuid": uuid, "warning": "missing_xml"})
                continue
            txs = extract_tx_combos(coverage_dir, prefer_freq_tag)
            if not txs:
                warnings.append({"city": city, "xml_idx": xml_idx, "uuid": uuid, "warning": "no_tx_combos"})
                continue
            for tx_idx, (x, y, h, tilt) in enumerate(txs, start=1):
                tasks.append(
                    {
                        "city": city,
                        "folder_date": cfg["folder_date"],
                        "coverage_prefix": cfg["coverage_prefix"],
                        "source_root": str(source_root),
                        "source_coverage_dir": str(coverage_dir),
                        "xml_idx": xml_idx,
                        "uuid": uuid,
                        "xml": str(xml_path),
                        "tx_idx": tx_idx,
                        "x": x,
                        "y": y,
                        "h": h,
                        "tilt": tilt,
                    }
                )
    return tasks, warnings


def radio_filename(task: dict[str, Any], freq_ghz: float) -> str:
    return (
        f"txx_{task['x']:.1f}_y_{task['y']:.1f}_h{int(task['h'])}_"
        f"f{freq_ghz:.2f}G_tilt{int(task['tilt'])}_with_building.npy"
    )


def task_out_dir(out_root: Path, task: dict[str, Any]) -> Path:
    city = task["city"]
    xml_part = f"xml_{int(task['xml_idx']):03d}_{task['uuid']}"
    tx_part = (
        f"tx_{int(task['tx_idx']):03d}_"
        f"x{task['x']:.1f}_y{task['y']:.1f}_h{int(task['h'])}_tilt{int(task['tilt'])}"
    )
    tx_part = tx_part.replace("-", "m")
    return out_root / city / xml_part / tx_part


def generate_map(task: dict[str, Any], freq_ghz: float, samples: int) -> np.ndarray:
    scene = load_scene(task["xml"])
    scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="dipole", polarization="cross")
    scene.add(
        Transmitter(
            "tx",
            [float(task["x"]), float(task["y"]), int(task["h"])],
            [np.deg2rad(int(task["tilt"])), 0, 0],
        )
    )
    scene.frequency = float(freq_ghz) * 1e9
    scene.synthetic_array = True
    cm = scene.coverage_map(
        max_depth=5,
        diffraction=True,
        cm_cell_size=(4.0, 4.0),
        cm_center=[0, 0, 1.5],
        cm_orientation=[0, 0, 0],
        cm_size=[1024, 1024],
        num_samples=int(samples),
    )
    arr = 10.0 * np.log10(cm.path_gain.numpy()[0, :, :] + 1e-12)
    del scene, cm
    gc.collect()
    if tf is not None:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
    return arr.astype(np.float32)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".pid{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def file_stats(path: Path) -> dict[str, float]:
    arr = np.load(path)
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def build_manifest(task: dict[str, Any], samples: int, freqs_ghz: list[float]) -> dict[str, Any]:
    return {
        "schema": "clean_multifreq_single_tx_v1",
        "city": task["city"],
        "folder_date": task["folder_date"],
        "source_root": task["source_root"],
        "source_coverage_dir": task["source_coverage_dir"],
        "xml_idx": int(task["xml_idx"]),
        "uuid": task["uuid"],
        "xml": task["xml"],
        "tx_idx": int(task["tx_idx"]),
        "tx": [float(task["x"]), float(task["y"]), int(task["h"])],
        "tilt": int(task["tilt"]),
        "samples": int(samples),
        "frequencies_ghz": freqs_ghz,
        "rt_params": {
            "max_depth": 5,
            "diffraction": True,
            "cm_cell_size": [4.0, 4.0],
            "cm_center": [0, 0, 1.5],
            "cm_orientation": [0, 0, 0],
            "cm_size": [1024, 1024],
            "tx_array": "PlanarArray(1,1,0.5,0.5,iso,V)",
            "rx_array": "PlanarArray(1,1,0.5,0.5,dipole,cross)",
            "synthetic_array": True,
        },
        "records": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def manifest_is_complete(manifest_path: Path, freqs_ghz: list[float]) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    records = manifest.get("records", [])
    by_freq = {round(float(r.get("freq_ghz", -1)), 2): r for r in records}
    for freq in freqs_ghz:
        rec = by_freq.get(round(freq, 2))
        if not rec or not Path(rec.get("path", "")).exists():
            return False
    return True


def process_task(task: dict[str, Any], args: argparse.Namespace, freqs_ghz: list[float]) -> dict[str, Any]:
    out_dir = task_out_dir(args.out_root, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if args.resume and manifest_is_complete(manifest_path, freqs_ghz):
        return {"status": "skip_complete", "generated": 0, "skipped": len(freqs_ghz), "out_dir": str(out_dir)}

    manifest = build_manifest(task, args.samples, freqs_ghz)
    generated = 0
    skipped = 0
    records = []
    for freq in freqs_ghz:
        out_path = out_dir / radio_filename(task, freq)
        t0 = time.time()
        if args.resume and out_path.exists():
            stats = file_stats(out_path)
            skipped += 1
            record = {
                "freq_ghz": float(freq),
                "path": str(out_path),
                "seconds": 0.0,
                "source": "existing",
                **stats,
            }
        else:
            print(
                f"[generate] {task['city']} xml={task['xml_idx']:03d} tx={task['tx_idx']:03d} "
                f"freq={freq:.2f}GHz gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '')} out={out_path.name}",
                flush=True,
            )
            if args.dry_run:
                record = {
                    "freq_ghz": float(freq),
                    "path": str(out_path),
                    "seconds": 0.0,
                    "source": "dry_run",
                    "min": None,
                    "max": None,
                    "mean": None,
                }
            else:
                arr = generate_map(task, freq, args.samples)
                tmp_path = out_path.with_name(f".{out_path.name}.pid{os.getpid()}.tmp.npy")
                np.save(tmp_path, arr)
                os.replace(tmp_path, out_path)
                record = {
                    "freq_ghz": float(freq),
                    "path": str(out_path),
                    "seconds": time.time() - t0,
                    "source": "generated",
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "mean": float(np.nanmean(arr)),
                }
                del arr
                generated += 1
        records.append(record)
        manifest["records"] = records
        manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(manifest_path, manifest)
    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(manifest_path, manifest)
    return {"status": "complete", "generated": generated, "skipped": skipped, "out_dir": str(out_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cities", default="Davis,Woodland,Vacaville,SAC,HK")
    parser.add_argument("--freqs-ghz", default="2.4,3.65,4.9,5.8,24,28,39")
    parser.add_argument("--samples", type=int, default=int(2e8))
    parser.add_argument("--prefer-base-freq-tag", default="f3.65G")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.base_dir = args.base_dir.resolve()
    args.out_root = args.out_root.resolve()

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    unknown = sorted(set(cities) - set(CITY_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown cities: {unknown}")
    freqs_ghz = parse_freqs(args.freqs_ghz)
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    args.out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks, warnings = discover_tasks(args.base_dir, cities, args.prefer_base_freq_tag)
    shard_tasks = [task for i, task in enumerate(tasks) if i % args.num_shards == args.shard_index]
    status_path = logs_dir / f"status_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    summary: dict[str, Any] = {
        "schema": "clean_multifreq_batch_status_v1",
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "out_root": str(args.out_root),
        "cities": cities,
        "freqs_ghz": freqs_ghz,
        "samples": int(args.samples),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "total_tasks_all_shards": len(tasks),
        "tasks_this_shard": len(shard_tasks),
        "discovery_warnings": warnings,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_tasks": 0,
        "generated_maps": 0,
        "skipped_maps": 0,
        "failed": [],
    }
    write_json(status_path, summary)
    print(
        f"[start] shard {args.shard_index}/{args.num_shards} tasks={len(shard_tasks)} "
        f"all_tasks={len(tasks)} out={args.out_root}",
        flush=True,
    )

    for task in shard_tasks:
        try:
            result = process_task(task, args, freqs_ghz)
            summary["completed_tasks"] += 1
            summary["generated_maps"] += int(result.get("generated", 0))
            summary["skipped_maps"] += int(result.get("skipped", 0))
            summary["last_result"] = result
        except Exception as exc:
            failure = {
                "city": task.get("city"),
                "xml_idx": task.get("xml_idx"),
                "tx_idx": task.get("tx_idx"),
                "error": repr(exc),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            summary["failed"].append(failure)
            print(f"[failed] {failure}", flush=True)
        summary["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(status_path, summary)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(status_path, summary)
    print(f"[done] shard {args.shard_index}/{args.num_shards}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
