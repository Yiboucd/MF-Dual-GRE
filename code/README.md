# MF-Dual-GRE source

This directory contains the project source snapshot used to construct and
evaluate MFGAT and the shared-frequency residual refiner. Module filenames are
retained so that the original imports work. Source hashes and the small
portability changes are listed in `../provenance/code_sources.csv`.

## Environment

From the repository root, use Python 3.10 and install:

```sh
python -m pip install -r requirements-code.txt
```

The listed versions passed CPU imports and CLI-help checks on 17 November 2026.
They describe that source-check environment, not a recovered lockfile for every
historical training run. GPU training additionally requires a PyTorch build
compatible with the local GPU and driver. The code uses native PyTorch graph
attention; PyTorch Geometric is not required.

## Data layout and portable paths

Extract the map archives and geometry archive into the same dataset directory.
An example manifest resides at:

```text
data/dataset/<city>/<xml-export>/<transmitter>/manifest.json
```

Release manifests declare `release_path_base: "dataset_root"`. Their map paths,
scene XML paths and `source_coverage_dir` are relative to `dataset/`. The loader
binds those paths when reading a manifest, so no original server directory is
required. Absolute paths remain supported for newly generated local data.
Building masks are read from `source_coverage_dir`, preferring
`B_FOOTPRINT_GEOM_1024m_4m.npy` and otherwise using `B_MAP_GEOM_1024m_4m.npy`.

For direct Python access, add `code/` to the import path and use
`load_manifest(Path(...))` before `load_maps(manifest)` from
`build_clean_superpixel_graphs`. The returned map tensor has shape
`(height, width, 7)`, in GHz order `2.4, 3.65, 4.9, 5.8, 24, 28, 39`.
Stored values are path gain in dB. `rss_to_norm` maps them to the shared clipped
training range using `clip((gain + 120) / 40, 0, 1)`.

## Pretrained checkpoints

The original 2%-per-carrier, seed-2026 checkpoints and checksums are linked in
[data/CHECKPOINTS.md](../data/CHECKPOINTS.md). Save the two release assets under
`data/checkpoints/`. The recorded architecture arguments accompany them in
`data/model_config_phase1_2pct.json` and `data/model_config_phase2_2pct.json`.

```sh
python code/load_pretrained.py --phase 1 \
  --checkpoint data/checkpoints/mfgat-phase1-2pct-seed2026.pt \
  --config data/model_config_phase1_2pct.json
python code/load_pretrained.py --phase 2 \
  --checkpoint data/checkpoints/mf-dual-gre-phase2-2pct-seed2026.pt \
  --config data/model_config_phase2_2pct.json
```

This CPU check verifies the file hash, constructs the original model class and
loads every state-dictionary entry strictly. For Python use, `load_phase1` and
`load_phase2` in `load_pretrained` return `(model.eval(), config)`. The loader is
a release adapter; it does not alter either model's forward computation.

## Supported workflow

These are example invocations of the released CLIs. Run them from the
repository root. Change the dataset and output paths for the local installation.
Use `--help` on any entry point for all arguments.

### 1. Construct geometry-derived graph templates

```sh
python code/build_clean_superpixel_graphs_v2.py \
  --data-root ./data/dataset --out-root ./work/graphs \
  --cities Davis,Woodland,Vacaville --superpixels 8000 \
  --known-percent 2 --seed 2026
```

The v2 builder constructs regions from building distance and coordinates,
without using radio values to define the regions. The template files also
contain reference values for scoring. The controlled trainer below draws its
own observations and holdout; it does not use the template's legacy sparse
sampling as the stated training budget.

### 2. Train Phase 1 from sparse observations

```sh
python code/train_stage1_controlled_v10.py \
  --graph-root ./work/graphs --out-dir ./work/mfgat_2pct \
  --model mfgat --mfgat-architecture frequency_joint_anchor \
  --rates 2 2 2 2 2 2 2 --target-indices 0 1 2 3 4 5 6 \
  --mask-mode independent --auxiliary-mode true \
  --split-mode xml_inductive \
  --fixed-test-rate-percent 20 --fixed-test-seed 9001 \
  --mask-seed 3101 --seed 2026 --epochs 90 \
  --num-layers 2 --batch-size 1 --dropout 0.05 --lr 0.001 \
  --smoothness-weight 0 --patience 12 --device cuda:0
```

`--rates` are percentages of valid free-space pixels for each carrier. The
training loss reconstructs hidden sparse anchors, and checkpoint selection
uses anchors reserved within that measurement budget. Dense reference maps
are accessed for evaluation, not as Phase-1 training labels. `xml_inductive`
trains across training examples and freezes the resulting model for test
examples; the alternative `transductive` mode is a different setting.

The trainer's export-ID split retains the original 820/180/160 partition.
The paper's primary 120-example test cohort is the subsequent complete-source
identity correction, with the other 40 examples reported separately. Use the
released split lists when reporting these cohorts. Do not equate the trainer's
unfiltered 160-example metric with the paper's 120-example primary result.

### 3. Project a frozen Phase-1 model into a fine-grid cache

```sh
python code/precompute_stage1_controlled_coarse_v10.py \
  --run-dir ./work/mfgat_2pct --out-root ./work/coarse_2pct \
  --prior-method model --device cuda:0
```

The cache records the Phase-1 priors, sparse observations and masks,
environmental context, dense targets, and fixed evaluation masks. It preserves
the run's observation protocol. A separate `--mask-seed-override` can generate
another observation realization; it does not create another model-training seed.

### 4. Train the optional dense-supervised refiner

```sh
python code/train_stage2_refiner_v12.py \
  --coarse-root ./work/coarse_2pct --out-dir ./work/refiner_2pct \
  --include-sparse-inputs --cross-frequency-input --frequency-mixer \
  --hard-observation-consistency \
  --width 48 --depth 3 --channel-cap 192 --epochs 80 \
  --patience 25 --amp --device cuda:0
```

This is the shared-frequency residual U-Net refiner with bottleneck frequency
attention. Phase 1 remains frozen, and dense source-scene maps supervise
Phase 2. Therefore the complete two-phase system has a different supervision
requirement from MFGAT alone. The refiner uses joint spatial flips and rotations
during training.

The main entry points are `train_stage1_controlled_v10.py` and
`train_stage2_refiner_v12.py`. Older modules are retained as their import
dependencies, rather than as alternative recommended experiment protocols.
`train_stage2_complete_cnn_baselines_v10.py` supplies baseline helpers used by
the refiner.

## Ray-tracing source

Ray tracing is optional for consuming the released maps. The generator API is
Sionna 0.19.0; it differs from the later Sionna RT API. Use a separate environment
with `requirements-rt.txt` and the system dependencies required by Sionna.

The original single-transmitter generator accepts an XML scene directly:

```sh
python code/generate_clean_single_tx_multifreq_example.py \
  --xml ./data/dataset/geometry/<city>/<export>/<scene>.xml \
  --out-dir ./work/generated_example \
  --x -472 --y -4 --h 50 --tilt 0 --samples 200000000
```

Replace the XML and transmitter values with those of the desired manifest.
This command generates seven simulated maps, rather than importing field
measurements. Its small output manifest is not a complete benchmark manifest
with city/export identity and geometry-mask metadata.

The original batch generator is also included as source. Its discovery CLI
expects the original `coverage_map/info.txt` preprocessing tree and historical
TX filenames. The portable release reorganizes those files, so the batch
discovery command is not a turnkey command for the portable archive. The
source files do not include an automated OSM-to-Blender city conversion.

## Verification and scope

`../provenance/code_validation.json` records syntax, CPU imports and CLI-help
checks, portable sample loading, and strict loading of both original
checkpoints against their released architecture configurations. No new
training, model forward pass or RT execution was performed during release
preparation. The dataset documentation identifies what arrays, scene assets
and splits are included; a source snapshot is not itself a claim that all
historical checkpoints or original execution environments are distributed.

Imported libraries retain their own licenses. Scene and map licensing is
described separately in the repository's data documentation.
