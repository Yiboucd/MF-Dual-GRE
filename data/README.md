# Multi-band radio map dataset

The release contains **1,160 simulated examples** with seven co-registered
`256 × 256` path-gain maps per example. Frequencies are 2.4, 3.65, 4.9, 5.8,
24, 28 and 39 GHz. Each map covers `1,024 × 1,024 m²` at 4 m resolution.
Values are float32 **path gain in dB**, stored as `10 log10(G + 10⁻¹²)`.
They are not received power in dBm. The released arrays retain their stored
values; model-side clipping to [−120, −80] dB is a separate preprocessing step.

Maps were generated using OpenStreetMap-derived Blender scenes and Sionna RT
0.19.0, with `2 × 10⁸` ray samples per map. There are 380 examples from Davis,
360 from Woodland and 420 from Vacaville. The 58 scene export groups correspond
to **55 distinct complete scene sources**; three Davis scenes were sampled twice
with different transmitter sets.

## Download

The [versioned GitHub release](https://github.com/Yiboucd/MF-Dual-GRE/releases/tag/data-v1.0.0)
contains a small sample archive, three full city archives, and scene geometry.
[artifacts.json](artifacts.json) lists exact sizes and SHA-256 checksums.

From the repository root:

```sh
python -m pip install -r requirements-data.txt
python tools/download_data.py --subset sample
python tools/inspect_example.py

# Full dataset, including all scene geometry and building masks
python tools/download_data.py --subset full
python tools/inspect_example.py --all
```

To download only one city, use `--subset full --cities Davis`. Archives can
also be downloaded manually and extracted into a common dataset root. All
manifest paths are relative to that root. The implementation's manifest loader
resolves these paths from the dataset location; see [the code guide](../code/README.md).

## Data and split index

[examples.csv](examples.csv) contains one row per example, preserving its
original index position and stable sample, transmitter, export and source
identities. `manifest` is relative to the extracted dataset root.
`original_manifest_sha256` refers to the original generation metadata before
release paths were normalized; it is not the checksum of the published JSON.
The radio-map arrays themselves have not been modified.

| File | Examples | Use |
| --- | ---: | --- |
| [train.csv](train.csv) | 820 | Training |
| [val.csv](val.csv) | 180 | Validation |
| [source_disjoint_test120.csv](source_disjoint_test120.csv) | 120 | Primary test: complete sources absent from training and validation |
| [seen_source_new_tx_test40.csv](seen_source_new_tx_test40.csv) | 40 | Additional test: new transmitters in sources shared with training or validation |
| [original_test160.csv](original_test160.csv) | 160 | Original aggregate, containing both test groups |

The 120/40 distinction follows a source-identity audit of the original test
partition. It does not establish geographic separation or cross-city testing.
Use the 120-example cohort for the manuscript's primary comparison.

The fixed held-out free-space pixels are excluded from observations in every
band. The manuscript's primary equal-per-band rates are 0.1%, 0.25%, 0.5%, 1%,
2%, 4% and 8%. Training seed, observation-mask seed and holdout seed are separate
quantities. Use the implementation's protocol rather than treating five input
masks as five independently trained models.

## Loading

```python
from pathlib import Path
from tools.inspect_example import load_example

root = Path('data/dataset').resolve()
manifest = next(root.glob('*/xml_*/tx_*/manifest.json'))
maps_db, building_mask, metadata = load_example(root, manifest)
# maps_db: [height, width, band], float32, in the frequency order above
# building_mask: [height, width], bool; True marks building cells
valid_mask = ~building_mask
```

Manifest records associate each frequency with its NPY file. The manifest also
identifies transmitter configuration and source geometry. Scene XML/PLY files
and source building masks are distributed in the geometry archive.

See [DATA_LICENSE.md](../DATA_LICENSE.md) for ODbL licensing and
**© OpenStreetMap contributors** attribution.

The [regional RadioGAT training record](radiogat_regional_0p1_training_record.json)
preserves the configured epoch limit and selected epoch for the archived
0.1% regional run.
