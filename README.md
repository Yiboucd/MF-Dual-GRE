# MF-Dual-GRE

Dataset and reference implementation for **MF-Dual-GRE: Joint Multi-Band Graph
Attention and Fine-Resolution Radio Map Reconstruction**, by Yibo Ma, Songyang
Zhang, Yueling Zhou and Zhi Ding.

MF-Dual-GRE combines a sparse-supervised multi-frequency graph attention
estimator (MFGAT, Phase 1) with an optional dense-supervised fine-resolution
refiner (Phase 2). MFGAT jointly estimates carrier maps from sparse observations
whose locations can differ across bands. The refiner uses regional estimates,
the original sparse measurements and scene context.

## Dataset and downloads

**[Download the versioned data release](https://github.com/Yiboucd/MF-Dual-GRE/releases/tag/data-v1.0.0)**

| Property | Value |
| --- | --- |
| Examples | 1,160, each containing seven co-registered maps |
| Frequencies | 2.4, 3.65, 4.9, 5.8, 24, 28, 39 GHz |
| Grid | 256 × 256, 4 m cells, 1,024 × 1,024 m² |
| Quantity | Simulated path gain in dB, float32 NPY |
| Scene sources | 55 distinct 3-D sources in Davis, Woodland and Vacaville |
| Scene exports | 58 groups, with 20 transmitter examples per group |
| Training / validation | 820 / 180 examples |
| Primary / additional test | 120 source-disjoint / 40 shared-source, new-transmitter examples |

The release includes the stored radio maps, portable per-example metadata,
source XML/PLY geometry, building masks, OpenStreetMap source extracts, and
fixed split lists. The radio-map array bytes are preserved. Dataset archives
are separate from the Git repository; sizes and SHA-256 checksums are recorded
in [data/artifacts.json](data/artifacts.json).

## Quick start

Python 3.9 or newer is sufficient to download and inspect the data. Neither a
GPU nor Sionna is needed for this step.

```sh
git clone https://github.com/Yiboucd/MF-Dual-GRE.git
cd MF-Dual-GRE
python -m pip install -r requirements-data.txt
python tools/download_data.py --subset sample
python tools/inspect_example.py
```

For all examples and source geometry:

```sh
python tools/download_data.py --subset full
python tools/inspect_example.py --all
```

See the [data guide](data/README.md) for layout, loading, split identities and
license attribution. See the [implementation guide](code/README.md) for graph
construction, sparse-supervised training, optional refinement and code provenance.
Two [original checkpoints at 2% per-band observations](data/CHECKPOINTS.md) are
provided as separate release assets.

## Experimental protocol

Phase 1 uses sparse radio-value supervision. Phase 2 additionally uses dense
training-scene targets. At test time, the trained model receives sparse
observations and environmental geometry. The dataset is simulated; the release
does not contain field measurements.

The primary test cohort contains 120 examples from six complete scene sources
absent from training and validation. Another 40 examples have new transmitters
in sources shared with training or validation. Both cohorts and the original
160-example aggregate are explicitly indexed in [data/](data/). The source
distinction was established by an audit of the original partition and does
not imply geographic separation.

## Citation

Until a publication identifier is available, cite this versioned repository:

```bibtex
@misc{ma2026mfdualgre,
  author = {Yibo Ma and Songyang Zhang and Yueling Zhou and Zhi Ding},
  title = {{MF-Dual-GRE}: Multi-Band Radio Map Dataset and Implementation},
  year = {2026},
  howpublished = {\url{https://github.com/Yiboucd/MF-Dual-GRE}},
  note = {Data release v1.0.0}
}
```

Machine-readable metadata are provided in [CITATION.cff](CITATION.cff).

## License and attribution

Original implementation code: [MIT](LICENSE). Dataset database and
OpenStreetMap-derived geometry: [ODbL 1.0](DATA_LICENSE.md).
**© OpenStreetMap contributors.** Third-party software retains its own licenses.
