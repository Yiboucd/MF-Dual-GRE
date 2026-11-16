# Original checkpoints at 2% observations per carrier

Two original model files are attached to the
[versioned release](https://github.com/Yiboucd/MF-Dual-GRE/releases/tag/data-v1.0.0):

- [MFGAT Phase 1](https://github.com/Yiboucd/MF-Dual-GRE/releases/download/data-v1.0.0/mfgat-phase1-2pct-seed2026.pt)
- [MF-Dual-GRE Phase 2](https://github.com/Yiboucd/MF-Dual-GRE/releases/download/data-v1.0.0/mf-dual-gre-phase2-2pct-seed2026.pt)

Both use model seed 2026 and the 8,000-region Phase-1 setting. Their exact sizes
and SHA-256 values are in [models.json](models.json) and
[SHA256SUMS.txt](SHA256SUMS.txt). These files are byte-identical to the original
checkpoints whose hashes were checked in the manuscript's evaluation record.

The released model classes and cache-projection CLI are documented in
[code/README.md](../code/README.md). The Phase-2 refiner receives a frozen
Phase-1 prior together with sparse measurements and context. The manuscript's
fine-grid scores retain the original prior caches; regenerating graph
templates or prior caches in another environment is not, by itself, an exact
replay of those scores. The complete collection of per-rate checkpoints and
original cached tensors is not part of this release.
