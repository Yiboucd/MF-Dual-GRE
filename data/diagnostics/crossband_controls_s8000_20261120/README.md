# S8000 cross-frequency controls: November 20 snapshot

This snapshot contains all eight configurations completed at the time of
collection from an eleven-configuration queue. The three 28 GHz fixed-budget
configurations were still running or queued. Missing results are not imputed.
The original eleven-arm plan and the queue snapshot are included.

## Protocol

The requested graph size is 8,000 regions. The source-disjoint split contains
820 training, 180 validation and 120 primary test examples. Each condition is
trained separately with the same MFGAT architecture and target-band hidden
sparse-measurement loss; auxiliary values are inputs rather than extra
supervision targets. Checkpoint selection uses target-band reserved sparse
measurements in validation scenes.

There is one model seed (2026) and five evaluation masks (4101--4105). Each mask
scores 1,420,900 target-band pixels across 120 examples. Pixel SSE and count are
pooled within a mask, then the five RMSEs are averaged. The reported mask
standard deviation does not measure variation across independently trained
models. Values use the benchmark's clipped dB reference scale.

The true and shuffled conditions use identical auxiliary positions. Shuffled
values come from a different physical transmitter in the same source and split,
during both training and evaluation. Both conditions draw auxiliary locations
from the common non-holdout domain of the recipient and donor; target locations
retain recipient-only holdout exclusion. Rates are nested within each band.

## Additional measurements at a fixed target observation rate

The 3.65 GHz target is observed at 0.5%. The 28 GHz auxiliary rate varies.

| Auxiliary rate | No auxiliary | 0.5% | 5% |
| --- | ---: | ---: | ---: |
| True auxiliary RMSE (dB) | 4.466212 | 4.203134 | 3.665665 |
| Different-transmitter auxiliary RMSE (dB) | -- | 4.387570 | 4.396728 |

This completed comparison is the manuscript's auxiliary-information experiment:
additional auxiliary observations from the paired target example improve
reconstruction at fixed target-band observations. These are simulated
observations, not field measurements.

## Fixed-budget diagnostic

This separate diagnostic reallocates measurements rather than adding them.
The summed per-band rates are 2%, equivalent to an overall rate of (2/7)% over
the fixed seven-carrier set. Target-only sampling uses 2% at the target. The
two joint conditions use 0.5% at the target and 0.25% at each other carrier.

| Input | 3.65 GHz target RMSE (dB) | 28 GHz target RMSE (dB) |
| --- | ---: | ---: |
| Target only | 3.905027 | Pending |
| True auxiliaries | 4.019056 | Pending |
| Different-transmitter auxiliaries | 4.380370 | Pending |

For the completed 3.65 GHz comparison, reallocating the budget to true auxiliary
observations increases RMSE by 0.114028 dB relative to target-only sampling.
These results do not support a general claim that redistributing a fixed budget
improves reconstruction. The manuscript focuses on the distinct question of
using additional available auxiliary measurements; this diagnostic and its raw
records are retained here in full. All eleven planned runs remain in the queue.

## Files and checks

- `experiment_plan.json`: unchanged original configuration and protocol plan.
- `queue_status_snapshot.json`: queue state at collection time.
- `controls_summary.csv`: all eight completed means and mask standard deviations.
- `partial_verification.json`: independent recomputation and provenance checks.
- `outputs/<configuration>/`: result, run manifest, five raw per-example CSVs,
  five mask summaries and their prepared-input fingerprints.

All forty raw CSVs were independently checked against the exact primary cohort,
pixel counts, raw/checkpoint hashes, SSE decomposition and stock-evaluator
bridge. Their recomputed means and standard deviations match the run results.
The raw CSV bytes are preserved through Git so their recorded SHA-256 values
remain valid. Model checkpoints themselves are not included in this snapshot.
