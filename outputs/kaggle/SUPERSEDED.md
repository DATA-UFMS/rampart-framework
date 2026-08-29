# Which runs the paper reads, and which it does not

This directory holds every cloud run the study produced, including runs whose numbers
were later found to be wrong. They are kept rather than deleted: a claim that was
withdrawn is evidence about the study, and the diary of failures depends on the logs
that produced them still existing. But a reader grepping for a figure will hit both the
current value and the retracted one, so this file says which is which.

Three of the paper's own errors came from reading the wrong table in a multi-panel log.
Do not skip this file.

## Authoritative — these are the runs the paper reads

| run | supplies |
|---|---|
| `rerun-rampart-r3c-fix5-wb-all` | channels, absorption, correlations, paired contrasts with the Nadeau-Bengio dual, the weight w, and fold-resampled r/Lin intervals -- four World Bank configurations |
| `rerun-rampart-r3c-fix5-inep` | the same for the INEP panel (capped regime) |
| `rerun-rampart-r3c-fix5-inep-uncapped` | the classical ladder on INEP with the cap off: the panel's valid sample-size control, and absorption at the 0.029% share -- the third point of the share staircase |
| `rerun-rampart-r3c-fix5-singly-wb`, `-inep` | the single-probe kNN sweep: batch column reproduces tab_calibration digit for digit; batch-single is the measured batch term, single-closed the panel term |
| `rerun-rampart-r3c-fix5-routes-wb`, `-inep` | the decay curve and buffer widths, now with the NB dual on the GAP2-LEAK contrast |
| `rerun-rampart-r3c-fix2-pertinep` | the matched-share absorption arm (313 probes on INEP), same protocol vintage |
| `rerun-rampart-r3c-rs-wb`, `-inep-f01/-f23/-f45/-f67` | the randomized-saturation audit (interference reframe): 5 classical models, saturations 0.05-0.30 plus the s=0 clean arm, 40 fixed-size SRS replicates per cell, per-row losses in `rampart/replicated_saturation_*.parquet` (gitignored, ~150 MB; re-download with `kaggle kernels output`). Reduced by `scripts/validation/analyze_replicated_saturation.py` to S(s), D(s), B(s) with design-based t intervals over replicates |
| `rerun-rampart-r3c-em-wb`, `-inep-r0/r5/.../r35` | the exposure mapping S(s,d): distances 0-6, 8, 10 (interior arms withhold their year and rebuild lags; bit-identity invariance selftest), saturations 0.10/0.30, 40 replicates sharded by blocks of 5. Per-row losses in `rampart/exposure_mapping_*.parquet` (gitignored). Reduced by `analyze_exposure_mapping.py` to the S(s,d) curves and the interference radius at 1/2/5% equivalence margins; canonical cells in `em_cell_estimates.parquet` (61,200 rows) |
| `rerun-rampart-r3c-rs-mlp-wb`, `-rs-mlp-inep`, `-em-mlp-wb`, `-em-mlp-inep-d0..d10` | the neural rung (`ladder_mlp`, opt-in via `RAMPART_MODELS`; see `neural_rung()` in ladder.py) run through the identical rs and em designs after the classical audit; the INEP exposure mapping is sharded by distance (`RAMPART_DISTANCES`) because a 12h Kaggle session holds one distance of 40 replicates. Consolidated into the same canonical parquets (`rs_cell_estimates.parquet` now 24,960 rows, `em_cell_estimates.parquet` 73,440), with the classical rows verified bit-identical before and after |

Every table and figure in the paper is regenerated from these by
`paper_tkdd/make_tables_interference.py`, `make_figures_interference.py` and `make_drivers.py`
(`make_tables.py` only emits the legacy calibration table). Nothing is transcribed by hand.

## Superseded — do not cite these numbers

| run | superseded because | numbers in it that the paper no longer uses |
|---|---|---|
| `rerun-rampart-r3c-routes-wb` | the decay curve omitted the `RESERVE_NEAR` arm, so buffer widths were read across a 7.8-year stretch containing no measurement — the exact gap that arm exists to close | buffer widths **4.3 / 7.8 / 11.2** (now 4.3 / 8.0 / 11.4) |
| `rerun-rampart-r3c-routes-inep` | same defect | buffer widths **3.4 / 5.5 / 7.2** (now 3.4 / 5.1 / 6.5) |
| `rerun-rampart-r3c-wb-all` | intervals ran at 4,000 resamples while the protocol declared 15,000, and the correlation section counted the duplicate rung `ladder_knn` (= `knn_k5`) twice | r(absorption, memorisation) floor **0.973** (now 0.972); r vs aggregate **0.680** at 5% (now 0.667); all interval widths |
| `rerun-rampart-r3c` | same two defects, INEP panel | same |
| `rerun-rampart-r3c-fix-wb-all` | resample count fixed, duplicate rung still double-counted | r floor **0.973** |
| `rerun-rampart-r3c-fix-inep` | same | same |
| `rerun-rampart-r3c-fix2-*`, `fix3-*`, `fix4-*`, `fix-routes-*` | every measured value identical to its fix5 counterpart (verified: tab_channels regenerates byte-identical across fix3/fix4/fix5; the decay tables match line for line); each generation only adds printed receipts -- fix3 paired contrasts and w, fix4 r/Lin fold intervals, fix5 the Nadeau-Bengio dual | none -- values live on in fix5 |
| `rampart-r3c-pertwb`, `rampart-r3c-pertinep`, `rampart-r3c-uncapped` | earlier protocol vintage (one replicate draw, pre-reordering frame): the same ridge at the same twelve probes reads 0.2894 there against 0.3287 under the current protocol. The vintage difference was once misread as a cap effect | absorption **0.2894 / 0.4055 / 0.1699 / 0.2598 / 0.0035 / 0.0484** |

## The trap that produced two of the study's own errors

A single log can carry **several panels**, and Kaggle **duplicates every stdout block**.
`rerun-rampart-r3c-routes-wb` contains both `worldbank` and `worldbank_clean`, each
printed twice — four decay tables with two different sets of numbers. Reading "the last
table in the file" returns `worldbank_clean` while appearing to return `worldbank`.

Select the panel explicitly by its header and read the first table inside that block:

```python
import json
def stdout(path):
    return ''.join(x['data'] for x in json.load(open(path))
                   if x['stream_name'] == 'stdout')

def panel_block(text, panel):
    start = text.index(f'on {panel}:')
    nxt = text.find('--- channels on ', start + 10)
    return text[start:nxt if nxt > 0 else len(text)]
```

Both generator scripts in `paper_tkdd/` do this and say so in their docstrings.

## Checking a log against itself

Every run produced after the fixes prints a provenance block at the start and a resample
audit at the end. The audit reports the resample counts that actually executed, against
the one the configuration declares:

```
--- resample audit ---
    236 intervals at  15000 resamples
```

More than one line means the run mixed counts and no single number describes its output.
The superseded runs above predate this block, which is why their defect had to be found
by reading source rather than by reading the log.
