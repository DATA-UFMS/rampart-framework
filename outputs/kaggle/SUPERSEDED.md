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

## Fleet generations F1 and F2 (30 Aug - 2 Sep 2026): which panel each one is authoritative for

Two fleets ran after the generations above. **Neither is consolidated into
`rs_cell_estimates.parquet` or `em_cell_estimates.parquet`**, and that is a
decision, not an omission: merging them would silently change cells that the
paper's main tables and figures read. Each fleet is read only by its own
receipt emitter, named below. A reader who greps a number from the paper's
main tables will never land in an F1 or F2 shard, and a reader who greps a
number from a robustness paragraph will never land in the canonical parquets.

| generation | authoritative for | read by | NOT used for |
|---|---|---|---|
| `rerun-rampart-r3c-rs-*` + `-rs-mlp-*` (consolidated in `rs_cell_estimates.parquet`, 24,960 rows) | the randomized-saturation audit on World Bank and INEP, six rungs, at the registered seed | `make_tables_interference.py`, `make_figures_interference.py`, `make_drivers.py` | anything about the SINASC panel or about second-generation boosting |
| `rerun-rampart-r3c-em-*` + `-em-mlp-*` (consolidated in `em_cell_estimates.parquet`, 73,440 rows) | the exposure mapping S(s,d) and the interference radius | `make_tables_interference.py` (radius tables) | the same |
| **F1 `rerun-rampart-f1-rs-mlp-{wbclean,inep}-s101..s110`** | the multi-seed sensitivity of the neural rung (P-F1.1): 10 optimizer seeds on World Bank, 3 on INEP | `make_multiseed_receipt.py` -> `multiseed_receipt.txt` | replacing the registered seed anywhere in the paper |
| **F1 `rerun-rampart-f1-rs-boost-{wbclean,inep-fa,inep-fb}`** | the second-generation boosting rungs (P-F1.3): `ladder_xgboost`, `ladder_lightgbm` | `make_boosting_receipt.py` -> `boosting_receipt.txt` | the six-rung ladder of the main tables, which does not include them |
| **F2 `rerun-rampart-f2-rs-*-sinasc-*`** | the whole third panel (SINASC), 14 folds x 6 rungs | `make_sinasc_receipt.py` -> `sinasc_receipt.txt`, `tab_sinasc.tex` | the audit's main tables, figures and fit inventory, which report the two panels above |

### The registered seed is not superseded by the fleet seeds

`rerun-rampart-r3c-rs-mlp-wb` ran the neural rung at the registered seed (42)
and its cells are the ones in `rs_cell_estimates.parquet`. The F1 seeds
101-110 do **not** supersede it and must not be substituted for it: the
prediction P-F1.1 was registered against the seed-42 value, so replacing that
value with a fleet seed, or with the across-seed mean, would be reading the
prediction after the fact. What the fleet establishes is that seed 42 is the
most negative of eleven seeds on World Bank (S(0.30) = -6.26 against a median
of +0.7 and a mean of about +1.5, with across-seed sd about 4.5 against a
conditional half-width of about 1). The paper keeps the registered number and
qualifies every negative sign with that fact.

### Inside F2: the contingency split, and which shard owns which fold

The SINASC fleet is sharded by fold and by rung group. The calibration kernel
`f0-3` measured the classical roster about 1.6x slower than the conservative
budget, which triggered the pre-written contingency: folds 11, 12 and 13 were
split into `cls4` (the four classical rungs without gradient boosting) and
`gb` (gradient boosting alone). Two shards stopped at the 11 h guard with one
fold outstanding and were closed by follow-up kernels:

- fold 3: `rerun-rampart-f2-rs-classical-sinasc-f0-3` reports `skipped=3`.
  The authoritative shard for fold 3 is
  `rerun-rampart-f2-rs-classical-sinasc-f3`.
- fold 7: `rerun-rampart-f2-rs-classical-sinasc-f6-7` reports `skipped=7`.
  The authoritative shard for fold 7 is
  `rerun-rampart-f2-rs-classical-sinasc-f7`.

Every (fold, rung) of the SINASC panel is therefore produced exactly once
across the 17 shards. There are no duplicate folds to choose between, and no
`.superseded-<UTC>` directory was created by any landing in either fleet.
`make_sinasc_receipt.py` prints the provenance of each shard it reads, and
refuses to emit the table unless all 14 folds are present.

### Runs whose numbers exist but were never read, and non-landings

- Kernels that returned `Status.ERROR` within about 11 minutes (F1
  `s103-fb` v1; F2 classical `f9` v1 and `f7` v1) were retried at identical
  configuration and completed. The failed attempt produces no landing and no
  parquet; its log is not retrievable after the retry. It is a Kaggle
  transient, not a defect in a measured value: `f9` v2 at identical
  configuration completed in 5.48 h.
- A provisional SINASC receipt (`sinasc_receipt.provisional.txt`) existed
  while the classical rungs covered only 7-8 of 14 folds, with **different
  fold sets per rung**, which makes any comparison between rungs invalid.
  It was deleted when the definitive receipt was emitted at 14/14. Do not
  reconstruct it: fold heterogeneity on this panel is enormous (ridge
  S(0.30) reads 127/145/123 on folds 0-2 against 4.6/3.8/2.7 on folds 8-10),
  so a partial fold set is not a noisy version of the answer, it is a
  different quantity.
- The simulated neural mechanism (commit `742eb14`, prediction P-F1.2) was
  built and committed, and its full 6,000-draw run was **never executed**.
  `sim_groundtruth_receipt.txt` declares on its first line the mechanisms it
  covers (`lookup/knn1/ridge/tree`) and does not include it. No number from
  a simulated neural mechanism appears anywhere.

### What the F1 and F2 shards contain, and what is deposited

Per-row, per-replicate parquets: 131 files and about 298 MB for F1, 31 files
and about 650 MB for F2. They are not archived (the pipeline is deterministic
by construction; code plus logs plus seeds regenerate them). The kernel log of
every shard is deposited, with the clone URL anonymised for double-anonymous
review, together with an `INVENTORY.md` listing every landed shard, its folds,
its rung set and its parquet sizes.

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
