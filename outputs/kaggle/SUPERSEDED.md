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
| `rerun-rampart-r3c-fix4-wb-all` | channels, absorption, correlations, the paired ridge-forest contrasts, the mixture weight w, and fold-resampled intervals for r and Lin's concordance -- four World Bank configurations |
| `rerun-rampart-r3c-fix4-inep` | the same for the INEP panel |
| `rerun-rampart-r3c-fix-routes-wb` | the decay curve and buffer widths, World Bank |
| `rerun-rampart-r3c-fix-routes-inep` | the same for INEP |
| `rerun-rampart-r3c-fix2-pertinep` | the matched-share absorption arm (313 probes on INEP), under the same protocol vintage as everything above |

Every table and figure in the paper is regenerated from these by `paper_tkdd/make_tables.py`
and `paper_tkdd/make_figures.py`. Nothing is transcribed by hand.

## Superseded — do not cite these numbers

| run | superseded because | numbers in it that the paper no longer uses |
|---|---|---|
| `rerun-rampart-r3c-routes-wb` | the decay curve omitted the `RESERVE_NEAR` arm, so buffer widths were read across a 7.8-year stretch containing no measurement — the exact gap that arm exists to close | buffer widths **4.3 / 7.8 / 11.2** (now 4.3 / 8.0 / 11.4) |
| `rerun-rampart-r3c-routes-inep` | same defect | buffer widths **3.4 / 5.5 / 7.2** (now 3.4 / 5.1 / 6.5) |
| `rerun-rampart-r3c-wb-all` | intervals ran at 4,000 resamples while the protocol declared 15,000, and the correlation section counted the duplicate rung `ladder_knn` (= `knn_k5`) twice | r(absorption, memorisation) floor **0.973** (now 0.972); r vs aggregate **0.680** at 5% (now 0.667); all interval widths |
| `rerun-rampart-r3c` | same two defects, INEP panel | same |
| `rerun-rampart-r3c-fix-wb-all` | resample count fixed, duplicate rung still double-counted | r floor **0.973** |
| `rerun-rampart-r3c-fix-inep` | same | same |
| `rerun-rampart-r3c-fix2-*`, `rerun-rampart-r3c-fix3-*` | every channel value identical to fix4 (verified for the ridge at both hops of the chain); each generation only adds printed receipts -- fix3 the paired contrasts and the weight w, fix4 the fold-resampled intervals on r and Lin's concordance | none -- values live on in fix4 |
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
