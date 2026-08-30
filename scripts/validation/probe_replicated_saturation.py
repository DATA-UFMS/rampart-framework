#!/usr/bin/env python3
"""Randomized-saturation audit: replicated contamination draws, per-row losses.

Contamination is interference. Inserting an evaluation row into training changes
the shared model, and therefore can change the loss of every OTHER evaluation
row. Framed as causal inference under interference (Hudgens--Halloran 2008;
Aronow--Samii 2017; randomized saturation: Baird et al. 2018), the estimands at
saturation s are

    S(s) = E[Y(0,0) - Y(0,s)]   spillover on rows NOT inserted, and
    D(s) = E[Y(0,s) - Y(1,s)]   direct effect of a row's own insertion,

with the aggregate evaluation bias B(s) = S(s) + s * D(s). Under a fixed-size
simple random draw of the inserted set and an ADDITIVE per-row loss, the
handed-row mean improvement estimates S + D, the unseen-row mean improvement
estimates S, and their difference estimates D -- exactly, not asymptotically.
Both conditions are load-bearing: Bernoulli draws add ratio bias (of order 1/n
under smoothness, with the denominator bounded away from zero), and
group-relative improvements (ratio of sums per group) break the algebra
entirely, so this probe records raw per-row losses and leaves aggregation to
the analysis.

What the existing channel probe cannot supply is inference: one draw per
(fold, dose) prices nothing. Here every (fold, model, saturation) cell is
replicated over independent draws of the inserted set, sharing the clean fit.
Training is deterministic given (data, seed), so the draw is the ONLY source of
randomness and Fisher randomization tests over re-draws are exact per fold for
the sharp null. Interval estimation for S(s) additionally uses the
saturation-zero arm, which is the clean fit itself.

Per-row records go to one parquet file per (dataset, fold) in the working
directory -- kernel stdout cannot carry ~10^7 rows -- with columns:
fold, model, saturation, rep, arm (clean|leak), row, year, entity, handed,
y_true, y_pred. Loss is (y_true - y_pred)^2, derivable, not stored.

Environment knobs (all optional):
    RAMPART_REPS         replicates per cell, default 40
    RAMPART_SATURATIONS  comma list, default 0.05,0.10,0.20,0.30
    RAMPART_FOLDS        comma list of fold indices to run (kernel sharding)

Run: .venv/bin/python scripts/validation/probe_replicated_saturation.py [dataset]
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_harness import (declare_provenance, fold_rng, folds,
                           ladder_roster, panel, prepared)

from core.models.ladder import entity_effect_frames  # noqa: E402

MODELS = ladder_roster()
REPS = int(os.environ.get('RAMPART_REPS', '40'))
SATURATIONS = tuple(
    float(x) for x in
    os.environ.get('RAMPART_SATURATIONS', '0.05,0.10,0.20,0.30').split(','))
FOLD_FILTER = (tuple(int(x) for x in os.environ['RAMPART_FOLDS'].split(','))
               if os.environ.get('RAMPART_FOLDS') else None)


def per_row_frame(fold, name, saturation, rep, arm, yr, ent, handed_mask,
                  truth, preds):
    """One record per evaluation row, typed small: the volume is the point."""
    return pd.DataFrame({
        'fold': np.int16(fold),
        'model': pd.Categorical([name] * len(truth)),
        'saturation': np.float32(saturation),
        'rep': np.int16(rep),
        'arm': pd.Categorical([arm] * len(truth)),
        'row': np.arange(len(truth), dtype=np.int32),
        'year': np.asarray(yr, dtype=np.int16),
        'entity': pd.Categorical(np.asarray(ent).astype(str)),
        'handed': np.asarray(handed_mask, dtype=bool),
        'y_true': np.asarray(truth, dtype=np.float32),
        'y_pred': np.asarray(preds, dtype=np.float32),
    })


def main(dataset='worldbank'):
    df, columns, cfg = panel(dataset)
    windows = folds(cfg)
    print(f"{dataset}: {len(windows)} folds, {len(MODELS)} models, "
          f"saturations {list(SATURATIONS)}, {REPS} replicates per cell, "
          f"fixed-size SRS draws, per-row losses to parquet")
    declare_provenance(reps=REPS, saturations=SATURATIONS,
                       fold_filter=FOLD_FILTER or 'all')
    print("  no resampled intervals in this probe: inference is design-based,"
          " offline, from the per-row records")
    print()

    for fold, (a, b, test_start, test_end) in enumerate(windows):
        if FOLD_FILTER is not None and fold not in FOLD_FILTER:
            continue
        made = prepared(df, columns, a, b, test_start, test_end)
        if made is None:
            continue
        (X_train, y_train, e_train, yr_train,
         X_test, y_test, e_test, yr_test) = made
        fit_frame, eval_frame, _m, _g = entity_effect_frames(
            X_train, X_test, y_train, e_train, e_test)
        truth = np.asarray(y_test, dtype=float)
        no_mask = np.zeros(len(eval_frame), dtype=bool)
        records = []

        clean_losses = {}
        for rung in MODELS:
            model = rung.make()
            model.fit(fit_frame, y_train)
            preds = np.asarray(model.predict(eval_frame), dtype=float)
            clean_losses[rung.name] = (truth - preds) ** 2
            records.append(per_row_frame(
                fold, rung.name, 0.0, 0, 'clean', yr_test, e_test, no_mask,
                truth, preds))

        for saturation in SATURATIONS:
            count = max(1, int(round(saturation * len(eval_frame))))
            s_hat = {rung.name: [] for rung in MODELS}
            d_hat = {rung.name: [] for rung in MODELS}
            for rep in range(REPS):
                # Numeric key parts only: hash() of numbers is stable across
                # processes, and the draw must be reproducible per cell.
                rng = fold_rng(fold, saturation, rep)
                handed = np.sort(rng.choice(len(eval_frame), size=count,
                                            replace=False))
                mask = np.zeros(len(eval_frame), dtype=bool)
                mask[handed] = True
                X_leak = pd.concat([fit_frame, eval_frame.iloc[handed]],
                                   ignore_index=True)
                y_leak = pd.concat([y_train, y_test.iloc[handed]],
                                   ignore_index=True)
                for rung in MODELS:
                    model = rung.make()
                    model.fit(X_leak, y_leak)
                    preds = np.asarray(model.predict(eval_frame), dtype=float)
                    records.append(per_row_frame(
                        fold, rung.name, saturation, rep, 'leak', yr_test,
                        e_test, mask, truth, preds))
                    # Additive per-row improvement, the scale the estimands need.
                    gain = clean_losses[rung.name] - (truth - preds) ** 2
                    s_hat[rung.name].append(float(gain[~mask].mean()))
                    d_hat[rung.name].append(
                        float(gain[mask].mean() - gain[~mask].mean()))
            for rung in MODELS:
                s, d = np.array(s_hat[rung.name]), np.array(d_hat[rung.name])
                print(f"  fold {fold} s={saturation:.2f} {rung.name:>26}: "
                      f"S_hat {s.mean():+.4f} (rep sd {s.std(ddof=1):.4f})  "
                      f"D_hat {d.mean():+.4f} (rep sd {d.std(ddof=1):.4f})",
                      flush=True)

        out = Path(f'replicated_saturation_{dataset}_fold{fold}.parquet')
        pd.concat(records, ignore_index=True).to_parquet(out, index=False)
        print(f"  fold {fold}: {sum(len(r) for r in records)} rows -> {out}",
              flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
