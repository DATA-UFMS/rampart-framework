#!/usr/bin/env python3
"""Does the kNN calibration residual decompose the way Section 5.4 diagnoses?

The closed form alpha_k = (2k-1)/k^2 is derived for a SINGLE appended copy; the
instrument appends twelve probes as a batch, so copies of other probes can enter
a query's neighbour set and carry a term the derivation does not model. The paper
names two mechanisms -- the exchangeable-neighbourhood limit on the second panel,
the batch term on the first -- and flags both as diagnosis, "which we have not
run". This probe runs it: the same probe rows, the same seeds, appended once as a
batch and once singly, so

    batch  - single = the batch term, measured
    single - closed = the panel term, measured

and the single-probe reading also yields R_k exactly, via
alpha_single = 1 - ((k-1)/k)^2 R_k.

REGISTERED PREDICTIONS, written before the first run:
  P1  k = 1 reads exactly 1.0000 in both modes on both panels (anchor).
  P2  On the World Bank the batch term is positive at k >= 3 and the single
      reading sits nearer the closed form than the batch reading does.
  P3  On INEP the single-probe deficits at k = 2, 3, 5 are of the size the
      exchangeable limit predicts: -(k-1)/(k^2 (k+1)) = -0.083, -0.056, -0.027.

The batch column must reproduce tab_calibration's measured column digit for
digit -- same panel path, same seeds, same pooling -- or this log and the
channels log are not about the same instrument.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from probe_harness import audit_resamples, declare_provenance, folds, panel, prepared

from core.models.absorption import (  # noqa: E402
    absorption_coefficient, knn_expected_absorption)
from core.models.icl import matched_context  # noqa: E402
from core.models.ladder import entity_effect_frames  # noqa: E402
from core.scientific_config import RANDOM_SEED  # noqa: E402
from statistical_validation.dependent_bootstrap import (  # noqa: E402
    fold_dependence_span, moving_block_ci)

KNN_K = (1, 2, 3, 5, 10, 20)


def knn(k):
    from sklearn.neighbors import KNeighborsRegressor
    return lambda: KNeighborsRegressor(n_neighbors=k)


def main(dataset='worldbank'):
    df, columns, cfg = panel(dataset)
    windows = folds(cfg)
    block = fold_dependence_span(cfg.walk_forward_config)
    print(f"{dataset}: {len(windows)} folds, kNN sweep k={list(KNN_K)}, "
          f"batch and single-probe absorption on shared draws")
    declare_provenance(fold_block=block)
    print()

    batch = {k: [] for k in KNN_K}
    single = {k: [] for k in KNN_K}

    for fold, (a, b, test_start, test_end) in enumerate(windows):
        made = prepared(df, columns, a, b, test_start, test_end)
        if made is None:
            continue
        (X_train, y_train, e_train, yr_train,
         X_test, y_test, e_test, yr_test) = made
        fit_frame, eval_frame, _m, _g = entity_effect_frames(
            X_train, X_test, y_train, e_train, e_test)
        for k in KNN_K:
            make = matched_context(knn(k))
            clean = make()
            clean.fit(fit_frame, y_train)
            baseline = np.asarray(clean.predict(eval_frame), dtype=float)
            # Identical arguments except `batch`, and identical seeds, so both
            # modes draw the same probe rows and differ only in how they append.
            common = dict(probes=None, seed=RANDOM_SEED + fold,
                          baseline=baseline)
            batch[k].append(absorption_coefficient(
                make, fit_frame, y_train, eval_frame, y_test,
                **common)['absorption'])
            single[k].append(absorption_coefficient(
                make, fit_frame, y_train, eval_frame, y_test,
                batch=False, **common)['absorption'])
        print(f"  fold {fold} done", flush=True)

    print(f"\n{'k':>4} {'closed':>8} {'batch':>20} {'single':>20} "
          f"{'batch-single':>13} {'single-closed':>14} {'R_k':>7} {'R_limit':>8}")
    for k in KNN_K:
        closed = knn_expected_absorption(k)
        pb, (blo, bhi), _ = moving_block_ci(batch[k], block=block)
        ps, (slo, shi), _ = moving_block_ci(single[k], block=block)
        # alpha_single = 1 - ((k-1)/k)^2 R_k  =>  R_k from the single reading.
        r_k = ((1 - ps) * k * k / ((k - 1) ** 2)) if k > 1 else float('nan')
        r_lim = k * k / (k * k - 1) if k > 1 else float('nan')
        print(f"{k:>4} {closed:>8.4f} "
              f"{pb:>7.4f}[{blo:+.3f},{bhi:+.3f}] "
              f"{ps:>7.4f}[{slo:+.3f},{shi:+.3f}] "
              f"{pb - ps:>+13.4f} {ps - closed:>+14.4f} "
              f"{r_k:>7.3f} {r_lim:>8.3f}")

    print("\n  batch-single is the batch term; single-closed is the panel term.")
    print("  R_k is measured from the single reading via the exact identity;")
    print("  R_limit is the exchangeable-neighbourhood prediction k^2/(k^2-1).")
    audit_resamples()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
