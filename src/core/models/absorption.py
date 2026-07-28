#!/usr/bin/env python3
"""How much of a handed answer does a model keep?

Append one evaluation row, with its true label, to the frame a model is fitted
on. Refit. See how far that row's own prediction moves toward its truth, as a
fraction of the error it started with. A model that shrinks the extra row away
keeps little; a model that reproduces it keeps all of it.

    absorption = (prediction_after - prediction_before) / (truth - prediction_before)

Near zero means regularisation absorbed the row. Near one means the model
reproduces a training row exactly, so a contaminated row is recovered whole.

**Why this exists, and what it replaced.** The capacity ladder was ordered by the
class III severities Roth reports -- naive Bayes 0.37 through decision tree 1.11.
Measured on our panel, that order does not predict inflation: the correlation
between rung position and inflation is -0.72, -0.55 and +0.04 across the three
doses. The borrowed order is not weak, it is inverted at low dose. Gradient
boosting inflates about five times more than the random forest ranked above it.

Measured absorption says why, and the numbers are legible:

    ridge              0.11   regularisation shrinks the row away
    random forest      0.30   bootstrap averaging over 200 trees dilutes it
    k-nearest          0.24   the duplicate is one of k neighbours
    gradient boosting  1.00   sequential residual fitting drives it to zero
    decision tree      1.00   an unbounded leaf holds the row alone

Roth's ordering was measured on classifiers by AUC over 2,047 datasets; ours is
regression by R^2 over folds of one panel. He warns against carrying his numbers
across, and this is what carrying them across looks like. Absorption is measured
inside the same setup that the inflation is measured in, so no analogy is needed.

**It also measures the mechanism the study argues for.** The claim is that in an
in-context learner the context *is* the fitted model, so a contaminated row is
not shrunk by any regularisation term because there is no fitting to regularise.
That is a statement about absorption, and absorption is computable for an
in-context model the same way it is for a ridge: put the row in the context and
see whether the answer comes back. So the mechanism becomes a measurement that
stands independently of the inflation experiment.

**The entity effect is held fixed.** The caller passes frames that already carry
it. Recomputing it with the probe row included would let the answer travel
through a feature rather than through the model, and this function is about the
model.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG

#: Below this, the row was already predicted so well that the ratio is dividing
#: by noise. Such probes are skipped and counted, never silently averaged in.
_RESIDUAL_FLOOR = 1e-9


def absorption_coefficient(
    make_estimator: Callable[[], object],
    X_fit: pd.DataFrame, y_fit: pd.Series,
    X_eval: pd.DataFrame, y_eval: pd.Series,
    *, probes: Optional[int] = None, seed: int = RANDOM_SEED,
    baseline: Optional[Sequence[float]] = None,
) -> Dict:
    """Absorption over several single-row probes, with its spread.

    Several rather than one: a single probe row makes the coefficient depend on
    which row was drawn, and the design resamples folds, so a per-fold quantity
    needs to be stable within the fold rather than only across folds.

    `baseline` accepts predictions already computed on the clean fit, so a caller
    that has them does not pay for the fit twice. Passing a stale vector would
    silently redefine the quantity, so its length is checked.
    """
    if probes is None:
        probes = SCIENTIFIC_CONFIG['in_context_models']['absorption_probes']

    if baseline is None:
        clean = make_estimator()
        clean.fit(X_fit, y_fit)
        before_all = np.asarray(clean.predict(X_eval), dtype=float)
    else:
        before_all = np.asarray(baseline, dtype=float)
        if len(before_all) != len(X_eval):
            raise ValueError(
                f"baseline has {len(before_all)} predictions for "
                f"{len(X_eval)} evaluation rows; a mismatched vector would "
                f"redefine what is being measured")

    rng = np.random.default_rng(seed)
    count = min(int(probes), len(X_eval))
    picked = np.sort(rng.choice(len(X_eval), size=count, replace=False))

    values, skipped = [], 0
    for position in picked:
        before = float(before_all[position])
        truth = float(y_eval.iloc[position])
        residual = truth - before
        if abs(residual) < _RESIDUAL_FLOOR:
            skipped += 1
            continue
        widened = make_estimator()
        widened.fit(
            pd.concat([X_fit, X_eval.iloc[[position]]], ignore_index=True),
            pd.concat([pd.Series(y_fit), pd.Series(y_eval).iloc[[position]]],
                      ignore_index=True))
        after = float(widened.predict(X_eval.iloc[[position]])[0])
        values.append((after - before) / residual)

    if not values:
        return {'absorption': float('nan'), 'probes_used': 0,
                'probes_skipped': int(skipped),
                'note': 'every probe row was already fitted exactly'}

    array = np.asarray(values, dtype=float)
    return {
        'absorption': float(np.mean(array)),
        'absorption_sd': float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        'per_probe': [float(v) for v in array],
        'probes_used': int(len(array)),
        'probes_skipped': int(skipped),
        'seed': int(seed),
    }
