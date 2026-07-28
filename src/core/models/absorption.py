#!/usr/bin/env python3
"""How much of a handed answer does a model keep?

Append a few evaluation rows, with their true labels, to the frame a model is
fitted on. Refit. Measure how much of the squared error on exactly those rows
disappears.

    absorption = 1 - sum_probe (y - after)^2 / sum_probe (y - before)^2

Zero means the extra rows changed nothing about themselves; regularisation
absorbed them. One means the model reproduces them exactly, so a contaminated row
is recovered whole.

**Sums, not a mean of ratios, and this is the third time.** The first version
averaged `(after - before) / (truth - before)` over single probe rows. That is a
mean of ratios with a residual in the denominator, and it exploded exactly where
it should have: a decision tree with eight-row leaves read 6.87, a three-nearest-
neighbour model read -0.23 with a standard deviation of 2.13. Both are nonsense
for a quantity bounded by construction. The same defect had already been caught
twice in this study -- once in the ratio form of the amplification estimand, once
in the per-model `d_z` -- and it was reintroduced here. A ratio of sums has no
small denominator to divide by unless the model is already perfect on every probe
row, which is reported rather than divided.

**One refit, not one per probe.** The probe rows go in together, which is also how
the violation being studied happens, and it costs a single extra fit. For an
in-context model a refit is a forward pass over the whole context, so the
difference is between one pass and several.

**What it is, mechanically.** It is the memorisation channel, isolated. Under
class III the evaluation window splits in two: rows the model was handed and rows
it was not. Improvement on the handed rows is memorisation and nothing else, and
that is what this measures on a small number of rows instead of on a dose.
Measured on the same panel, the two agree.

    ridge              0.20   the fit shifts a little toward the handed rows
    k-nearest (k=5)    0.36   the duplicate is one of five neighbours
    random forest      0.37   bootstrap averaging dilutes it across 200 trees
    gradient boosting  0.99   sequential residual fitting drives it to zero
    decision tree      1.00   an unbounded leaf holds the row alone

**It is calibrated against a closed form.** For k-nearest neighbours the
duplicate sits at distance zero from its own query, so it is always among the k
neighbours and pulls the prediction about 1/k of the way to the truth. The
squared error should therefore fall by (2k - 1) / k^2. Measured across k in
{1, 2, 3, 5, 10, 20} it matches to within 0.06, and at k=1 exactly. An instrument
that agrees with an analytic prediction it was not fitted to is an instrument.

**Why this replaced a borrowed ordering.** The capacity ladder was ranked by the
class III severities Roth reports over 2,047 classification datasets. That rank
does not predict inflation on this panel -- correlations of -0.72, -0.55 and
+0.04 across three doses. Absorption is measured inside the same setup as the
inflation it explains, so no analogy between a classifier's AUC and a regressor's
R^2 is required.

**It is not pure memorisation, and the impurity is measurable.** The probe rows
are appended as a batch, so the reading is taken at whatever dose that batch
represents -- twelve rows into a 64-row window is about 19%. At that dose a model
that cannot memorise at all still moves: a ridge with a penalty of 1e6 has
effectively zero coefficients and predicts its intercept, and twelve extra labels
shift that intercept. The improvement on the probe rows therefore contains a
global component along with the memorisation.

That component has an empirical floor rather than an argument: read the same way,
a heavily penalised ridge measures nothing but the global shift, because it has no
capacity for anything else. Reported alongside, so the reader can subtract it by
eye. `statistical_validation.leakage_channels` separates the two properly, on the
partition the dose defines, and also reports `local_excess` -- improvement on
handed rows above improvement on held-out ones.

**The entity effect is held fixed.** The caller passes frames that already carry
it. Recomputing it with the probe rows included would let the answer travel
through a feature rather than through the model, and this is about the model.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG

#: Below this the model already predicts every probe row to numerical precision,
#: so there is no error left to remove and the quantity is undefined. Reported as
#: such, never divided.
_ERROR_FLOOR = 1e-12


def absorption_coefficient(
    make_estimator: Callable[[], object],
    X_fit: pd.DataFrame, y_fit: pd.Series,
    X_eval: pd.DataFrame, y_eval: pd.Series,
    *, probes: Optional[int] = None, seed: int = RANDOM_SEED,
    baseline: Optional[Sequence[float]] = None,
) -> Dict:
    """Absorption over a small set of probe rows, as a ratio of sums.

    `baseline` accepts predictions already computed on the clean fit, so a caller
    that has them does not pay for the fit twice. A vector of the wrong length
    would silently redefine the quantity, so its length is checked.
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

    truth = np.asarray(y_eval.iloc[picked], dtype=float)
    before = before_all[picked]
    error_before = float(np.sum((truth - before) ** 2))

    record = {'probes_used': int(count), 'seed': int(seed),
              'error_before': error_before,
              # The count relative to the window, because absorption read at a
              # 19% dose is not absorption read at the single-row margin and the
              # number alone does not say which this is.
              'probe_dose': float(count) / len(X_eval) if len(X_eval) else float('nan')}

    if error_before < _ERROR_FLOOR:
        return {**record, 'absorption': float('nan'),
                'note': 'the model already predicts every probe row exactly, '
                        'so there is no error for the handed answers to remove'}

    widened = make_estimator()
    widened.fit(
        pd.concat([X_fit, X_eval.iloc[picked]], ignore_index=True),
        pd.concat([pd.Series(y_fit), pd.Series(y_eval).iloc[picked]],
                  ignore_index=True))
    after = np.asarray(widened.predict(X_eval.iloc[picked]), dtype=float)
    error_after = float(np.sum((truth - after) ** 2))

    return {**record, 'absorption': 1.0 - error_after / error_before,
            'error_after': error_after}


def knn_expected_absorption(k: int) -> float:
    """What absorption a k-nearest-neighbour regressor should read, analytically.

    A duplicate of the query row sits at distance zero, so it is always among the
    k neighbours and contributes 1/k of the averaged prediction. The prediction
    moves about 1/k of the way to the truth, so the squared error falls by
    1 - (1 - 1/k)^2.

    Approximate rather than exact: it assumes the neighbour displaced by the
    duplicate carried the mean of the k, which holds when the neighbours are
    similar and not otherwise. Exact at k=1, where the duplicate is the only
    neighbour and the prediction is the truth.
    """
    if k < 1:
        raise ValueError(f"k must be at least one and is {k}")
    return (2.0 * k - 1.0) / (k * k)
