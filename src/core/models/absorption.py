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

**It is checked against a closed form, and the check has structure.** For k-nearest neighbours the
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
    replicates: Optional[int] = None,
    fraction: Optional[float] = None,
    batch: bool = True,
) -> Dict:
    """Absorption over a small set of probe rows, as a ratio of sums.

    `baseline` accepts predictions already computed on the clean fit, so a caller
    that has them does not pay for the fit twice. A vector of the wrong length
    would silently redefine the quantity, so its length is checked.

    **Why there are replicates.** Twelve probe rows are drawn by position from the
    evaluation frame, so the reading depends on which twelve. It depends on it
    enough to move the headline number fivefold: when `prepared` began sorting the
    evaluation frame by year -- a change made so the context cap could live in the
    estimator -- the same panel, the same count and the same seed went from a
    largest closed-form gap of 0.0108 to 0.0577. Nothing about the panel or the
    estimator changed; only which rows the generator landed on. Averaging over
    replicates converges to about 0.043 and both of those are draws around it.

    A replicate is a fresh probe set at the *same* count, so it cuts that variance
    without touching the dose -- more probes would reduce the variance too, but it
    would also stop being a reading at the twelve-row margin. The pooling is a
    ratio of sums over all replicates, never a mean of per-replicate ratios: a
    residual in the denominator has produced a divergent mean three times in this
    repository already.
    """
    # `fraction` sets the probe count as a share of the frame being perturbed, and it
    # exists because a fixed COUNT is not comparable across panels. Absorption appends
    # a fixed twelve rows, so the perturbation as a share of training is 12/n: 3.13%
    # on the World Bank at n≈384 and 0.12% on the larger panel, whose models read
    # 10,000 rows under the context cap -- twenty-six times apart. A quantity that
    # falls with n then measures the perturbation rather than the model, and it did.
    # Matched at 3.13% on both panels, the ridge goes from n^-0.75 to n^-0.20 and the
    # random forest from n^-0.43 to n^-0.15, while 1-NN and the unbounded tree stay at
    # exactly 1.0000, which is what structural invariance looks like. Any cross-panel
    # reading has to match the fraction, never the count -- and both ends of the
    # comparison have to share a cap setting, or the row charges a configuration
    # difference to n.
    if fraction is not None:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
        probes = max(1, int(round(float(fraction) * len(X_fit))))
    if probes is None:
        probes = SCIENTIFIC_CONFIG['in_context_models']['absorption_probes']
    if replicates is None:
        replicates = SCIENTIFIC_CONFIG['in_context_models'].get(
            'absorption_replicates', 1)
    replicates = max(1, int(replicates))

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

    # One generator for every replicate, so a run with R replicates contains the
    # R=1 draw as its first and the reading is nested rather than incomparable.
    rng = np.random.default_rng(seed)
    count = min(int(probes), len(X_eval))

    record = {'probes_used': int(count), 'seed': int(seed),
              'batch': bool(batch),
              'replicates': int(replicates),
              # Recorded because it is the only quantity that makes two panels
              # comparable, and because the count alone hides it.
              'perturbation_share': (float(count) / len(X_fit)
                                     if len(X_fit) else float('nan')),
              'fraction_requested': (float(fraction) if fraction is not None
                                     else None),
              # The count relative to the window, because absorption read at a
              # 19% dose is not absorption read at the single-row margin and the
              # number alone does not say which this is. Unchanged by replicates:
              # each one appends the same count, which is the point of them.
              'probe_dose': float(count) / len(X_eval) if len(X_eval) else float('nan')}

    total_before = total_after = 0.0
    # `batch=False` appends the probes one at a time -- one refit per probe, the
    # numerators and denominators still pooled as one ratio of sums. It exists
    # because the kNN closed form is derived for a SINGLE appended copy while the
    # instrument appends twelve at once, so copies of other probes can enter a
    # query's neighbour set and carry a term the derivation does not model. The
    # single-probe reading is the derivation's regime; batch minus single is the
    # batch term, measured rather than argued. The draw is shared: both modes
    # consume the generator identically, so they read the same probe rows and
    # their difference is the appending mode and nothing else.
    for _ in range(replicates):
        picked = np.sort(rng.choice(len(X_eval), size=count, replace=False))
        truth = np.asarray(y_eval.iloc[picked], dtype=float)
        total_before += float(np.sum((truth - before_all[picked]) ** 2))

        if batch:
            widened = make_estimator()
            widened.fit(
                pd.concat([X_fit, X_eval.iloc[picked]], ignore_index=True),
                pd.concat([pd.Series(y_fit), pd.Series(y_eval).iloc[picked]],
                          ignore_index=True))
            after = np.asarray(widened.predict(X_eval.iloc[picked]), dtype=float)
            total_after += float(np.sum((truth - after) ** 2))
        else:
            for j, row in enumerate(picked):
                one = picked[j:j + 1]
                widened = make_estimator()
                widened.fit(
                    pd.concat([X_fit, X_eval.iloc[one]], ignore_index=True),
                    pd.concat([pd.Series(y_fit), pd.Series(y_eval).iloc[one]],
                              ignore_index=True))
                after = np.asarray(widened.predict(X_eval.iloc[one]),
                                   dtype=float)
                total_after += float((truth[j] - after[0]) ** 2)

    record['error_before'] = total_before
    if total_before < _ERROR_FLOOR:
        return {**record, 'absorption': float('nan'),
                'note': 'the model already predicts every probe row exactly, '
                        'so there is no error for the handed answers to remove'}

    return {**record, 'absorption': 1.0 - total_after / total_before,
            'error_after': total_after}


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
