#!/usr/bin/env python3
"""Aggregate severity hides two effects that order models oppositely.

Under class III a fraction of the evaluation rows sit in the training frame. The
usual severity measure -- how much better the model scores once contaminated --
sums over the whole evaluation window and so adds together two different things:

    local     improvement on rows the model was handed. Memorisation, and
              nothing else can produce it.
    global    improvement on rows it was not handed. A change in generalisation:
              the contaminated rows shifted the fit, and the shift reached rows
              that never leaked.

They are not variations of one quantity. Measured on the World Bank panel, the
random forest beats the ridge on the local channel at every dose (0.37 against
0.20 at 5%) while the ridge beats the forest on the global one (0.062 against
0.013). The aggregate is dominated by the global channel, because the rows that
did not leak outnumber those that did by nineteen to one at a 5% dose -- so the
aggregate reports the ridge as the more severely affected model, and the capacity
ordering the literature documents appears to fail. It does not fail. It lives in
the local channel, and the aggregate buries it.

Mechanically the split is unsurprising once stated: a linear fit moves all its
coefficients to accommodate an extra row, and the movement applies to every
prediction; a deep tree changes the one leaf the row falls into. What is worth
saying is that the standard measure adds them, and that their ratio depends on
dose -- so a severity ranking computed this way can reorder models when the
duplication rate changes.

**The global channel needs a control, and this is the part that is easy to skip.**
The rows that did not leak also benefit from the training frame simply being
larger. That is not leakage; it is more data. Separating them requires an arm
where the frame grows by the same number of rows with nothing crossing the split
-- rows duplicated from inside the training window. See `InjectionSpec.control`.

    global = (held-out improvement, leak arm) - (held-out improvement, control arm)

**Two channels, and that is not all of them.** This is complete for leakage that
hands over *data* and incomplete for leakage that hands over a *decision*.
Selection leakage -- peeking at the evaluation set to choose a configuration --
moves no rows, so the memorisation channel is zero by construction and everything
lands in the generalisation bucket, where it does not belong: nothing generalised,
a winner was picked on noise. It needs a third arm rather than a finer partition,
and `scripts/validation/probe_selection.py` supplies one, by choosing on one half
of the evaluation window and reporting on the other. Measured that way, the
genuine share is 1.000 for a twenty-candidate ridge grid and 0.554 for a
nine-candidate forest grid -- which is also an independent confirmation that the
bias does not come from the number of candidates.

**Ratios of sums throughout.** Every quantity here is one total squared error
divided by another. A mean of per-row ratios has a residual in each denominator,
and that construction has already produced a reading of 6.87 for a bounded
quantity once in this study.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

#: Below this there is no error to remove and a ratio would divide by noise.
_ERROR_FLOOR = 1e-12

#: Fewer handed rows than this and the local channel is an average over a handful
#: of observations. Not refused -- reported, because at a 5% dose on a 64-row
#: window three rows is what the design actually produces and hiding that would
#: be worse than showing it.
_THIN_PARTITION = 5


def _sum_squared(truth: np.ndarray, predicted: np.ndarray,
                 mask: np.ndarray) -> float:
    if not mask.any():
        return float('nan')
    return float(np.sum((truth[mask] - predicted[mask]) ** 2))


def _reduction(before: float, after: float) -> float:
    """Fraction of squared error removed, or nan when there was none to remove."""
    if not np.isfinite(before) or before < _ERROR_FLOOR:
        return float('nan')
    return 1.0 - after / before


def handed_mask(entities: Sequence, years: Sequence,
                keys_moved: Iterable[Sequence]) -> np.ndarray:
    """Which evaluation rows the leak arm put into the training frame.

    Keyed on (entity, year), which is the row identity the disjointness gate
    already enforces as unique within a split. Both arms record the same key
    list, so the partition is identical across arms by construction rather than
    by coincidence.
    """
    moved = {(str(entity), int(year)) for entity, year in keys_moved}
    return np.array([(str(entity), int(year)) in moved
                     for entity, year in zip(entities, years)], dtype=bool)


def decompose_fold(
    truth: Sequence[float],
    clean: Sequence[float],
    leaked: Sequence[float],
    *, mask: np.ndarray,
    control: Optional[Sequence[float]] = None,
) -> Dict:
    """Split one fold's inflation into its local and global parts.

    `control` is the same fold under the sample-size control arm. Without it the
    global channel is reported as `global_uncontrolled` and the attributable part
    is left as None, because calling an uncontrolled number `global` would let a
    sample-size effect be read as leakage.
    """
    truth = np.asarray(truth, dtype=float)
    clean = np.asarray(clean, dtype=float)
    leaked = np.asarray(leaked, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not (len(truth) == len(clean) == len(leaked) == len(mask)):
        raise ValueError(
            f"lengths disagree: truth {len(truth)}, clean {len(clean)}, "
            f"leaked {len(leaked)}, mask {len(mask)}. These are per-row vectors "
            f"over one evaluation window and a mismatch means they are not.")

    held = ~mask
    clean_handed = _sum_squared(truth, clean, mask)
    clean_held = _sum_squared(truth, clean, held)
    leak_handed = _sum_squared(truth, leaked, mask)
    leak_held = _sum_squared(truth, leaked, held)

    local = _reduction(clean_handed, leak_handed)
    global_uncontrolled = _reduction(clean_held, leak_held)

    # The exact identity that ties the parts to the whole: aggregate reduction is
    # the two channels weighted by their share of the clean error. Returned rather
    # than asserted, so a caller can check the arithmetic of its own inputs.
    total_clean = np.nansum([clean_handed, clean_held])
    aggregate = _reduction(
        total_clean, float(np.nansum([leak_handed, leak_held])))
    weight = (clean_handed / total_clean
              if np.isfinite(clean_handed) and total_clean >= _ERROR_FLOOR
              else float('nan'))

    result = {
        'rows_handed': int(mask.sum()),
        'rows_held': int(held.sum()),
        'local': local,
        'global_uncontrolled': global_uncontrolled,
        'aggregate': aggregate,
        'local_weight': weight,
        'thin_handed_partition': bool(mask.sum() < _THIN_PARTITION),
        # Improvement on handed rows above improvement on held-out ones.
        #
        # `local` is not pure memorisation: a contaminated row shifts the fit, and
        # the shift reaches the handed rows too. Measured -- a ridge penalised to
        # a thousandth of its coefficients, which cannot memorise anything, still
        # improves on rows it was handed, because its intercept moved. Subtracting
        # the held-out improvement removes a shift that is homogeneous across
        # rows, which is an assumption and not a fact: a shift concentrated near
        # the handed rows would survive it. Reported for that reason rather than
        # substituted for `local`.
        'local_excess': (local - global_uncontrolled
                         if np.isfinite(local) and np.isfinite(global_uncontrolled)
                         else float('nan')),
    }

    if control is None:
        result['sample_size_effect'] = None
        result['global'] = None
        return result

    control = np.asarray(control, dtype=float)
    if len(control) != len(truth):
        raise ValueError(
            f"the control arm has {len(control)} predictions against "
            f"{len(truth)} evaluation rows")
    size_effect = _reduction(clean_held, _sum_squared(truth, control, held))
    result['sample_size_effect'] = size_effect
    result['global'] = (global_uncontrolled - size_effect
                        if np.isfinite(global_uncontrolled)
                        and np.isfinite(size_effect) else float('nan'))
    return result


def check_identity(fold: Dict, tolerance: float = 1e-9) -> bool:
    """Whether the aggregate equals the channels weighted by their error shares.

    An arithmetic self-check on a decomposition, which is the kind of claim that
    should not rest on the reader trusting the implementation. Uses the
    uncontrolled global, since the identity is about how the evaluation window was
    partitioned and not about attribution.
    """
    weight = fold.get('local_weight')
    parts = (fold.get('local'), fold.get('global_uncontrolled'),
             fold.get('aggregate'))
    if weight is None or not all(np.isfinite(v) for v in (weight, *parts)):
        return False
    local, global_uncontrolled, aggregate = parts
    rebuilt = weight * local + (1.0 - weight) * global_uncontrolled
    return abs(rebuilt - aggregate) < tolerance


def summarise(folds: Sequence[Dict], *, block: int,
              iters: Optional[int] = None) -> Dict[str, Dict]:
    """Per-channel point estimates and intervals over folds.

    Folds are the resampling unit and they overlap, so the interval comes from
    the moving-block bootstrap with the block derived from the fold
    configuration; see `statistical_validation.dependent_bootstrap`.

    The resample count is resolved here rather than left to the callee's own
    default. Passing None down would make the number this function used depend on
    a default declared somewhere else, which is a second source of truth for a
    value the protocol owns.
    """
    from core.scientific_config import SCIENTIFIC_CONFIG
    from statistical_validation.dependent_bootstrap import moving_block_ci

    if iters is None:
        iters = SCIENTIFIC_CONFIG['bootstrap_iters']

    out: Dict[str, Dict] = {}
    for channel in ('local', 'local_excess', 'global', 'global_uncontrolled',
                    'sample_size_effect', 'aggregate'):
        values = [fold.get(channel) for fold in folds]
        values = [v for v in values if v is not None and np.isfinite(v)]
        point, interval, record = moving_block_ci(values, block=block,
                                                  iters=iters)
        out[channel] = {'point': point, 'ci95': interval, 'inference': record}

    out['coverage'] = {
        'folds': len(folds),
        'folds_with_thin_handed_partition':
            sum(1 for fold in folds if fold.get('thin_handed_partition')),
        'identity_holds': all(check_identity(fold) for fold in folds),
    }
    return out
