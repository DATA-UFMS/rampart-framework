#!/usr/bin/env python3
"""Intervals over folds that are not independent of each other.

The walk-forward design slides a window forward one year at a time, so two
consecutive folds can be scored on a year they both contain. Resampling folds
independently then treats one observation as two and reports an interval
narrower than the data supports.

The fix is a moving-block bootstrap: resample contiguous runs of folds rather
than folds, with the run long enough to carry the dependence. What makes this
worth a module rather than three lines at each call site is the block length.
It was written out as 2 in the first probe, with a comment saying consecutive
World Bank folds share a test year -- true there, and derivable rather than
observed:

    World Bank   test_len=2, step=1  ->  folds f and f+1 share one year, f and
                                        f+2 share none.        block = 2
    INEP         test_len=1, step=1  ->  no two folds share a year at all.
                                        block = 1

So the constant is not a constant. Carrying 2 to INEP would widen every INEP
interval for a dependence that panel does not have, and the direction of that
error is the comfortable one -- intervals too wide are never flagged by a
reviewer, which is exactly why it would survive.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG


def fold_dependence_span(walk_forward_config: Dict) -> int:
    """How many consecutive folds can be scored on a shared year.

    A fold's evaluation window covers `test_len` years and the next fold's
    starts `step` years later, so windows overlap while the offset is under the
    window length. One means the folds are disjoint and the ordinary bootstrap
    over folds is already correct.
    """
    test_len = int(walk_forward_config['test_len'])
    step = int(walk_forward_config['step'])
    if step <= 0:
        raise ValueError(f"step must be positive and is {step}")
    return max(1, math.ceil(test_len / step))


def moving_block_ci(
    values: Sequence[float], *, block: int,
    iters: Optional[int] = None, seed: int = RANDOM_SEED,
    ci: float = 0.95,
) -> Tuple[float, Tuple[float, float], Dict]:
    """Percentile interval for the mean, resampling contiguous runs of folds.

    Returns the point estimate, the interval, and a record of how it was
    produced -- the block length above all, because an interval whose
    dependence assumption is not stated cannot be checked.

    Percentile rather than BCa. BCa's acceleration term is estimated by
    jackknifing single observations, which assumes exchangeability; the whole
    reason for this function is that single folds are not exchangeable here.
    """
    observed = np.asarray([v for v in values if v is not None], dtype=float)
    observed = observed[np.isfinite(observed)]
    n = len(observed)
    # Clamped before the record is built, not after. Recording the requested
    # length while resampling with another is the kind of disagreement that
    # cannot be caught downstream: the artifact would name an assumption the
    # numbers were not produced under.
    block = max(1, min(int(block), n)) if n else max(1, int(block))
    record = {'n_folds': n, 'block': int(block),
              'method': 'moving_block_percentile'}

    if n == 0:
        return float('nan'), (float('nan'), float('nan')), {**record,
                                                            'method': 'insufficient_data'}
    point = float(np.mean(observed))
    if n == 1:
        return point, (float('nan'), float('nan')), {**record,
                                                     'method': 'single_fold'}

    if np.std(observed) < np.finfo(float).eps * 100:
        # Every resample has the same mean. Reported as the point estimate and
        # named, rather than passed off as an interval that happened to be tight.
        return point, (point, point), {**record, 'method': 'degenerate_zero_variance'}

    # A block that is a large share of the sample leaves few distinct starting
    # positions, so the resampled means collapse toward the sample mean and the
    # interval narrows -- narrower for acknowledging dependence, which is
    # backwards. Recorded rather than refused: with nine folds and a block of two
    # this is comfortable, and the caller who is not should be able to see it.
    if block > 1:
        record['distinct_starts'] = int(n - block + 1)
        record['block_share_of_sample'] = round(block / n, 3)
        if block / n > 0.5:
            record['warning'] = (
                'the block covers over half the sample; few distinct starts '
                'remain and the interval understates the spread')

    if iters is None:
        iters = SCIENTIFIC_CONFIG['bootstrap_iters']

    rng = np.random.default_rng(seed)
    starts = max(1, n - block + 1)
    # Enough whole blocks to cover n, then truncated: sampling blocks until the
    # count is reached and then cutting keeps every draw the same length, so the
    # resampled means are comparable across draws.
    per_draw = math.ceil(n / block)
    picks = rng.integers(0, starts, size=(iters, per_draw))
    offsets = np.arange(block)
    indices = (picks[:, :, None] + offsets[None, None, :]).reshape(iters, -1)[:, :n]
    means = observed[indices].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return point, (float(low), float(high)), {**record, 'iters': int(iters),
                                             'ci': ci}


def excludes_zero(interval: Tuple[float, float], *, direction: int = 0) -> bool:
    """Whether an interval lies entirely on one side of zero.

    `direction` is not optional in spirit. Called without it this answers "is
    there an effect", which is the wrong question whenever the hypothesis names
    a sign -- and it produced a wrong reading here: a probe asking whether the
    forest inflates more than the ridge counted an interval entirely *below*
    zero as a detection, so an effect in the opposite direction to the one the
    literature reports was reported as reproducing it.

    direction=+1 requires the interval above zero, -1 below, 0 either side.
    """
    low, high = interval
    if not (math.isfinite(low) and math.isfinite(high)):
        return False
    if direction > 0:
        return low > 0.0
    if direction < 0:
        return high < 0.0
    return low > 0.0 or high < 0.0
