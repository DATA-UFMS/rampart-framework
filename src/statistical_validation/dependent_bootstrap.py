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
from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG

#: How many intervals were produced at each resample count, for the run so far.
#:
#: Every record already carries its own `iters`, and that was not enough: the
#: records go into the tables and the count does not, so a log could be read for
#: months without revealing which count produced it. Proving that a published
#: interval came from the configured count meant reading the git history of
#: `scientific_config` and dating it against the run -- and that reading is only
#: as good as the assumption that no call site overrode the default, which two
#: of them do.
#:
#: A module-level tally is deliberate. Threading a collector through every caller
#: would make the audit opt-in at each site, and the sites that forget are
#: precisely the ones worth auditing.
_RESAMPLE_LEDGER: Counter = Counter()


def observed_resample_counts() -> Dict[int, int]:
    """Resample counts actually used so far, mapped to how many intervals used them.

    Empty until an interval is produced. More than one key means the run mixed
    counts, which is reportable on its own: a paper that states a single
    resample count is then stating something false about part of its own tables.
    """
    return dict(_RESAMPLE_LEDGER)


def reset_resample_ledger() -> None:
    """Forget what has been observed. For tests, and for probes that loop panels."""
    _RESAMPLE_LEDGER.clear()


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

    # Tallied here rather than at the top: the paths above return without
    # resampling at all, and counting them would report resamples that never ran.
    _RESAMPLE_LEDGER[int(iters)] += 1

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


def moving_block_correlation_ci(
    per_fold_a, per_fold_b, *, block: int,
    iters: Optional[int] = None, seed: int = RANDOM_SEED, ci: float = 0.95,
) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    """Fold-resampled intervals for cross-model agreement statistics.

    The paper's correlations are computed over model-level points, each of which
    is a mean over folds -- so the sampling unit is the fold, not the model, and
    an interval that resamples anything else answers a different question. Both
    inputs are (models, folds) arrays; each resample draws contiguous fold blocks
    with the same scheme as `moving_block_ci`, recomputes every model's mean, and
    recomputes the statistic over models.

    Two statistics come back, because they answer different questions and the
    paper needs both. Pearson r rewards any linear relationship, including one
    with the wrong slope or offset; Lin's concordance rewards agreement with the
    identity line specifically, which is the claim an instrument makes when its
    scatter is drawn against y = x.
    """
    a = np.asarray(per_fold_a, dtype=float)
    b = np.asarray(per_fold_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(
            f"expected two (models, folds) arrays of one shape, got {a.shape} "
            f"and {b.shape}")
    models, n = a.shape
    block = max(1, min(int(block), n))
    if iters is None:
        iters = int(SCIENTIFIC_CONFIG['bootstrap_iters'])
    _RESAMPLE_LEDGER[int(iters)] += 2  # one interval per statistic

    def stats(idx):
        ma = np.nanmean(a[:, idx], axis=1)
        mb = np.nanmean(b[:, idx], axis=1)
        va, vb = ma.var(), mb.var()
        if va < 1e-18 or vb < 1e-18:
            return float('nan'), float('nan')
        cov = ((ma - ma.mean()) * (mb - mb.mean())).mean()
        r = cov / np.sqrt(va * vb)
        ccc = 2 * cov / (va + vb + (ma.mean() - mb.mean()) ** 2)
        return float(r), float(ccc)

    point_r, point_c = stats(np.arange(n))
    rng = np.random.default_rng(seed)
    starts = max(1, n - block + 1)
    per_draw = math.ceil(n / block)
    picks = rng.integers(0, starts, size=(iters, per_draw))
    offsets = np.arange(block)
    indices = (picks[:, :, None] + offsets[None, None, :]).reshape(iters, -1)[:, :n]
    draws = np.array([stats(indices[i]) for i in range(iters)])
    lo = (1.0 - ci) / 2.0
    r_lo, r_hi = np.nanpercentile(draws[:, 0], [100 * lo, 100 * (1 - lo)])
    c_lo, c_hi = np.nanpercentile(draws[:, 1], [100 * lo, 100 * (1 - lo)])
    return {'pearson': (point_r, (float(r_lo), float(r_hi))),
            'concordance': (point_c, (float(c_lo), float(c_hi)))}


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
