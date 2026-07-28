#!/usr/bin/env python3
"""Which primary criterion detects a known effect on the data we actually have?

The pre-specification leaves one thing open (section 4.2): whether the primary
comparison stays the paired step between two models -- amplification under the
in-context model minus amplification under the random forest -- or becomes the
trend across the whole capacity ladder.

The corrected feasibility probe found the paired step excluding zero at only one
of three doses with nine folds. That is a power finding, and power is not
something to settle by argument when both criteria can be computed on the same
folds and compared.

**What stands in for the effect we cannot yet measure.** No in-context model is
involved here, deliberately: this asks whether each criterion can recover the
effect the literature already documents. Roth reports class III severity rising
with model capacity, so the two criteria are pointed at the same known truth:

    paired step   does the random forest inflate more than the ridge?
    ladder trend  does inflation rise across the five rungs?

Both are directional: the literature says more capacity means more inflation, so
only a positive interval counts. An interval entirely below zero is not a weaker
detection, it is the opposite finding, and the first version of this script
counted it as a detection.

Both are computed per fold, both are averaged over folds, and both get the same
moving-block interval. So the comparison is between the criteria and not between
two ways of doing inference.

**What this cannot tell us.** That one criterion detects a capacity effect more
reliably does not establish that it detects an in-context effect more reliably.
The ladder trend also answers a different question -- whether a model sits off
the capacity relation, rather than whether it beats one comparator -- and which
question the paper should ask is a judgement the numbers below inform but do not
make.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

# The harness first: importing it puts src/ on the path, so the core imports
# below resolve. Ordered on purpose, not alphabetically.
from probe_harness import DOSES, contaminate, fold_rng, folds, panel, prepared

from core.models.ladder import LADDER, fit_rung  # noqa: E402
from statistical_validation.dependent_bootstrap import (  # noqa: E402
    excludes_zero, fold_dependence_span, moving_block_ci)

#: The two rungs the paired criterion compares. Named rather than positional so
#: that inserting a rung cannot silently change which pair is tested.
PAIR = ('ladder_ridge', 'ladder_random_forest')


def main():
    df, cols, cfg = panel()
    windows = folds(cfg)
    block = fold_dependence_span(cfg.walk_forward_config)
    print(f"World Bank: {len(df)} rows, {len(cols)} features, "
          f"{len(windows)} folds")
    print(f"block length derived from the fold configuration: {block} "
          f"(test_len={cfg.walk_forward_config['test_len']}, "
          f"step={cfg.walk_forward_config['step']})\n")

    rows = []
    for fold, (a, b, test_start, test_end) in enumerate(windows):
        made = prepared(df, cols, a, b, test_start, test_end)
        if made is None:
            continue
        (X_train, y_train, e_train, yr_train,
         X_test, y_test, e_test, yr_test) = made

        clean = {rung.name: fit_rung(X_train, y_train, X_test, y_test,
                                     e_train, e_test, rung=rung,
                                     architecture='probe')['r2']
                 for rung in LADDER}

        for dose in DOSES:
            # Seeded on the fold and dose so a rerun of one cell reproduces.
            rng = fold_rng(fold, dose)
            Xc, yc, ec, yrc = contaminate(
                X_train, y_train, e_train, yr_train,
                X_test, y_test, e_test, yr_test, dose=dose, rng=rng)
            for rank, rung in enumerate(LADDER, start=1):
                leaked = fit_rung(Xc, yc, X_test, y_test, ec, e_test,
                                  rung=rung, architecture='probe')['r2']
                rows.append({'fold': fold, 'dose': dose, 'rung': rung.name,
                             'rank': rank, 'clean': clean[rung.name],
                             'leaked': leaked,
                             'inflation': leaked - clean[rung.name]})

    out = pd.DataFrame(rows)
    print("mean inflation by rung and dose")
    print(f"{'rung':>26} {'rank':>5} " +
          ' '.join(f"{d:>9.0%}" for d in DOSES))
    for rung in LADDER:
        cells = []
        for dose in DOSES:
            sub = out[(out['rung'] == rung.name) & (out['dose'] == dose)]
            cells.append(f"{sub['inflation'].mean():>+9.4f}")
        print(f"{rung.name:>26} {LADDER.index(rung) + 1:>5} " + ' '.join(cells))

    print("\n" + "=" * 72)
    print("CRITERION A -- the paired step, random forest minus ridge")
    print(f"{'dose':>6} {'mean':>10} {'CI95':>26} {'detects':>9}")
    detected_a = 0
    for dose in DOSES:
        wide = (out[out['dose'] == dose]
                .pivot(index='fold', columns='rung', values='inflation'))
        if not set(PAIR) <= set(wide.columns):
            continue
        step = (wide[PAIR[1]] - wide[PAIR[0]]).dropna().to_numpy()
        point, interval, _record = moving_block_ci(step, block=block)
        found = excludes_zero(interval, direction=+1)
        detected_a += found
        print(f"{dose:>6.2f} {point:>+10.4f} "
              f"{f'[{interval[0]:+.4f}, {interval[1]:+.4f}]':>26} "
              f"{'yes' if found else 'no':>9}")

    print("\nCRITERION B -- the trend across the ladder, slope per fold")
    print(f"{'dose':>6} {'mean':>10} {'CI95':>26} {'detects':>9}")
    detected_b = 0
    for dose in DOSES:
        slopes = []
        for fold, group in out[out['dose'] == dose].groupby('fold'):
            group = group.dropna(subset=['inflation'])
            if len(group) < 3:
                continue
            slopes.append(float(np.polyfit(group['rank'], group['inflation'], 1)[0]))
        point, interval, _record = moving_block_ci(slopes, block=block)
        found = excludes_zero(interval, direction=+1)
        detected_b += found
        print(f"{dose:>6.2f} {point:>+10.4f} "
              f"{f'[{interval[0]:+.4f}, {interval[1]:+.4f}]':>26} "
              f"{'yes' if found else 'no':>9}")

    print("\n" + "=" * 72)
    print(f"  paired step detects the known effect at {detected_a}/{len(DOSES)} doses")
    print(f"  ladder trend detects the known effect at {detected_b}/{len(DOSES)} doses")
    if detected_b > detected_a:
        print("\n  The trend recovers the capacity effect where the two-point step")
        print("  does not, on identical folds and identical inference.")
    elif detected_b < detected_a:
        print("\n  The step recovers it where the trend does not. The extra rungs")
        print("  add noise rather than signal, and the pre-registered criterion")
        print("  should stay as it is.")
    else:
        print("\n  Neither dominates on this panel. The choice does not turn on")
        print("  power and has to be made on what the paper wants to claim.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
