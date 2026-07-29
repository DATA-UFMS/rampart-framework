"""Is there a cheap stand-in for the decay slope? Pre-registered in 4.2s.

The buffer prescription costs a leakage experiment. Someone holding a panel and
wanting to know their own buffer cannot run one -- they would have to inject
contamination to learn how much contamination they tolerate. And the slope differs
between panels, 105% on the World Bank against 78% on INEP, so a published constant
is not available either. Residual autocorrelation does not stand in: it reads 0.03
and 0.00 at lag 2 on two panels whose channels differ.

The candidate measured here needs no injection. For each source period of
`probe_global_routes` -- the arms, ordered by how far they sit from the evaluation
window -- fit a model on **that period's rows alone** and score it on the evaluation
window. Same rows, same window, no new design parameter. If leaking rows from
distance d hurts in proportion to how well data at distance d predicts the window
on its own, then predictiveness-alone is the diagnostic and the buffer becomes
computable from a panel rather than from an experiment.

It also covers two arms raw distance cannot. ECHO is copies of rows already
present: distance undefined, channel zero. RESERVE sits far out but is surrounded by
years the model keeps: distance large, channel near zero. Both should land low on
predictiveness-alone for the same reason their channels are low.

Classical models only, deliberately. A diagnostic that needs a GPU is not one.

**Three defects, all found by audit rather than by the author, and all fixed here.**
They are recorded because each one moved the answer and none announced itself.

1. The arms were subsampled with `X.iloc[-budget:]`, commented as "the most recent
   rows". The World Bank parquet is entity-major -- AG 2000..2023, then AR -- so on
   that panel the tail took the last *entities alphabetically*: 16 of 32 on a 64-row
   block. The recency rule is borrowed from the context cap and does not belong
   here anyway, because an arm's identity IS its temporal position and a tail
   changes it. Subsampling is now random with a fixed seed, which holds year and
   entity composition in expectation.
2. The common row budget was computed per fold, so it varied across folds (2,672 on
   some INEP folds against 5,527 on others). It is now one budget for the panel.
3. RESERVE has zero rows on INEP folds 0 and 1 -- the reserved interior year falls
   where the panel starts -- so it was measured on six folds while every other arm
   used eight. Folds now enter only when every arm clears the floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.models.ladder import LADDER                                  # noqa: E402
from core.scientific_config import RANDOM_SEED                         # noqa: E402
from probe_global_routes import ARMS, block_of                         # noqa: E402
from probe_harness import folds, panel                                 # noqa: E402

#: The unbounded tree is out for the reason probe_global_routes gives: its clean
#: fit moves by tenths when any rows are added, so on this measurement it reports
#: instability rather than information, and its clean R^2 on the World Bank is
#: negative. Averaging it in would swamp the curve being measured.
STABLE = tuple(rung for rung in LADDER if rung.name != 'ladder_decision_tree')

#: An arm below this cannot be fitted at all, and a fold missing one arm cannot
#: contribute to a curve that compares arms.
MIN_ARM_ROWS = 8


def arms_of(window, gap, val_len):
    """The source periods and their temporal distances, as probe_global_routes builds them."""
    train_start, train_end, test_start, _test_end = window
    reserved = (train_start + train_end) // 2
    periods = {
        'GAP1': list(range(train_end + 1, train_end + 1 + gap)),
        'VAL': list(range(train_end + 1 + gap, train_end + 1 + gap + val_len)),
        'GAP2': list(range(train_end + 1 + gap + val_len,
                           train_end + 1 + 2 * gap + val_len)),
        'RESERVE': [reserved],
        'LEAK': list(range(test_start, window[3] + 1)),
    }
    distances = {arm: test_start - max(years) for arm, years in periods.items()}
    distances['LEAK'] = 0
    # ECHO is copies of training rows, so as a standalone training set it is a
    # sample of the interior. It anchors "carries nothing new about the window".
    periods['ECHO'] = [y for y in range(train_start, train_end + 1) if y != reserved]
    distances['ECHO'] = test_start - train_end
    return periods, distances


def predictiveness(dataset: str = 'worldbank'):
    """Per arm, how well its rows alone predict the evaluation window.

    One row per (fold, arm, model), with the arm's temporal distance and the R^2 its
    rows alone achieve, so the curve can be summarised without the summary choosing
    the shape.
    """
    df, columns, cfg = panel(dataset)
    windows = folds(cfg)
    gap = int(cfg.walk_forward_config['gap'])
    val_len = int(cfg.walk_forward_config['val_len'])
    print(f"{dataset}: {len(df)} rows, {len(windows)} folds, gap {gap} years")

    plans = []
    for fold, window in enumerate(windows):
        periods, distances = arms_of(window, gap, val_len)
        sizes = {arm: len(block_of(df, columns, periods[arm])[0]) for arm in ARMS}
        if min(sizes.values()) >= MIN_ARM_ROWS:
            plans.append((fold, window, periods, distances, min(sizes.values())))
    if not plans:
        print('  no fold has every arm above the floor')
        return pd.DataFrame()
    budget = min(p[4] for p in plans)
    dropped = len(windows) - len(plans)
    print(f"  {len(plans)} of {len(windows)} folds usable, {budget} rows per arm"
          + (f"  ({dropped} dropped: an arm under {MIN_ARM_ROWS} rows)" if dropped else ''))

    records = []
    for fold, window, periods, distances, _ in plans:
        test_start, test_end = window[2], window[3]
        evaluation = df[(df['year'] >= test_start) & (df['year'] <= test_end)]
        rng = np.random.default_rng(RANDOM_SEED + fold)
        for arm in ARMS:
            X, y, _entities = block_of(df, columns, periods[arm])
            if len(X) > budget:
                keep = np.sort(rng.choice(len(X), size=budget, replace=False))
                X = X.iloc[keep].reset_index(drop=True)
                y = y.iloc[keep].reset_index(drop=True)
            # The evaluation frame is filled from the arm, never from itself: the
            # arm is the training set here, so P5 puts the statistic there.
            fill = X.median()
            Xe = evaluation[columns].fillna(fill)
            valid = Xe.notna().all(axis=1) & evaluation['target'].notna()
            if valid.sum() < 3:
                continue
            Xe, ye = Xe[valid], evaluation['target'][valid]
            for rung in STABLE:
                model = rung.make()
                model.fit(X, y)
                records.append({
                    'fold': fold, 'arm': arm, 'model': rung.name,
                    'distance': distances[arm], 'rows': int(len(X)),
                    'r2': float(r2_score(ye, model.predict(Xe))),
                })
    return pd.DataFrame(records)


def curve_of(frame):
    """Arm means of distance and median R^2, ECHO excluded.

    ECHO's distance is the interior's but it carries nothing new, so it anchors
    rather than sitting on a distance curve. Median because an arm reduced to the
    common budget can put a model in the p-approaches-n regime and return an R^2 of
    several negative units, which one fold is then enough to carry the mean.
    """
    return (frame[frame['arm'] != 'ECHO']
            .groupby('arm')
            .agg(distance=('distance', 'mean'), r2=('r2', 'median'))
            .sort_values('distance'))


def slopes(curve):
    """Decay per year, raw and normalised by the curve's own LEAK value.

    Both are reported because the pre-registration fixed a 30% tolerance without
    fixing the estimator, and the verdict depends on the choice: the channel table
    it is compared against is already expressed as a fraction of LEAK, so
    normalising matches its treatment, while leaving the curve raw is what the
    instrument prints. An undeclared degree of freedom large enough to decide the
    outcome has to be shown, not resolved silently.
    """
    if len(curve) < 3:
        return {'raw': float('nan'), 'normalised': float('nan')}
    ordered = curve.sort_values('distance')
    raw = float(np.polyfit(ordered['distance'], ordered['r2'], 1)[0])
    base = float(ordered['r2'].iloc[0])          # the LEAK arm, at distance zero
    norm = (float(np.polyfit(ordered['distance'], ordered['r2'] / base, 1)[0])
            if base else float('nan'))
    return {'raw': raw, 'normalised': norm}


def report(dataset: str):
    frame = predictiveness(dataset)
    if frame.empty:
        return None
    print(f"\n{'arm':<9} {'dist':>5} {'rows':>6} {'R^2 med':>9} {'mean':>9} "
          f"{'sd':>8} {'folds':>6}")
    grouped = frame.groupby('arm')
    for arm in grouped['distance'].mean().sort_values().index:
        g = grouped.get_group(arm)
        print(f"{arm:<9} {g['distance'].mean():>5.1f} {g['rows'].mean():>6.0f} "
              f"{g['r2'].median():>9.4f} {g['r2'].mean():>9.4f} "
              f"{g['r2'].std():>8.4f} {g['fold'].nunique():>6}")
    curve = curve_of(frame)
    s = slopes(curve)
    print(f"\n  slope per year of distance: raw {s['raw']:+.5f}, "
          f"normalised by LEAK {s['normalised']:+.5f}")
    return {'dataset': dataset, 'frame': frame, 'curve': curve, 'slopes': s}


def main():
    datasets = sys.argv[1:] or ['worldbank']
    results = {}
    for dataset in datasets:
        print('=' * 78)
        out = report(dataset)
        if out is not None:
            results[dataset] = out

    if len(results) >= 2:
        names = list(results)
        print('\n' + '=' * 78)
        print('SLOPE RATIO BETWEEN PANELS -- the quantity 4.2s predicts')
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                for kind in ('raw', 'normalised'):
                    sa, sb = results[a]['slopes'][kind], results[b]['slopes'][kind]
                    ratio = sa / sb if sb else float('nan')
                    print(f"  {kind:<11} {a} {sa:+.5f}  vs  {b} {sb:+.5f}   "
                          f"ratio {ratio:.3f}")
        print('\n  Compare against the channel slope ratio recorded in 4.2o.')
        print('  The two estimators disagree, and the pre-registration did not')
        print('  fix which one, so the verdict is reported under both.')


if __name__ == '__main__':
    main()
