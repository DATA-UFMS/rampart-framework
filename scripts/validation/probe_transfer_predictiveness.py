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
present: distance undefined, channel zero. RESERVE sits 12.8 years out but is
surrounded by years the model keeps: distance large, channel near zero. Both should
land low on predictiveness-alone for the same reason their channels are low.

Classical models only, deliberately. A diagnostic that needs a GPU is not one.
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
from probe_global_routes import ARMS, block_of                         # noqa: E402
from probe_harness import folds, panel                                 # noqa: E402

#: The unbounded tree is out for the reason probe_global_routes gives: its clean
#: fit moves by tenths when any rows are added, so on this measurement it reports
#: instability rather than information, and its clean R^2 on the World Bank is
#: negative. Averaging it in would swamp the curve being measured.
STABLE = tuple(rung for rung in LADDER if rung.name != 'ladder_decision_tree')


def predictiveness(dataset: str = 'worldbank'):
    """Per arm, how well its rows alone predict the evaluation window.

    Returns a frame of one row per (fold, arm, model) with the arm's temporal
    distance and the R^2 its rows alone achieve, so the curve can be summarised
    without the summary choosing the shape.
    """
    df, columns, cfg = panel(dataset)
    windows = folds(cfg)
    gap = int(cfg.walk_forward_config['gap'])
    val_len = int(cfg.walk_forward_config['val_len'])
    print(f"{dataset}: {len(df)} rows, {len(windows)} folds, gap {gap} years")

    records = []
    for fold, (train_start, train_end, test_start, test_end) in enumerate(windows):
        reserved = (train_start + train_end) // 2
        periods = {
            'GAP1': list(range(train_end + 1, train_end + 1 + gap)),
            'VAL': list(range(train_end + 1 + gap, train_end + 1 + gap + val_len)),
            'GAP2': list(range(train_end + 1 + gap + val_len,
                               train_end + 1 + 2 * gap + val_len)),
            'RESERVE': [reserved],
            'LEAK': list(range(test_start, test_end + 1)),
        }
        distances = {arm: test_start - max(years) for arm, years in periods.items()}
        distances['LEAK'] = 0
        # ECHO is copies of training rows, so as a standalone training set it is a
        # sample of the interior. Its distance is the interior's, and it is here to
        # anchor "carries nothing new about the window".
        periods['ECHO'] = [y for y in range(train_start, train_end + 1)
                           if y != reserved]
        distances['ECHO'] = test_start - train_end

        evaluation = df[(df['year'] >= test_start) & (df['year'] <= test_end)]

        # Every arm trains on the same number of rows. As standalone training sets
        # the arms are not the same size -- RESERVE is one year, the distance arms
        # are two, ECHO is the whole interior -- so without this the curve reads
        # sample size as distance, which is the confound that has already had to be
        # resolved twice on these panels. Subsampled from the most recent rows, the
        # same rule the context cap uses, so the choice is not a new one.
        blocks = {arm: block_of(df, columns, periods[arm]) for arm in ARMS}
        usable = {arm: b for arm, b in blocks.items() if len(b[0]) >= 8}
        if not usable:
            continue
        budget = min(len(b[0]) for b in usable.values())

        for arm, (X, y, _) in usable.items():
            if len(X) > budget:
                X = X.iloc[-budget:].reset_index(drop=True)
                y = y.iloc[-budget:].reset_index(drop=True)
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


def slope(frame):
    """Decay of predictiveness per year of distance, over the distance arms.

    ECHO is excluded: its distance is the interior's but it carries nothing new, so
    it is an anchor rather than a point on a distance curve. Fitted on arm means so
    that arms with more folds do not weigh more.
    """
    curve = (frame[frame['arm'] != 'ECHO']
             .groupby('arm')
             .agg(distance=('distance', 'mean'), r2=('r2', 'median')))
    if len(curve) < 3:
        return float('nan'), curve
    fit = np.polyfit(curve['distance'], curve['r2'], 1)
    return float(fit[0]), curve.sort_values('distance')


def report(dataset: str):
    frame = predictiveness(dataset)
    if frame.empty:
        print('  no arm produced a usable fit')
        return None
    # Median alongside mean, because an arm reduced to the smallest arm's row count
    # can put a model in the p-approaches-n regime and produce an R^2 of several
    # negative units, which one fold is then enough to carry the mean. On the World
    # Bank one year is 32 rows against 24 columns and this is not hypothetical. The
    # summary the curve is fitted on is the median, and the mean stays visible so
    # the instability is legible rather than smoothed away.
    print(f"\n{'arm':<9} {'dist':>5} {'rows':>6} {'R^2 med':>9} {'mean':>9} "
          f"{'sd':>8} {'folds':>6}")
    grouped = frame.groupby('arm')
    order = grouped['distance'].mean().sort_values()
    for arm in order.index:
        g = grouped.get_group(arm)
        print(f"{arm:<9} {g['distance'].mean():>5.1f} {g['rows'].mean():>6.0f} "
              f"{g['r2'].median():>9.4f} {g['r2'].mean():>9.4f} "
              f"{g['r2'].std():>8.4f} {g['fold'].nunique():>6}")
    per_year, curve = slope(frame)
    print(f"\n  predictiveness decays {per_year:+.5f} of median R^2 per year")
    return {'dataset': dataset, 'slope': per_year, 'frame': frame, 'curve': curve}


def main():
    datasets = sys.argv[1:] or ['worldbank']
    results = {}
    for dataset in datasets:
        print('=' * 78)
        out = report(dataset)
        if out is not None:
            results[dataset] = out

    if len(results) >= 2:
        print('\n' + '=' * 78)
        print('SLOPE RATIO BETWEEN PANELS -- the quantity 4.2s predicts')
        names = list(results)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa, sb = results[a]['slope'], results[b]['slope']
                print(f"  {a} {sa:+.5f}  vs  {b} {sb:+.5f}   "
                      f"ratio {sa / sb:.3f}" if sb else f"  {b} slope is zero")
        print('\n  Compare against the channel slopes recorded in 4.2j and 4.2k.')
        print('  Prediction 2 holds if this ratio is within 30% of that one.')


if __name__ == '__main__':
    main()
