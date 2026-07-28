#!/usr/bin/env python3
"""Selection leakage: a third channel the two-channel decomposition cannot see.

Roth measures four classes and finds selection the largest -- larger than
memorisation at practical sample sizes -- and reports that about 90% of the effect
is noise exploitation rather than genuine improvement. The channel decomposition
built here splits severity into memorisation, on rows the model was handed, and a
generalisation shift, on rows it was not. Selection leakage hands over no rows at
all: the same evaluation set is scored, and what leaks is *which configuration
scored best on it*. So the decomposition sees no memorisation channel and puts the
whole effect in the generalisation bucket, where it does not belong -- nothing
generalised, a winner was picked on noise.

This measures that, and gives the bucket its own test.

    HONEST     hyperparameters chosen on the validation window, as the protocol
               says. The baseline.
    PEEK       chosen on the evaluation window itself, then reported on it. The
               violation.
    SPLIT      chosen on one half of the evaluation window, reported on the other.
               The discriminating arm.

**Why SPLIT is the test.** If the peeking gain were genuine -- if the winning
configuration really were better -- it would still be better on rows that took no
part in choosing it. If the gain is noise exploitation, it evaporates. So

    genuine share  =  (SPLIT - HONEST) / (PEEK - HONEST)

with the remainder being the part that exists only because the same rows were used
to choose and to report. Roth's 90% noise share is a corpus-level estimate obtained
by a different route -- modelling the spread of scores -- so this is an independent
check of it rather than a replication.

**The grid is the protocol's own.** Residual shrinkage for the hierarchical ridge
and depth by leaf size for the forest, exactly the values
`SCIENTIFIC_CONFIG['hierarchical_model']` declares, because the question is what
the published pipeline's own selection step would buy an author who ran it against
the wrong window.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from probe_harness import folds, panel, prepared

from core.models.ladder import entity_effect_frames  # noqa: E402
from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG  # noqa: E402
from statistical_validation.dependent_bootstrap import (  # noqa: E402
    excludes_zero, fold_dependence_span, moving_block_ci)

ARMS = ('HONEST', 'PEEK', 'SPLIT')


def grids():
    """The configuration spaces the published pipeline already searches."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    _hm = SCIENTIFIC_CONFIG['hierarchical_model']
    alphas = np.logspace(_hm['ridge_alpha_log10_start'],
                         _hm['ridge_alpha_log10_stop'], _hm['ridge_alpha_count'])
    ridge = [(f'alpha={a:.3g}', (lambda a=a: Ridge(alpha=a))) for a in alphas]
    forest = [
        (f'depth={d},leaf={l}',
         (lambda d=d, l=l: RandomForestRegressor(
             n_estimators=_hm['rf_n_estimators'], max_depth=d,
             min_samples_split=_hm['rf_min_samples_split'],
             min_samples_leaf=l, max_features=_hm['rf_max_features'],
             random_state=RANDOM_SEED, n_jobs=_hm['rf_n_jobs'])))
        for d in _hm['rf_max_depth_grid'] for l in _hm['rf_min_samples_leaf_grid']]
    return {'ridge_grid': ridge, 'forest_grid': forest}


def score(truth, predicted, mask=None):
    if mask is None:
        return float(mean_squared_error(truth, predicted))
    if not mask.any():
        return float('nan')
    return float(mean_squared_error(np.asarray(truth)[mask],
                                    np.asarray(predicted)[mask]))


def main(dataset='worldbank', entity_cap=None):
    df, columns, cfg = panel(dataset)
    if entity_cap is not None:
        keep = sorted(df['entity_id'].unique())[:int(entity_cap)]
        df = df[df['entity_id'].isin(keep)].reset_index(drop=True)
        print(f"subsampled to {len(keep)} entities, {len(df)} rows")
    windows = folds(cfg)
    block = fold_dependence_span(cfg.walk_forward_config)
    spaces = grids()
    print(f"{dataset}: {len(windows)} folds, block {block}, "
          f"grids {[(k, len(v)) for k, v in spaces.items()]}\n")

    inflation = {(space, arm): [] for space in spaces for arm in ARMS}
    winners = {(space, arm): [] for space in spaces for arm in ARMS}

    gap = int(cfg.walk_forward_config['gap'])
    val_len = int(cfg.walk_forward_config['val_len'])

    for fold, (train_start, train_end, test_start, test_end) in enumerate(windows):
        val_start = train_end + 1 + gap
        val_end = val_start + val_len - 1
        made = prepared(df, columns, train_start, train_end, test_start, test_end)
        val_made = prepared(df, columns, train_start, train_end,
                            val_start, val_end)
        if made is None or val_made is None:
            continue
        (X_train, y_train, e_train, _yr,
         X_test, y_test, e_test, _yt) = made
        X_val, y_val, e_val = val_made[4], val_made[5], val_made[6]

        fit_frame, eval_frame, _m, _g = entity_effect_frames(
            X_train, X_test, y_train, e_train, e_test)
        _f, val_frame, _m, _g = entity_effect_frames(
            X_train, X_val, y_train, e_train, e_val)

        # The evaluation window split in two, deterministically by position so
        # the halves do not depend on a seed the arms could differ on.
        half = np.zeros(len(X_test), dtype=bool)
        half[::2] = True

        for space, candidates in spaces.items():
            fitted = {}
            for label, make in candidates:
                model = make()
                model.fit(fit_frame, y_train)
                fitted[label] = (
                    np.asarray(model.predict(val_frame), dtype=float),
                    np.asarray(model.predict(eval_frame), dtype=float))

            def best(on_val, mask=None):
                if on_val:
                    return min(fitted, key=lambda k: score(y_val, fitted[k][0]))
                return min(fitted,
                           key=lambda k: score(y_test, fitted[k][1], mask))

            chosen = {
                'HONEST': best(on_val=True),
                'PEEK': best(on_val=False),
                'SPLIT': best(on_val=False, mask=half),
            }
            # Every arm is reported on the same rows: the half that took no part
            # in any selection. Otherwise HONEST and PEEK would be scored on more
            # rows than SPLIT and the three would not be comparable.
            reported = ~half
            for arm, label in chosen.items():
                winners[(space, arm)].append(label)
                inflation[(space, arm)].append(
                    score(y_test, fitted[label][1], reported))
        print(f"  fold {fold}: done", flush=True)

    print("\n" + "=" * 78)
    print("mean squared error on the reported half, by arm")
    print(f"{'space':>12} " + ' '.join(f"{a:>12}" for a in ARMS))
    for space in spaces:
        print(f"{space:>12} " +
              ' '.join(f"{np.mean(inflation[(space, a)]):>12.4f}" for a in ARMS))

    print("\nselection inflation, as error removed relative to the honest arm")
    print(f"{'space':>12} {'PEEK':>10} {'SPLIT':>10} {'genuine share':>15}")
    for space in spaces:
        honest = np.asarray(inflation[(space, 'HONEST')])
        peek = np.asarray(inflation[(space, 'PEEK')])
        split = np.asarray(inflation[(space, 'SPLIT')])
        peek_gain = float(np.mean(1.0 - peek / honest))
        split_gain = float(np.mean(1.0 - split / honest))
        share = split_gain / peek_gain if abs(peek_gain) > 1e-9 else float('nan')
        print(f"{space:>12} {peek_gain:>+10.4f} {split_gain:>+10.4f} "
              f"{share:>15.3f}")

    print("\nis the peeking gain distinguishable from zero? (paired over folds)")
    print(f"{'space':>12} {'mean':>10} {'CI95':>26} {'verdict':>10}")
    for space in spaces:
        honest = np.asarray(inflation[(space, 'HONEST')])
        peek = np.asarray(inflation[(space, 'PEEK')])
        paired = 1.0 - peek / honest
        point, interval, _r = moving_block_ci(paired, block=block, iters=4000)
        print(f"{space:>12} {point:>+10.4f} "
              f"{f'[{interval[0]:+.4f}, {interval[1]:+.4f}]':>26} "
              f"{'above zero' if excludes_zero(interval, direction=+1) else 'covers zero':>10}")

    print("\nhow often does the peeking winner differ from the honest one?")
    for space in spaces:
        differ = sum(1 for h, p in zip(winners[(space, 'HONEST')],
                                       winners[(space, 'PEEK')]) if h != p)
        print(f"  {space:>12}: {differ}/{len(winners[(space, 'HONEST')])} folds")

    print("\n" + "=" * 78)
    print("WHAT THIS SAYS ABOUT THE DECOMPOSITION")
    print("  Selection leakage hands over no rows, so the memorisation channel is")
    print("  zero by construction and the two-channel split puts everything in the")
    print("  generalisation bucket. The genuine share above is what actually")
    print("  generalises; the remainder is a winner picked on noise, which is a")
    print("  third channel and needs its own arm rather than a finer partition.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
