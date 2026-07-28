#!/usr/bin/env python3
"""Feasibility probe: is the capacity ladder detectable with n=9 folds?

Not the experiment. The experiment compares ICL against classical models; this
asks the question that has to be answered first, and needs no ICL code at all:
with the panels and the fold count we actually have, can we recover an effect
the literature says is there?

Roth reports, for Class III (memorisation), that severity scales with model
capacity -- d_z from 0.37 (naive Bayes) to 1.11 (decision tree) at 10%
duplication, measured across 2,047 datasets. Our design proposes to use the
Ridge-to-RandomForest step as the internal calibration for "how much of the
amplification is capacity alone". That only works if the step is visible here.

If it is not, the pre-registered positive control fails, and no statement about
in-context learning follows from this measurement. Better to know on 27 July
than on 8 August.

Simplifications, stated because they bound what the probe can conclude:
  - plain RidgeCV and RandomForestRegressor on the same design matrix, not the
    hierarchical wrappers the pipeline uses. The question is about capacity
    ordering, and the wrappers do not change which of the two has more.
  - one panel, World Bank, because it is the fast one.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/eos/pesquisa/eos/rampart-framework/src')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from datasets.worldbank import *  # noqa
from core.dataset_config import get_dataset
from core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED

PANEL = ('/home/eos/pesquisa/eos/dw-vs-dl-dropout-prediction-latam/'
         'azure_results_v7_wb/collection/raw_data/complete_data.parquet')
DOSES = (0.05, 0.10, 0.30)          # Roth's own duplication rates
LAGS = (2, 3)


def panel():
    cfg = get_dataset('worldbank')
    df = (pd.read_parquet(PANEL)
          .rename(columns={'country_code': 'entity_id',
                           'lower_secondary_completion_rate': 'target_source_rate'}))
    df['target'] = 100.0 - df['target_source_rate']
    for k in LAGS:
        lag = df[['entity_id', 'year', 'target']].copy()
        lag['year'] += k
        df = df.merge(lag.rename(columns={'target': f'lag_{k}'}),
                      on=['entity_id', 'year'], how='left')
    features = [c for c in cfg.feature_columns if c in df.columns]
    return df, features + [f'lag_{k}' for k in LAGS], cfg


def folds(cfg):
    w = cfg.walk_forward_config
    start, end = cfg.temporal_range
    out, train_end = [], start + w['min_train'] - 1
    while True:
        val_start = train_end + w['gap'] + 1
        val_end = val_start + w['val_len'] - 1
        test_start = val_end + w['gap'] + 1
        test_end = test_start + w['test_len'] - 1
        if test_end > end:
            return out
        out.append((start, train_end, test_start, test_end))
        train_end += w['step']


def models():
    hm = SCIENTIFIC_CONFIG['hierarchical_model']
    alphas = np.logspace(hm['ridge_alpha_log10_start'],
                         hm['ridge_alpha_log10_stop'], hm['ridge_alpha_count'])
    return {
        'ridge': lambda: RidgeCV(alphas=alphas),
        'random_forest': lambda: RandomForestRegressor(
            n_estimators=hm['rf_n_estimators'], max_depth=hm['rf_max_depth_grid'][-1],
            min_samples_split=hm['rf_min_samples_split'],
            min_samples_leaf=hm['rf_min_samples_leaf_grid'][0],
            max_features=hm['rf_max_features'],
            random_state=RANDOM_SEED, n_jobs=1),
    }


def score(make, train, test, cols):
    """R2 out of sample. Median imputation fitted on train only -- P5 holds even
    in the probe, because a leaky baseline would inflate both arms and hide the
    contrast the probe exists to measure."""
    fill = train[cols].median()
    Xtr, Xte = train[cols].fillna(fill), test[cols].fillna(fill)
    keep = Xtr.notna().all(axis=1) & train['target'].notna()
    if keep.sum() < 20 or Xte.isna().any().any():
        return None
    model = make()
    model.fit(Xtr[keep], train['target'][keep])
    valid = test['target'].notna()
    if valid.sum() < 3:
        return None
    return float(r2_score(test['target'][valid],
                          model.predict(Xte[valid])))


def main():
    df, cols, cfg = panel()
    windows = folds(cfg)
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"World Bank: {len(df)} rows, {len(cols)} columns, "
          f"{len(windows)} folds\n")

    rows = []
    for fold, (a, b, ts, te) in enumerate(windows):
        train = df[(df['year'] >= a) & (df['year'] <= b)]
        test = df[(df['year'] >= ts) & (df['year'] <= te)]
        for name, make in models().items():
            clean = score(make, train, test, cols)
            if clean is None:
                continue
            for dose in DOSES:
                # Class III, memorisation: a fraction of the test rows, with
                # their true labels, pasted into the training set.
                n = max(1, int(round(dose * len(test))))
                stolen = test.sample(n=n, random_state=int(rng.integers(1 << 31)))
                leaked = score(make, pd.concat([train, stolen]), test, cols)
                if leaked is None:
                    continue
                rows.append({'fold': fold, 'model': name, 'dose': dose,
                             'clean': clean, 'leaked': leaked,
                             'inflation': leaked - clean})

    out = pd.DataFrame(rows)
    print(f"{'dose':>6} {'model':>15} {'n':>3} {'mean I':>9} {'sd':>8} {'d_z':>8}")
    ladder = {}
    for dose in DOSES:
        for name in models():
            sub = out[(out['dose'] == dose) & (out['model'] == name)]
            if sub.empty:
                continue
            mean, sd = sub['inflation'].mean(), sub['inflation'].std(ddof=1)
            dz = mean / sd if sd > 0 else float('nan')
            ladder[(dose, name)] = dz
            print(f"{dose:>6.2f} {name:>15} {len(sub):>3} {mean:>9.4f} "
                  f"{sd:>8.4f} {dz:>8.3f}")
        print()

    print("POSITIVE CONTROL -- does capacity order the severity, as Roth reports?")
    passed = 0
    for dose in DOSES:
        r, f = ladder.get((dose, 'ridge')), ladder.get((dose, 'random_forest'))
        if r is None or f is None:
            continue
        ok = f > r
        passed += ok
        print(f"  dose {dose:.2f}: ridge d_z={r:+.3f}  RF d_z={f:+.3f}  "
              f"-> {'RF above, as expected' if ok else 'NOT reproduced'}")
    print(f"\n  reproduced at {passed}/{len(DOSES)} doses (per-model d_z, unstable)")

    # The stable form: the paired difference, with a block bootstrap over folds
    # that respects the overlap between consecutive World Bank test windows.
    print("\nSTABLE FORM -- paired difference in mean inflation, RF minus Ridge")
    rng2 = np.random.default_rng(RANDOM_SEED)
    for dose in DOSES:
        wide = (out[out['dose'] == dose]
                .pivot(index='fold', columns='model', values='inflation')
                .dropna())
        if wide.empty or 'ridge' not in wide or 'random_forest' not in wide:
            continue
        diff = (wide['random_forest'] - wide['ridge']).to_numpy()
        n = len(diff)
        # Moving-block bootstrap, block length 2: consecutive folds share a
        # test year, so resampling folds independently understates the spread.
        draws = []
        for _ in range(15000):
            picked = []
            while len(picked) < n:
                start = int(rng2.integers(0, max(1, n - 1)))
                picked.extend(diff[start:start + 2])
            draws.append(float(np.mean(picked[:n])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        dz = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else float('nan')
        verdict = 'ABOVE ZERO' if lo > 0 else 'covers zero'
        print(f"  dose {dose:>4.2f}: mean {np.mean(diff):+.4f}  "
              f"CI95 [{lo:+.4f}, {hi:+.4f}]  d_z(D)={dz:+.2f}  -> {verdict}")
    return 0 if passed >= 2 else 1


if __name__ == '__main__':
    raise SystemExit(main())
