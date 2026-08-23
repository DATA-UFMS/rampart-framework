#!/usr/bin/env python3
"""Exposure mapping S(s, d): spillover by temporal distance, lags rebuilt per arm.

The interference radius question: how far from the evaluation window can a
leaked row sit and still move the evaluation? For each distance d, insert
count = round(s * n_eval) rows from the year test_start - d into training and
read the mean per-row improvement of the evaluation rows -- replicated over
independent draws, additive per-row losses, the same design-based machinery
as the replicated-saturation audit.

Three regimes of d, decided per panel from the walk-forward geometry:

    d = 0        rows from the evaluation window itself: rows split
                 handed/unhanded, S = gain on unhanded. Same draw as the
                 replicated-saturation audit but NOT the same treatment:
                 here the entity-effect column (a target encoding) is
                 recomputed on the widened training set at every d -- the
                 encoding seeing the leak is part of the treatment, and one
                 semantics has to hold across the whole curve. The rs audit
                 froze the encoding at the clean fit, so encoding-sensitive
                 models (the ridge above all) read S differently in the two
                 designs; encoding-insensitive ones agree.
    exterior d   the source year lies between train_end and test_start --
                 already excluded from training by protocol, so the standard
                 clean fit is the valid baseline.
    interior d   the source year W sits INSIDE the training window. The arm
                 withholds W from its own clean baseline AND -- the fix this
                 probe exists for -- rebuilds the target lags with W removed
                 from the lag source, so no remaining training row carries
                 target(W) through lag_2/lag_3. The earlier routes probe
                 filtered the rows but inherited full-panel lags, which is
                 the confound the external review named: its interior arms
                 never truly withheld the year.

Registered checks, written before the first run:
  P1  RAMPART_SELFTEST=1: multiplying every value of the withheld year by ten
      in the raw panel leaves the interior arm's clean predictions
      BIT-IDENTICAL (ridge and unbounded tree). If any channel from the
      withheld year survives -- rows, lags, imputation medians, entity
      means -- this fails loudly.
  P2  At d = 0 the S readings of encoding-INSENSITIVE models (random
      forest, gradient boosting) land in the range the replicated-
      saturation audit measured on the same panels. No agreement is
      predicted for the ridge: the recomputed target encoding is a real
      difference in treatment, not noise (measured before the first run:
      WB fold 0 ridge S +20.4 frozen vs +13.1 recomputed).
  P3  S(s, d) is read as a curve; no monotonicity is assumed. The
      interference radius is the smallest d whose interval sits inside the
      practical-equivalence region around zero, decided at analysis time.

Insertion mirrors the audit's contract: pools are imputed with the arm's own
training-window median (P5: the fill is computed before contamination), the
entity-effect column is recomputed on the widened training set (the target
encoding seeing the leak is part of the treatment), inserted rows keep their
true years and labels. Per-row records go to one parquet per (dataset, fold)
with `distance` and `inserted_year` columns; `arm = clean` rows are written
once per (model, distance) because interior distances have their own
baselines. Shards by replicate blocks (RAMPART_REP_OFFSET + RAMPART_REPS)
repeat the clean rows identically -- the reader deduplicates.

Environment knobs:
    RAMPART_DISTANCES    default 0,1,2,3,4,5,6,8,10  (skipped if unrealisable)
    RAMPART_SATURATIONS  default 0.10,0.30
    RAMPART_REPS         replicates in this shard, default 40
    RAMPART_REP_OFFSET   absolute index of the first replicate, default 0
    RAMPART_FOLDS        optional fold filter
    RAMPART_SELFTEST     run P1 and exit

Run: .venv/bin/python scripts/validation/probe_exposure_mapping.py [dataset]
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_harness import LAGS, declare_provenance, fold_rng, folds, panel

from core.models.ladder import LADDER, entity_effect_frames  # noqa: E402

DISTANCES = tuple(int(x) for x in
                  os.environ.get('RAMPART_DISTANCES',
                                 '0,1,2,3,4,5,6,8,10').split(','))
SATURATIONS = tuple(float(x) for x in
                    os.environ.get('RAMPART_SATURATIONS', '0.10,0.30').split(','))
REPS = int(os.environ.get('RAMPART_REPS', '40'))
REP_OFFSET = int(os.environ.get('RAMPART_REP_OFFSET', '0'))
FOLD_FILTER = (tuple(int(x) for x in os.environ['RAMPART_FOLDS'].split(','))
               if os.environ.get('RAMPART_FOLDS') else None)


def rebuild_lags(df, withheld_year):
    """The arm's frame: lag source loses the withheld year, then its rows go.

    The join is exact-year, as in probe_harness.panel(): a lag whose source
    year is withheld becomes NaN and is later filled with the arm's own
    training median -- never a positional shift, which would quietly span
    the hole (tests/test_lag_anti_leak.py exists for that property).
    """
    out = df.drop(columns=[f'lag_{k}' for k in LAGS])
    source = df.loc[df['year'] != withheld_year,
                    ['entity_id', 'year', 'target']]
    for k in LAGS:
        lag = source.copy()
        lag['year'] += k
        out = out.merge(lag.rename(columns={'target': f'lag_{k}'}),
                        on=['entity_id', 'year'], how='left')
    return out[out['year'] != withheld_year].reset_index(drop=True)


def sliced(df, columns, a, b, test_start, test_end):
    """prepared()'s contract, kept locally because the arm also needs `fill`:
    stable year sort, training-window-only median, rows still NaN dropped."""
    train = df[(df['year'] >= a) & (df['year'] <= b)].sort_values(
        'year', kind='stable')
    test = df[(df['year'] >= test_start) & (df['year'] <= test_end)].sort_values(
        'year', kind='stable')
    fill = train[columns].median()
    frames = []
    for part in (train, test):
        X = part[columns].fillna(fill)
        ok = X.notna().all(axis=1) & part['target'].notna()
        frames.append((X[ok].reset_index(drop=True),
                       part['target'][ok].reset_index(drop=True),
                       part['entity_id'][ok].reset_index(drop=True),
                       part['year'][ok].reset_index(drop=True)))
    (X_tr, y_tr, e_tr, _), (X_te, y_te, e_te, yr_te) = frames
    if len(X_tr) < 20 or len(X_te) < 3:
        return None
    return X_tr, y_tr, e_tr, X_te, y_te, e_te, yr_te, fill


def pool_at(df, columns, year, fill):
    """Insertable rows of one panel year, imputed with the ARM's fill (P5)."""
    part = df[df['year'] == year]
    X = part[columns].fillna(fill)
    ok = X.notna().all(axis=1) & part['target'].notna()
    return (X[ok].reset_index(drop=True),
            part['target'][ok].reset_index(drop=True),
            part['entity_id'][ok].reset_index(drop=True))


def per_row_frame(fold, name, distance, year_d, sat, rep, arm, yr, ent,
                  handed_mask, truth, preds):
    return pd.DataFrame({
        'fold': np.int16(fold),
        'model': pd.Categorical([name] * len(truth)),
        'distance': np.int16(distance),
        'inserted_year': np.int16(year_d),
        'saturation': np.float32(sat),
        'rep': np.int16(rep),
        'arm': pd.Categorical([arm] * len(truth)),
        'row': np.arange(len(truth), dtype=np.int32),
        'year': np.asarray(yr, dtype=np.int16),
        'entity': pd.Categorical(np.asarray(ent).astype(str)),
        'handed': np.asarray(handed_mask, dtype=bool),
        'y_true': np.asarray(truth, dtype=np.float32),
        'y_pred': np.asarray(preds, dtype=np.float32),
    })


def arm_frames(df, columns, a, b, test_start, test_end, distance):
    """The arm's frames plus its insertion pool, per the regime of d."""
    year_d = test_start - distance
    if distance == 0:
        made = sliced(df, columns, a, b, test_start, test_end)
        if made is None:
            return None
        return made[:7] + ('window',)
    if year_d > b:            # exterior: protocol already withholds the year
        made = sliced(df, columns, a, b, test_start, test_end)
        if made is None:
            return None
        X_tr, y_tr, e_tr, X_te, y_te, e_te, yr_te, fill = made
        return (X_tr, y_tr, e_tr, X_te, y_te, e_te, yr_te,
                pool_at(df, columns, year_d, fill))
    if year_d < int(df['year'].min()):
        return None
    arm_df = rebuild_lags(df, year_d)          # interior: withhold + re-lag
    made = sliced(arm_df, columns, a, b, test_start, test_end)
    if made is None:
        return None
    X_tr, y_tr, e_tr, X_te, y_te, e_te, yr_te, fill = made
    # The pool keeps the ORIGINAL lags of the withheld year's own rows: what
    # they know about earlier training years is part of the treatment.
    return (X_tr, y_tr, e_tr, X_te, y_te, e_te, yr_te,
            pool_at(df, columns, year_d, fill))


def selftest(dataset='worldbank_clean'):
    """P1: the interior arm must be blind to the withheld year, bit for bit."""
    df, columns, cfg = panel(dataset)
    a, b, ts, te = folds(cfg)[0]
    distance = ts - ((a + b) // 2)             # a mid-training interior year
    year_d = ts - distance
    mutated = df.copy()
    hit = mutated['year'] == year_d
    mutated.loc[hit, 'target'] = mutated.loc[hit, 'target'] * 10 + 7
    for c in columns:
        if c in mutated.columns and not c.startswith('lag_'):
            mutated.loc[hit, c] = mutated.loc[hit, c] * 3 + 1
    # Propagate the mutation into the panel's precomputed lag columns too:
    # rows at year_d + k hold target(year_d) in lag_k. Today rebuild_lags
    # drops and rebuilds them, but a future edit that reused the stale
    # columns would be invisible to a mutation that skips them -- the
    # verification pass named this blindness, so the mutation closes it.
    from probe_harness import LAGS as _lags
    for k in _lags:
        src = mutated.loc[mutated['year'] == year_d,
                          ['entity_id', 'target']].set_index('entity_id')['target']
        at = mutated['year'] == year_d + k
        mutated.loc[at, f'lag_{k}'] = (
            mutated.loc[at, 'entity_id'].map(src)
            .fillna(mutated.loc[at, f'lag_{k}']))
    outputs = []
    for frame in (df, mutated):
        X_tr, y_tr, e_tr, X_te, _y, e_te, _yr, _pool = arm_frames(
            frame, columns, a, b, ts, te, distance)
        fit_frame, eval_frame, _m, _g = entity_effect_frames(
            X_tr, X_te, y_tr, e_tr, e_te)
        preds = []
        for rung in LADDER:
            if rung.name not in ('ladder_ridge', 'ladder_decision_tree'):
                continue
            model = rung.make()
            model.fit(fit_frame, y_tr)
            preds.append(np.asarray(model.predict(eval_frame), dtype=float))
        outputs.append(preds)
    identical = all(np.array_equal(x, y) for x, y in zip(*outputs))
    print(f'P1 selftest on {dataset}: withheld year {year_d} mutated '
          f'(target*10+7, features*3+1); interior-arm clean predictions '
          f'bit-identical: {"PASS" if identical else "FAIL"}')
    return 0 if identical else 1


def main(dataset='worldbank_clean'):
    if os.environ.get('RAMPART_SELFTEST'):
        return selftest(dataset)
    df, columns, cfg = panel(dataset)
    windows = folds(cfg)
    print(f'{dataset}: {len(windows)} folds, {len(LADDER)} classical models, '
          f'distances {list(DISTANCES)}, saturations {list(SATURATIONS)}, '
          f'replicates [{REP_OFFSET}, {REP_OFFSET + REPS})')
    declare_provenance(distances=DISTANCES, saturations=SATURATIONS,
                       reps=REPS, rep_offset=REP_OFFSET,
                       fold_filter=FOLD_FILTER or 'all')
    print('  per-arm baselines: interior distances withhold their year and '
          'rebuild lags\n')

    for fold, (a, b, ts, te) in enumerate(windows):
        if FOLD_FILTER is not None and fold not in FOLD_FILTER:
            continue
        records = []
        for distance in DISTANCES:
            made = arm_frames(df, columns, a, b, ts, te, distance)
            if made is None:
                print(f'  fold {fold} d={distance}: unrealisable, skipped')
                continue
            X_tr, y_tr, e_tr, X_te, y_te, e_te, yr_te, pool = made
            year_d = ts - distance
            truth = np.asarray(y_te, dtype=float)
            no_mask = np.zeros(len(X_te), dtype=bool)

            fit_frame, eval_frame, _m, _g = entity_effect_frames(
                X_tr, X_te, y_tr, e_tr, e_te)
            clean_losses = {}
            for rung in LADDER:
                model = rung.make()
                model.fit(fit_frame, y_tr)
                preds = np.asarray(model.predict(eval_frame), dtype=float)
                clean_losses[rung.name] = (truth - preds) ** 2
                records.append(per_row_frame(
                    fold, rung.name, distance, year_d, 0.0, 0, 'clean',
                    yr_te, e_te, no_mask, truth, preds))

            for sat in SATURATIONS:
                count = max(1, int(round(sat * len(X_te))))
                s_reads = {rung.name: [] for rung in LADDER}
                for rep in range(REP_OFFSET, REP_OFFSET + REPS):
                    rng = fold_rng(fold, distance, sat, rep)
                    if distance == 0:
                        handed = np.sort(rng.choice(len(X_te), size=count,
                                                    replace=False))
                        mask = np.zeros(len(X_te), dtype=bool)
                        mask[handed] = True
                        Xa, ya, ea = (X_te.iloc[handed], y_te.iloc[handed],
                                      e_te.iloc[handed])
                    else:
                        pool_X, pool_y, pool_e = pool
                        if not len(pool_X):
                            break
                        picked = np.sort(rng.choice(
                            len(pool_X), size=count,
                            replace=len(pool_X) < count))
                        mask = no_mask
                        Xa, ya, ea = (pool_X.iloc[picked], pool_y.iloc[picked],
                                      pool_e.iloc[picked])
                    wide_X, wide_eval, _wm, _wg = entity_effect_frames(
                        pd.concat([X_tr, Xa], ignore_index=True), X_te,
                        pd.concat([y_tr, ya], ignore_index=True),
                        pd.concat([e_tr, ea], ignore_index=True), e_te)
                    wide_y = pd.concat([y_tr, ya], ignore_index=True)
                    for rung in LADDER:
                        model = rung.make()
                        model.fit(wide_X, wide_y)
                        preds = np.asarray(model.predict(wide_eval),
                                           dtype=float)
                        records.append(per_row_frame(
                            fold, rung.name, distance, year_d, sat, rep,
                            'leak', yr_te, e_te, mask, truth, preds))
                        gain = clean_losses[rung.name] - (truth - preds) ** 2
                        s_reads[rung.name].append(
                            float(gain[~mask].mean()))
                for rung in LADDER:
                    v = np.array(s_reads[rung.name])
                    if not len(v):
                        continue
                    print(f'  fold {fold} d={distance} s={sat:.2f} '
                          f'{rung.name:>26}: S {v.mean():+.4f} '
                          f'(rep sd {v.std(ddof=1) if len(v) > 1 else 0:.4f})',
                          flush=True)

        out = Path(f'exposure_mapping_{dataset}_fold{fold}'
                   f'_r{REP_OFFSET}.parquet')
        pd.concat(records, ignore_index=True).to_parquet(out, index=False)
        print(f'  fold {fold}: {sum(len(r) for r in records)} rows -> {out}',
              flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
