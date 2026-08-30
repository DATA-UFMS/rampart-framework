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
                 handed/unhanded, S = gain on unhanded. Shares the
                 replicated-saturation audit's marginal fixed-size design
                 (count = round(s * n_eval) evaluation rows, drawn without
                 replacement) but NOT its inserted rows: the rs audit seeds
                 fold_rng(fold, saturation, rep) and this probe seeds
                 fold_rng(fold, distance, saturation, rep), so the two draw
                 different sets (WB fold 0, s = 0.30: 0 of 40 masks
                 identical, mean overlap 3.8 of 13 rows). The d = 0 contrast
                 with rs is therefore UNPAIRED -- two independent samples of
                 the same design, never compared replicate by replicate.
                 Nor is it the same treatment:
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
  P1  RAMPART_SELFTEST=1, on every interior arm of every fold (RAMPART_FOLDS
      filters): mutating the withheld year in the raw panel (target*10+7,
      features*3+1, propagated into the precomputed lag columns) leaves
      every object the arm derives BIT-IDENTICAL -- the sliced X_tr, y_tr,
      X_te, y_te, the fill (training-window medians), the rebuilt lag
      columns, the entity-effect frames and means -- and, with them, the
      clean predictions of the ridge and the unbounded tree. The comparison
      names the object that differs. Predictions alone are a lossy
      projection (a surviving channel the two fits ignore would pass), so
      each arm also runs a SENTINEL: one feature column of every row of
      every year set to the withheld year's mean target, a channel that
      survives row filtering and that the fits barely read -- the tree not
      at all, the centred ridge only through the rounding residue of the
      centring (measured on WB: predictions bit-identical on 52 of 108 arms,
      moved by ~5e-12 on the other 56). The prediction check is therefore
      unreliable on this channel; the frame comparison must FAIL on the
      sentinel and PASS on the honest rebuild, and the selftest exits
      non-zero unless both hold on every arm.
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
    RAMPART_SELFTEST     run P1 on every interior arm (RAMPART_FOLDS filters)
                         and exit

Run: .venv/bin/python scripts/validation/probe_exposure_mapping.py [dataset]
"""

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_harness import (LAGS, declare_provenance, fold_rng, folds,
                           ladder_roster, panel)

from core.models.ladder import RUNGS, entity_effect_frames  # noqa: E402

DISTANCES = tuple(int(x) for x in
                  os.environ.get('RAMPART_DISTANCES',
                                 '0,1,2,3,4,5,6,8,10').split(','))
SATURATIONS = tuple(float(x) for x in
                    os.environ.get('RAMPART_SATURATIONS', '0.10,0.30').split(','))
MODELS = ladder_roster()
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


#: The two witnesses of the prediction check, fixed by design rather than read
#: from the roster: a RAMPART_MODELS subset without them would otherwise make
#: the check vacuous (no fits, `all([])` is True).
SELFTEST_RUNGS = ('ladder_ridge', 'ladder_decision_tree')


def mutate_withheld(df, columns, year_d):
    """The P1 mutation: every value of the withheld year moved, including the
    copies the precomputed lag columns hold at year_d + k.

    An affine map rather than a pure scale, so a zero moves too. The lag
    propagation is there because today rebuild_lags drops and rebuilds those
    columns, but a future edit that reused the stale ones would be invisible
    to a mutation that skips them -- the verification pass named this
    blindness, so the mutation closes it.
    """
    out = df.copy()
    hit = out['year'] == year_d
    out.loc[hit, 'target'] = out.loc[hit, 'target'] * 10 + 7
    for c in columns:
        if c in out.columns and not c.startswith('lag_'):
            out.loc[hit, c] = out.loc[hit, c] * 3 + 1
    src = (out.loc[hit, ['entity_id', 'target']]
           .set_index('entity_id')['target'])
    for k in LAGS:
        at = out['year'] == year_d + k
        out.loc[at, f'lag_{k}'] = (out.loc[at, 'entity_id'].map(src)
                                   .fillna(out.loc[at, f'lag_{k}']))
    return out


def sentinel_channel(df, columns, year_d):
    """A leak that survives row filtering: one feature column of EVERY row of
    every year replaced by the withheld year's mean target.

    Dropping the year's rows does not remove it and rebuilding the lags does
    not touch it. A constant column is also what the fits barely read: the
    tree has nothing to split on, and the centred ridge sees only the
    rounding residue of the centring (~1e-15), which moves its predictions
    by ~1e-12 on some arms and not at all on others. So it is the channel
    the prediction check cannot be relied on to see -- the frame comparison
    has to.
    """
    col = next(c for c in columns if not c.startswith('lag_'))
    out = df.copy()
    out[col] = float(df.loc[df['year'] == year_d, 'target'].mean())
    return out, col


def interior_objects(frame, columns, a, b, ts, te, distance):
    """Every object the interior arm derives from the panel, by name, so a
    comparison can say which one leaked.

    The insertion pool is left out on purpose: it is the withheld year's own
    rows and is meant to carry the mutation. Nothing else may.
    """
    year_d = ts - distance
    made = arm_frames(frame, columns, a, b, ts, te, distance)
    if made is None:
        return None
    X_tr, y_tr, e_tr, X_te, y_te, e_te, yr_te, _pool = made
    arm_df = rebuild_lags(frame, year_d)         # what arm_frames sliced
    fill = sliced(arm_df, columns, a, b, ts, te)[-1]
    fit_frame, eval_frame, means, global_mean = entity_effect_frames(
        X_tr, X_te, y_tr, e_tr, e_te)
    return {
        'X_tr': X_tr, 'y_tr': y_tr, 'e_tr': e_tr,
        'X_te': X_te, 'y_te': y_te, 'e_te': e_te, 'yr_te': yr_te,
        'fill': fill,
        'rebuilt_lags': arm_df[['entity_id', 'year']
                               + [f'lag_{k}' for k in LAGS]],
        'fit_frame': fit_frame, 'eval_frame': eval_frame,
        'entity_means': pd.Series(means, dtype=float).sort_index(),
        'global_mean': np.float64(global_mean),
    }


def differing(lhs, rhs):
    """Names of the objects that are not equal value for value. pandas
    `equals`: same shape, dtypes, index and values, NaN matching NaN."""
    out = []
    for name, x in lhs.items():
        y = rhs[name]
        same = (x.equals(y) if isinstance(x, (pd.DataFrame, pd.Series))
                else np.array_equal(x, y))
        if not same:
            out.append(name)
    return out


def clean_predictions(objects):
    """The original P1 check, kept: ridge and unbounded tree fit on the arm."""
    preds = []
    for name in SELFTEST_RUNGS:
        model = RUNGS[name].make()
        model.fit(objects['fit_frame'], objects['y_tr'])
        preds.append(np.asarray(model.predict(objects['eval_frame']),
                                dtype=float))
    return preds


def compare_arms(frames, columns, a, b, ts, te, distance):
    """Frame-level and prediction-level agreement of the interior arm built
    from two panels. Returns (names of differing objects, predictions equal),
    or None when the arm is unrealisable on either panel."""
    objects = [interior_objects(f, columns, a, b, ts, te, distance)
               for f in frames]
    if any(o is None for o in objects):
        return None
    preds = [clean_predictions(o) for o in objects]
    return (differing(*objects),
            all(np.array_equal(x, y) for x, y in zip(*preds)))


def selftest(dataset='worldbank_clean'):
    """P1, per interior arm: the arm must be blind to the withheld year, bit
    for bit.

    Two panels per arm, the raw one and one with the withheld year mutated
    (`mutate_withheld`). Every object the arm derives must be equal, and so
    must the ridge's and the tree's clean predictions -- the original check,
    kept, but no longer the only one, because predictions are a lossy
    projection of the frames. The sentinel run is the demonstration: the
    same comparison with `sentinel_channel` applied to both panels must
    FAIL on the frames, whether or not the predictions notice. Exits
    non-zero unless every arm fails on the sentinel and passes honestly.
    """
    df, columns, cfg = panel(dataset)
    arms = failures = 0
    started = time.perf_counter()
    for fold, (a, b, ts, te) in enumerate(folds(cfg)):
        if FOLD_FILTER is not None and fold not in FOLD_FILTER:
            continue
        for distance in range(ts - b, ts - a + 1):      # a <= year_d <= b
            year_d = ts - distance
            t0 = time.perf_counter()
            mutated = mutate_withheld(df, columns, year_d)
            honest = compare_arms((df, mutated), columns, a, b, ts, te,
                                  distance)
            if honest is None:
                print(f'  fold {fold} d={distance} (year {year_d}): '
                      f'unrealisable, skipped', flush=True)
                continue
            leaked, col = zip(*(sentinel_channel(f, columns, year_d)
                                for f in (df, mutated)))
            sentinel = compare_arms(leaked, columns, a, b, ts, te, distance)
            arms += 1
            head = f'P1 fold {fold} d={distance:>2} (withheld {year_d})'
            s_diff, s_preds = sentinel
            verdict = ('FAIL, as required (leak detected)' if s_diff
                       else 'PASS: LEAK NOT DETECTED')
            print(f'  {head} sentinel[{col[0]}]: frames '
                  f'{"differ " + str(s_diff) if s_diff else "identical"}, '
                  f'preds {"identical" if s_preds else "differ"} -> '
                  f'{verdict}', flush=True)
            h_diff, h_preds = honest
            ok = not h_diff and h_preds
            print(f'  {head} honest rebuild: frames '
                  f'{"differ " + str(h_diff) if h_diff else "identical"}, '
                  f'preds {"identical" if h_preds else "differ"} -> '
                  f'{"PASS" if ok else "FAIL"}   '
                  f'[{time.perf_counter() - t0:.1f}s]', flush=True)
            failures += not (ok and s_diff)
    print(f'P1 selftest on {dataset}: {arms} interior arms, '
          f'{arms - failures} with sentinel FAIL and honest PASS, '
          f'{failures} failing, {time.perf_counter() - started:.1f}s -> '
          f'{"PASS" if arms and not failures else "FAIL"}')
    return 0 if arms and not failures else 1


def main(dataset='worldbank_clean'):
    if os.environ.get('RAMPART_SELFTEST'):
        return selftest(dataset)
    df, columns, cfg = panel(dataset)
    windows = folds(cfg)
    print(f'{dataset}: {len(windows)} folds, {len(MODELS)} models, '
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
            for rung in MODELS:
                model = rung.make()
                model.fit(fit_frame, y_tr)
                preds = np.asarray(model.predict(eval_frame), dtype=float)
                clean_losses[rung.name] = (truth - preds) ** 2
                records.append(per_row_frame(
                    fold, rung.name, distance, year_d, 0.0, 0, 'clean',
                    yr_te, e_te, no_mask, truth, preds))

            for sat in SATURATIONS:
                count = max(1, int(round(sat * len(X_te))))
                s_reads = {rung.name: [] for rung in MODELS}
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
                    for rung in MODELS:
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
                for rung in MODELS:
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
