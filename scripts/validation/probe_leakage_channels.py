#!/usr/bin/env python3
"""Aggregate severity adds two effects that order models oppositely.

Under class III the evaluation window splits into rows the model was handed and
rows it was not. Improvement on the first is memorisation; improvement on the
second is a change in generalisation. The usual severity measure sums over both,
and this asks what that costs.

It also settles the anomaly that redirected this study. Ranked by aggregate
inflation, the ridge appears more severely affected than the random forest, which
inverts the capacity ordering the literature documents. The question is whether
the ordering is wrong or the measure is.

Three things measured on the same folds, because they share every fit:

  1. **The channels**, with a control arm. The rows that did not leak also benefit
     from the training frame being larger, which is not leakage. So a second arm
     adds the same number of rows duplicated from inside the training window, and
     the difference between the arms on held-out rows is what leakage explains.

  2. **The calibration of the axis.** For k-nearest neighbours the duplicate sits
     at distance zero from its own query, so absorption should read (2k-1)/k^2 --
     an analytic value the instrument was not fitted to. Sweeping k gives a
     calibration curve spanning the range the in-context models occupy.

  3. **Where the in-context models fall**, when the extra is installed.

Run: python scripts/validation/probe_leakage_channels.py
     .venv/bin/python scripts/validation/probe_leakage_channels.py   (with ICL)
"""

import os
import sys
import warnings
from importlib.util import find_spec
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

# The harness first: importing it puts src/ on the path.
from probe_harness import DOSES, entity_subsample, fold_rng, folds, panel, prepared
from probe_harness import audit_resamples, declare_provenance

from core.models.absorption import (  # noqa: E402
    absorption_coefficient, knn_expected_absorption)
from core.models.icl import matched_context  # noqa: E402
from core.models.ladder import LADDER, entity_effect_frames  # noqa: E402
from core.scientific_config import (  # noqa: E402
    RANDOM_SEED, SCIENTIFIC_CONFIG)
from statistical_validation.dependent_bootstrap import (  # noqa: E402
    fold_dependence_span, moving_block_ci)
from statistical_validation.leakage_channels import (
    channel_points,  # noqa: E402
    decompose_fold, summarise)

#: Swept to give the axis a calibration curve with a closed form.
KNN_K = (1, 2, 3, 5, 10, 20)

#: Read from the configuration rather than set here, so the reading the probe
#: reports and the reading the pipeline records are the same quantity. The count
#: is part of the definition: it fixes the dose at which absorption is read.
#: `RAMPART_PROBES=313` overrides it. Needed for the matched-perturbation arm, and a
#: count rather than a share because under the context cap the frame handed in is the
#: full window while the model only reads the most recent ten thousand -- so a share
#: of what was handed in is not a share of what was read, and the count is the only
#: way to say 3.13% of the effective context on both panels.
PROBES = (int(os.environ['RAMPART_PROBES'])
          if os.environ.get('RAMPART_PROBES', '').strip() else None)

#: `RAMPART_PROBE_FRACTION=0.0313` reads absorption at a matched share of the training
#: frame instead of a matched count, which is the only way two panels are comparable:
#: at a fixed twelve the perturbation is 12/n, 3.13% on the World Bank against 0.029%
#: on INEP, and a quantity falling as 1/n was measuring that rather than the model.
#: Off by default, because the count is what every recorded number was read at.
PROBE_FRACTION = (float(os.environ['RAMPART_PROBE_FRACTION'])
                  if os.environ.get('RAMPART_PROBE_FRACTION', '').strip()
                  else None)


def knn(k):
    from sklearn.neighbors import KNeighborsRegressor
    return lambda: KNeighborsRegressor(n_neighbors=k)


def duplicated_rungs():
    """Named rungs that are the same estimator as a swept one, by construction.

    `ladder_knn` reads `capacity_ladder.knn_n_neighbors`, which is 5, and the sweep
    already contains `knn_k5`. They are one model appearing twice, so the thirteen
    rows in the table are twelve distinct estimators -- and the correlation between
    absorption and the local channel was giving k=5 double weight. It survives the
    correction (0.989 / 0.990 / 0.984 against 0.989 / 0.990 / 0.983), which is why
    this is a reporting fix and not a retraction, but the count and the weighting
    both have to be right before either goes in a table.
    """
    k = SCIENTIFIC_CONFIG['capacity_ladder']['knn_n_neighbors']
    return {'ladder_knn': f'knn_k{k}'} if k in KNN_K else {}


def candidates():
    # With RAMPART_CAP_ALL the classical factories are wrapped too, so every model
    # reads the same context and the absorption column becomes comparable across
    # families. Off by default: the asymmetry is the deployed configuration.
    entries = [(f'knn_k{k}', matched_context(knn(k)), 'sweep',
                knn_expected_absorption(k)) for k in KNN_K]
    entries += [(rung.name, matched_context(rung.make), 'named', None)
                for rung in LADDER]
    if find_spec('tabpfn') is not None or find_spec('tabicl') is not None:
        from core.models.icl import FAMILIES
        for family in FAMILIES:
            # `available()` and not `find_spec`: the v3 arm's package is present
            # whenever the v2 arm's is, so a presence check builds it without the
            # credential its weights need and the run dies mid-fold.
            if family.available():
                entries.append((family.name, family.make, 'in-context', None))
    return entries


def main(dataset='worldbank', entity_cap=None):
    df, columns, cfg = panel(dataset)
    if entity_cap is not None:
        df = entity_subsample(df, entity_cap)
        # `keep` lives inside entity_subsample; naming it here raised NameError on
        # every capped run, so the subsample path had never actually printed.
        print(f"subsampled to {df['entity_id'].nunique()} entities, {len(df)} rows")
    windows = folds(cfg)
    entries = candidates()
    duplicates = set(duplicated_rungs())
    block = fold_dependence_span(cfg.walk_forward_config)
    distinct = len(entries) - len([n for n, *_ in entries if n in duplicates])
    # The reading settings go in the header, because absorption at one replicate and
    # absorption at five are different numbers on the same panel -- worldbank_clean
    # read 0.0514 at one and 0.0270 at five -- and a log that does not say which will
    # be compared against one that does not either.
    reps = SCIENTIFIC_CONFIG['in_context_models']['absorption_replicates']
    probes = PROBES or SCIENTIFIC_CONFIG['in_context_models']['absorption_probes']
    print(f"{dataset}: {len(windows)} folds, {distinct} distinct models "
          f"({len(entries)} rows, {len(entries) - distinct} duplicated by "
          f"construction), block {block}")
    declare_provenance(**{
        'absorption_probes (in use)': probes,
        'probe_fraction': PROBE_FRACTION if PROBE_FRACTION else 'not set',
        'fold_block': block,
    })
    print()

    channels = {}        # (name, dose) -> list of per-fold decompositions
    absorptions = {}     # name -> list of per-fold readings

    for fold, (a, b, test_start, test_end) in enumerate(windows):
        made = prepared(df, columns, a, b, test_start, test_end)
        if made is None:
            continue
        (X_train, y_train, e_train, yr_train,
         X_test, y_test, e_test, yr_test) = made
        fit_frame, eval_frame, _m, _g = entity_effect_frames(
            X_train, X_test, y_train, e_train, e_test)
        truth = np.asarray(y_test, dtype=float)

        clean_predictions = {}
        for name, make, _kind, _expected in entries:
            model = make()
            model.fit(fit_frame, y_train)
            clean_predictions[name] = np.asarray(model.predict(eval_frame),
                                                 dtype=float)
            absorptions.setdefault(name, []).append(absorption_coefficient(
                make, fit_frame, y_train, eval_frame, y_test,
                probes=PROBES, fraction=PROBE_FRACTION,
                seed=RANDOM_SEED + fold,
                baseline=clean_predictions[name])['absorption'])

        for dose in DOSES:
            # Drawn exactly as core.injection draws it: the picked rows first,
            # then the control arm's echo, from one generator. Both arms name the
            # same partition, which is what makes them comparable.
            rng = fold_rng(fold, dose)
            count = max(1, int(round(dose * len(eval_frame))))
            handed = np.sort(rng.choice(len(eval_frame), size=count,
                                        replace=False))
            echoed = np.sort(rng.choice(len(fit_frame), size=count,
                                        replace=True))
            mask = np.zeros(len(eval_frame), dtype=bool)
            mask[handed] = True

            # The years do not travel with the arms, and they do not need to. The
            # cap takes the tail of the frame it is handed, and that is the recency
            # rule only because `prepared` sorts by year and appended rows are newer
            # than all of training. An earlier version built the per-arm year vectors
            # here and discarded them, under a comment asserting the opposite -- dead
            # code defending a claim the code did not implement, which is what a
            # reviewer finds with one grep.
            arms = {
                'leak': (pd.concat([fit_frame, eval_frame.iloc[handed]],
                                   ignore_index=True),
                         pd.concat([y_train, y_test.iloc[handed]],
                                   ignore_index=True)),
                'control': (pd.concat([fit_frame, fit_frame.iloc[echoed]],
                                      ignore_index=True),
                            pd.concat([y_train, y_train.iloc[echoed]],
                                      ignore_index=True)),
            }

            for name, make, _kind, _expected in entries:
                fitted = {}
                for arm, (Xa, ya) in arms.items():
                    model = make()
                    model.fit(Xa, ya)
                    fitted[arm] = np.asarray(model.predict(eval_frame),
                                             dtype=float)
                channels.setdefault((name, dose), []).append(decompose_fold(
                    truth, clean_predictions[name], fitted['leak'],
                    mask=mask, control=fitted['control']))
        print(f"  fold {fold}: done", flush=True)

    # Absorption never passed through summarise -- it was a bare np.nanmean over the
    # per-fold list -- so the axis column and every ratio built from it had no interval
    # even in discarded form. The drop ratios promoted to a finding and then withdrawn
    # had none, and could not have had one. Same block bootstrap as every other channel.
    absorption_ci = {}
    for name, values in absorptions.items():
        clean = [v for v in values if v is not None and np.isfinite(v)]
        # Configured count, not a local one: this interval is reported in the
        # calibration table, and an interval whose resample count differs from the
        # one the paper states is the defect the resample audit exists to surface.
        point, interval, _rec = moving_block_ci(clean, block=block)
        absorption_ci[name] = interval
    mean_absorption = {name: float(np.nanmean(values))
                       for name, values in absorptions.items()}

    print("\n" + "=" * 78)
    print("1. IS THE AXIS CALIBRATED? kNN absorption against (2k-1)/k^2")
    print(f"{'config':>12} {'analytic':>10} {'measured':>10} {'gap':>8} "
          f"{'ci95 on measured':>22} {'covers?':>8}")
    worst = 0.0
    for name, _make, kind, expected in entries:
        if kind != 'sweep':
            continue
        measured = mean_absorption[name]
        low, high = absorption_ci[name]
        worst = max(worst, abs(measured - expected))
        # Whether the closed form is inside the interval is the question the gap
        # column was being read as answering, and it is not the same question: a
        # gap of 0.03 with an interval of width 0.30 is agreement, and a gap of
        # 0.03 with an interval of width 0.01 is not.
        covers = 'yes' if low <= expected <= high else 'NO'
        print(f"{name:>12} {expected:>10.4f} {measured:>10.4f} "
              f"{measured - expected:>+8.4f} "
              f"[{low:>+8.4f},{high:>+8.4f}] {covers:>8}")
    print(f"\n  largest gap {worst:.4f}, and the column that matters is the last "
          f"one: the gap alone never said whether the fold set can resolve it.")

    print("\n2. THE TWO CHANNELS, by dose. local = memorisation, "
          "global = generalisation shift")
    print("   (global already has the sample-size control subtracted)")
    for dose in DOSES:
        print(f"\n  dose {dose:.0%}")
        print(f"{'model':>26} {'absorp':>8} {'local':>9} {'excess':>9} "
              f"{'global':>9} {'size':>8} {'aggreg':>9}")
        for name, _make, _kind, _expected in entries:
            # One model counted twice would weight its k twice in the correlation.
            if name in duplicates:
                continue
            folds_here = channels.get((name, dose), [])
            if not folds_here:
                continue
            # No iters= here. This is the one place that reads ci95, and passing a
            # count of its own is how Table 3's intervals came from 4,000 while the
            # protocol declared 15,000.
            got = summarise(folds_here, block=block)
            # The identity check is computed per fold inside decompose_fold; printing
            # its coverage here puts the confirmation in the log itself, so the paper
            # can point at the artifact rather than at the test suite alone.
            cov = got.get('coverage', {})
            if cov.get('identity_holds') is False:
                print(f"        IDENTITY FAILED for {name} at dose {dose}")
            # The intervals were computed here all along and thrown away: summarise
            # returns ci95 for every channel and four probes indexed ['point'] and
            # nothing else. What that hid is not academic -- the adjacent-buffer
            # headline, "leaking the buffer costs more than leaking the test window",
            # is a ratio of 1.049 whose interval is [0.908, 1.238] and covers one.
            def band(channel, width=9):
                lo, hi = got[channel]['ci95']
                return (f"{got[channel]['point']:>+{width}.4f}"
                        f"[{lo:+.3f},{hi:+.3f}]")
            wbar = got['local_weight']['point']
            print(f"{name:>26} {mean_absorption[name]:>8.4f} "
                  f"{band('local')} {band('local_excess')} "
                  f"{band('global')} {band('sample_size_effect', 8)} "
                  f"{band('aggregate')} w={wbar:.3f}")

    print("\n" + "=" * 78)
    print("3. WHICH CHANNEL DOES ABSORPTION EXPLAIN?")
    print(f"{'dose':>6} {'r(absorption, local)':>22} "
          f"{'r(absorption, global)':>23} {'r(absorption, aggregate)':>26}")
    for dose in DOSES:
        columns_ = {'local': [], 'global': [], 'aggregate': [], 'axis': []}
        for name, _make, _kind, _expected in entries:
            # The duplicate rung is dropped here for the same reason section 2 drops
            # it: `ladder_knn` IS `knn_k5`, and leaving it in gives one estimator two
            # points in a thirteen-point correlation. Section 2 was fixed and this one
            # was not, so every published r was computed with k=5 double-weighted
            # while the paper stated the duplicate had been excluded. The effect is
            # small -- the floor moves from 0.973 to 0.972 -- and the disagreement
            # between what was claimed and what ran is the defect, not the digit.
            if name in duplicates:
                continue
            folds_here = channels.get((name, dose), [])
            if not folds_here:
                continue
            got = channel_points(folds_here)
            columns_['axis'].append(mean_absorption[name])
            for channel in ('local', 'global', 'aggregate'):
                columns_[channel].append(got[channel])
        frame = pd.DataFrame(columns_).replace([np.inf, -np.inf],
                                              np.nan).dropna()
        print(f"{dose:>6.2f} "
              f"{frame['axis'].corr(frame['local']):>22.3f} "
              f"{frame['axis'].corr(frame['global']):>23.3f} "
              f"{frame['axis'].corr(frame['aggregate']):>26.3f}")

    # The r above is a point over thirteen fold-means; its sampling unit is the
    # fold, so the interval resamples folds and recomputes the thirteen means and
    # the statistic per draw. Lin's concordance is printed beside it because the
    # scatter is drawn against the identity line, and r rewards any line at all.
    from statistical_validation.dependent_bootstrap import (
        moving_block_correlation_ci)
    print("\n   r and Lin's concordance vs the local channel, fold-resampled CIs")
    print(f"{'dose':>6} {'set':>6} {'r':>22} {'Lin':>22}")
    for dose in DOSES:
        names = [n for n, *_ in entries
                 if n not in duplicates and channels.get((n, dose))]
        folds_n = min(len(channels[(n, dose)]) for n in names)
        import numpy as _np
        L = _np.array([[channels[(n, dose)][f].get('local', _np.nan)
                        for f in range(folds_n)] for n in names])
        A = _np.array([[absorptions[n][f] if f < len(absorptions[n]) else _np.nan
                        for f in range(folds_n)] for n in names])
        for label, keep in (('all', [True] * len(names)),
                            ('free', [n not in ('knn_k1', 'ladder_decision_tree')
                                      for n in names])):
            idx = [i for i, k in enumerate(keep) if k]
            got = moving_block_correlation_ci(A[idx], L[idx], block=block)
            (r, (rl, rh)) = got['pearson']
            (c, (cl, ch)) = got['concordance']
            print(f"{dose:>6.2f} {label:>6} "
                  f"{r:>8.3f} [{rl:+.3f},{rh:+.3f}] "
                  f"{c:>8.3f} [{cl:+.3f},{ch:+.3f}]")

    print("\n4. THE ANOMALY: ridge against random forest, by channel")
    print(f"{'dose':>6} {'ridge loc':>10} {'RF loc':>9} {'ridge glo':>10} "
          f"{'RF glo':>9} {'ridge agg':>10} {'RF agg':>9}")
    for dose in DOSES:
        cells = []
        for channel in ('local', 'global', 'aggregate'):
            for name in ('ladder_ridge', 'ladder_random_forest'):
                got = channel_points(channels[(name, dose)])
                cells.append(got[channel])
        print(f"{dose:>6.2f} " + ' '.join(f"{c:>+9.4f}" for c in cells))
    print("\n  If the forest leads on local and the ridge on global, the capacity")
    print("  ordering holds and the aggregate measure is what inverted it.")

    # The docstring of contrast() names this exact comparison as the standard --
    # "the ridge against the forest" -- and until now it was never run: the paper
    # read marginal interval overlap instead, which does not test a difference.
    from statistical_validation.leakage_channels import contrast
    print("\n  paired ridge minus forest, per fold, moving-block bootstrap:")
    for dose in DOSES:
        a = channels.get(('ladder_ridge', dose))
        c = channels.get(('ladder_random_forest', dose))
        if not a or not c:
            continue
        for ch in ('local', 'aggregate'):
            got = contrast(a, c, ch, block=block)
            lo, hi = got['ci95']
            print(f"    dose {dose:>4.2f} {ch:>9}: {got['point']:+.4f} "
                  f"[{lo:+.3f},{hi:+.3f}]  n_pairs={got['n_pairs']}")

    thin = sum(1 for key, value in channels.items()
               for fold in value if fold['thin_handed_partition'])
    print(f"\n  folds with fewer than five handed rows: {thin} "
          f"(a 5% dose on a 64-row window hands over three)")
    audit_resamples()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
