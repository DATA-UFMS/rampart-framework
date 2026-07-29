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
from probe_harness import DOSES, fold_rng, folds, panel, prepared

from core.models.absorption import (  # noqa: E402
    absorption_coefficient, knn_expected_absorption)
from core.models.icl import matched_context  # noqa: E402
from core.models.ladder import LADDER, entity_effect_frames  # noqa: E402
from core.scientific_config import (  # noqa: E402
    RANDOM_SEED, SCIENTIFIC_CONFIG)
from statistical_validation.dependent_bootstrap import (  # noqa: E402
    fold_dependence_span)
from statistical_validation.leakage_channels import (  # noqa: E402
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
            if find_spec(family.package) is not None:
                entries.append((family.name, family.make, 'in-context', None))
    return entries


def main(dataset='worldbank', entity_cap=None):
    df, columns, cfg = panel(dataset)
    if entity_cap is not None:
        keep = sorted(df['entity_id'].unique())[:int(entity_cap)]
        df = df[df['entity_id'].isin(keep)].reset_index(drop=True)
        print(f"subsampled to {len(keep)} entities, {len(df)} rows")
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
    print(f"  absorption read at {probes} probes"
          + (f" x {reps} replicates" if reps > 1 else " x 1 replicate")
          + (f", fraction {PROBE_FRACTION}" if PROBE_FRACTION else "") + "\n")

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

    mean_absorption = {name: float(np.nanmean(values))
                       for name, values in absorptions.items()}

    print("\n" + "=" * 78)
    print("1. IS THE AXIS CALIBRATED? kNN absorption against (2k-1)/k^2")
    print(f"{'config':>12} {'analytic':>10} {'measured':>10} {'gap':>8}")
    worst = 0.0
    for name, _make, kind, expected in entries:
        if kind != 'sweep':
            continue
        measured = mean_absorption[name]
        worst = max(worst, abs(measured - expected))
        print(f"{name:>12} {expected:>10.4f} {measured:>10.4f} "
              f"{measured - expected:>+8.4f}")
    print(f"\n  largest gap {worst:.4f}. The instrument reproduces a value it "
          f"was not fitted to.")

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
            got = summarise(folds_here, block=block, iters=4000)
            print(f"{name:>26} {mean_absorption[name]:>8.4f} "
                  f"{got['local']['point']:>+9.4f} "
                  f"{got['local_excess']['point']:>+9.4f} "
                  f"{got['global']['point']:>+9.4f} "
                  f"{got['sample_size_effect']['point']:>+8.4f} "
                  f"{got['aggregate']['point']:>+9.4f}")

    print("\n" + "=" * 78)
    print("3. WHICH CHANNEL DOES ABSORPTION EXPLAIN?")
    print(f"{'dose':>6} {'r(absorption, local)':>22} "
          f"{'r(absorption, global)':>23} {'r(absorption, aggregate)':>26}")
    for dose in DOSES:
        columns_ = {'local': [], 'global': [], 'aggregate': [], 'axis': []}
        for name, _make, _kind, _expected in entries:
            folds_here = channels.get((name, dose), [])
            if not folds_here:
                continue
            got = summarise(folds_here, block=block, iters=1000)
            columns_['axis'].append(mean_absorption[name])
            for channel in ('local', 'global', 'aggregate'):
                columns_[channel].append(got[channel]['point'])
        frame = pd.DataFrame(columns_).replace([np.inf, -np.inf],
                                              np.nan).dropna()
        print(f"{dose:>6.2f} "
              f"{frame['axis'].corr(frame['local']):>22.3f} "
              f"{frame['axis'].corr(frame['global']):>23.3f} "
              f"{frame['axis'].corr(frame['aggregate']):>26.3f}")

    print("\n4. THE ANOMALY: ridge against random forest, by channel")
    print(f"{'dose':>6} {'ridge loc':>10} {'RF loc':>9} {'ridge glo':>10} "
          f"{'RF glo':>9} {'ridge agg':>10} {'RF agg':>9}")
    for dose in DOSES:
        cells = []
        for channel in ('local', 'global', 'aggregate'):
            for name in ('ladder_ridge', 'ladder_random_forest'):
                got = summarise(channels[(name, dose)], block=block, iters=2000)
                cells.append(got[channel]['point'])
        print(f"{dose:>6.2f} " + ' '.join(f"{c:>+9.4f}" for c in cells))
    print("\n  If the forest leads on local and the ridge on global, the capacity")
    print("  ordering holds and the aggregate measure is what inverted it.")

    thin = sum(1 for key, value in channels.items()
               for fold in value if fold['thin_handed_partition'])
    print(f"\n  folds with fewer than five handed rows: {thin} "
          f"(a 5% dose on a 64-row window hands over three)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
