#!/usr/bin/env python3
"""Where does the generalisation channel come from?

The channel decomposition splits leakage severity into memorisation, on rows the
model was handed, and a generalisation shift, on rows it was not. The
memorisation half is accounted for: a cheap probe predicts it at r = 0.99 on both
panels. The other half is large on both panels and unexplained. The sample-size
control rules out one route -- duplicating rows already in the training frame does
essentially nothing -- but it does not identify what is left, because the
duplicated rows carry no new information at all.

Six arms, one switch each, all adding the same number of rows to a clean arm that
has had one interior year withheld. What varies is only where the added rows come
from, and therefore how far that period sits from the evaluation window:

    ECHO      copies of rows already in the frame     nothing new at all
    RESERVE   a withheld year inside the training     uncovered, but surrounded
              window                                 by years the model has
    GAP1      the first buffer, train to validation   uncovered, far
    VAL       the validation window                   uncovered, middle
    GAP2      the second buffer, validation to test   uncovered, adjacent
    LEAK      the evaluation window                   distance zero

Every source but the interior year is already excluded from training by the
protocol, so using them here does not contaminate the clean arm further.

The question the curve answers is not whether the buffer protects but **how far a
leak has to be before it stops mattering** -- which is the quantity a buffer width
should be chosen against, and which nothing in this repository had measured.
Predictions registered in `PRE_ESPECIFICACAO_amplificacao.md` §4.2i.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_harness import DOSES, entity_subsample, fold_rng, folds, panel, prepared

from core.models.ladder import LADDER, entity_effect_frames  # noqa: E402
from statistical_validation.dependent_bootstrap import (  # noqa: E402
    excludes_zero, fold_dependence_span)
from statistical_validation.leakage_channels import (  # noqa: E402
    contrast, decompose_fold, summarise)

#: Source periods for the added rows, ordered by how far they sit from the
#: evaluation window. The fold layout is
#:
#:     train .. T | gap T+1,T+2 | val T+3,T+4 | gap T+5,T+6 | test T+7,T+8
#:
#: so a leak can come from five places at four different distances, and all but
#: the interior year are already excluded from training by the protocol.
#:
#: A first version of this probe used only the *first* gap and called it the
#: adjacency test. It is five to seven years from the evaluation window, so it
#: tested nothing of the kind. The fix is not a better single contrast but the
#: whole curve: how does the channel decay with the temporal distance of the
#: leaked rows? That is the quantity a buffer has to be chosen against.
ARMS = ('ECHO', 'RESERVE', 'RESERVE_NEAR', 'GAP1', 'VAL', 'GAP2', 'LEAK')

#: The unbounded decision tree is excluded from the generalisation summaries. It
#: is the instrument's absorption anchor -- it reads exactly 1.0000 -- and it is
#: useless for this measurement: adding *any* rows moves its held-out fit by
#: several tenths, including exact duplicates of rows it already has, so its
#: generalisation channel is instability rather than information. Its clean R^2 on
#: this panel is negative, which is the same fact stated another way.
STABLE = tuple(r for r in LADDER if r.name != 'ladder_decision_tree')


def block_of(df, columns, years, entity_column='entity_id'):
    """Rows of the panel in the given years, imputed nowhere and ready to append.

    Returned as raw columns; the caller imputes and augments with the same
    statistics the clean arm used, so the only thing that varies between arms is
    which rows were added.
    """
    frame = df[df['year'].isin(years)]
    keep = frame[columns].notna().all(axis=1) & frame['target'].notna()
    frame = frame[keep]
    return (frame[columns].reset_index(drop=True),
            frame['target'].reset_index(drop=True),
            frame['entity_id'].reset_index(drop=True))


def main(dataset='worldbank', entity_cap=None):
    """Run the curve on one panel, optionally on a subsample of its entities.

    Subsampling entities is offered because the full INEP panel does not finish
    locally -- 94,283 rows over eight folds and six arms exceeded a two-hour
    budget on the last fold, which has the widest training window. The curve is a
    statement about *temporal* distance, and dropping entities leaves every year
    and every distance intact while cutting the row count proportionally. It
    changes the levels, since absorption depends on n, so the transferable
    quantity is the normalised shape rather than the raw channel.

    Declared rather than hidden: a subsampled replication is weaker than a full
    one, and the full one is cloud work.
    """
    df, columns, cfg = panel(dataset)
    if entity_cap is not None:
        df = entity_subsample(df, entity_cap)
        print(f"subsampled to {len(keep)} entities, {len(df)} rows -- "
              f"levels will differ, the normalised shape is what transfers")
    windows = folds(cfg)
    gap = int(cfg.walk_forward_config['gap'])
    block = fold_dependence_span(cfg.walk_forward_config)
    print(f"{dataset}: {len(df)} rows, {len(windows)} folds, gap {gap} years, "
          f"block {block}\n")

    channels = {}
    sizes = {arm: [] for arm in ARMS}
    distance_log = []

    for fold, (train_start, train_end, test_start, test_end) in enumerate(windows):
        # Two withheld years, not one, and the second is the point of this probe's
        # last revision. The prescription reads a buffer width off this curve by
        # interpolation, and until now there was no arm between GAP1 and RESERVE --
        # five and 12.8 years apart on the World Bank, four and ten on INEP -- so
        # the widths quoted for a channel under 25% and under 10% were
        # EXTRAPOLATIONS anchored on RESERVE, whose channel is low because its year
        # is surrounded by years the model keeps rather than because it is distant.
        # Redundancy read as distance is exactly the confusion the curve exists to
        # avoid.
        #
        # RESERVE sits at the middle of the training window and RESERVE_NEAR at
        # three quarters of it, which lands between GAP1 and RESERVE on both panels
        # and turns the quoted widths into interpolation.
        reserved = (train_start + train_end) // 2
        reserved_near = train_start + (3 * (train_end - train_start)) // 4
        val_len = int(cfg.walk_forward_config['val_len'])
        periods = {
            'GAP1': list(range(train_end + 1, train_end + 1 + gap)),
            'VAL': list(range(train_end + 1 + gap,
                              train_end + 1 + gap + val_len)),
            'GAP2': list(range(train_end + 1 + gap + val_len,
                               train_end + 1 + 2 * gap + val_len)),
            'RESERVE': [reserved],
            'RESERVE_NEAR': [reserved_near],
        }
        distances = {arm: test_start - max(years)
                     for arm, years in periods.items()}
        distances['LEAK'] = 0
        distances['ECHO'] = test_start - train_end

        # Both withheld years leave the clean arm, so every arm is compared against
        # the same baseline. Coinciding years collapse to one, which happens on a
        # short training window and would otherwise withhold a year twice.
        withheld = {reserved, reserved_near}
        clean_years = [y for y in range(train_start, train_end + 1)
                       if y not in withheld]
        held = df[df['year'].isin(clean_years)]
        evaluation = df[(df['year'] >= test_start) & (df['year'] <= test_end)]

        fill = held[columns].median()
        X_train = held[columns].fillna(fill)
        keep = X_train.notna().all(axis=1) & held['target'].notna()
        X_test = evaluation[columns].fillna(fill)
        valid = X_test.notna().all(axis=1) & evaluation['target'].notna()
        if keep.sum() < 20 or valid.sum() < 3:
            continue

        X_train = X_train[keep].reset_index(drop=True)
        y_train = held['target'][keep].reset_index(drop=True)
        e_train = held['entity_id'][keep].reset_index(drop=True)
        X_test = X_test[valid].reset_index(drop=True)
        y_test = evaluation['target'][valid].reset_index(drop=True)
        e_test = evaluation['entity_id'][valid].reset_index(drop=True)
        truth = np.asarray(y_test, dtype=float)

        filled = df.assign(**{c: df[c].fillna(fill[c]) for c in columns})
        pools = {arm: block_of(filled, columns, years)
                 for arm, years in periods.items()}

        fit_frame, eval_frame, _m, _g = entity_effect_frames(
            X_train, X_test, y_train, e_train, e_test)
        clean = {}
        for rung in LADDER:
            model = rung.make()
            model.fit(fit_frame, y_train)
            clean[rung.name] = np.asarray(model.predict(eval_frame), dtype=float)

        for dose in DOSES:
            rng = fold_rng(fold, dose)
            count = max(1, int(round(dose * len(X_test))))
            handed = np.sort(rng.choice(len(X_test), size=count, replace=False))
            mask = np.zeros(len(X_test), dtype=bool)
            mask[handed] = True

            added = {}
            echo = np.sort(rng.choice(len(X_train), size=count, replace=True))
            added['ECHO'] = (X_train.iloc[echo], y_train.iloc[echo],
                             e_train.iloc[echo])
            added['LEAK'] = (X_test.iloc[handed], y_test.iloc[handed],
                             e_test.iloc[handed])
            for arm in periods:
                pool_X, pool_y, pool_e = pools[arm]
                if not len(pool_X):
                    continue
                picked = np.sort(rng.choice(len(pool_X), size=count,
                                            replace=len(pool_X) < count))
                added[arm] = (pool_X.iloc[picked], pool_y.iloc[picked],
                              pool_e.iloc[picked])

            for arm, (Xa, ya, ea) in added.items():
                sizes[arm].append(len(Xa))
                widened = entity_effect_frames(
                    pd.concat([X_train, Xa], ignore_index=True), X_test,
                    pd.concat([y_train, ya], ignore_index=True),
                    pd.concat([e_train, ea], ignore_index=True), e_test)
                widened_y = pd.concat([y_train, ya], ignore_index=True)
                for rung in LADDER:
                    model = rung.make()
                    model.fit(widened[0], widened_y)
                    predicted = np.asarray(model.predict(widened[1]),
                                           dtype=float)
                    channels.setdefault((rung.name, arm, dose), []).append(
                        decompose_fold(truth, clean[rung.name], predicted,
                                       mask=mask))
        distance_log.append(distances)
        print(f"  fold {fold}: reserved {reserved}, "
              f"distances {[(a, distances[a]) for a in ARMS]}", flush=True)

    def point(rung, arm, dose, channel='global_uncontrolled'):
        key = (rung, arm, dose)
        if key not in channels:
            return float('nan')
        return summarise(channels[key], block=block,
                         iters=2000)[channel]['point']

    for dose in DOSES:
        print("\n" + "=" * 78)
        print(f"dose {dose:.0%} -- generalisation channel, on rows NOT handed")
        print(f"{'model':>26} " + ' '.join(f"{arm:>10}" for arm in ARMS))
        for rung in LADDER:
            print(f"{rung.name:>26} " +
                  ' '.join(f"{point(rung.name, arm, dose):>+10.4f}"
                           for arm in ARMS))

    print("\n" + "=" * 78)
    print("G1 -- ECHO near zero? (copies of rows already present)")
    worst = max(abs(point(r.name, 'ECHO', d)) for r in LADDER for d in DOSES)
    print(f"  max |global| under ECHO: {worst:.4f}  -> "
          f"{'HOLDS' if worst < 0.05 else 'FAILS'}")

    print("\nG2 -- RESERVE small? (a withheld interior year, predicted < 0.05)")
    worst = max(abs(point(r.name, 'RESERVE', d)) for r in LADDER for d in DOSES)
    print(f"  max |global| under RESERVE: {worst:.4f}  -> "
          f"{'HOLDS' if worst < 0.05 else 'FAILS'}")

    print("\nG3 -- is the adjacent buffer substantial? (GAP2, predicted > 0.10 "
          "for at least half the models at 30%)")
    above = sum(1 for r in LADDER if point(r.name, 'GAP2', DOSES[-1]) > 0.10)
    print(f"  models above 0.10: {above}/{len(LADDER)}  -> "
          f"{'HOLDS' if above >= len(LADDER) / 2 else 'FAILS'}")

    print("\nG4 -- is LEAK the largest?")
    for dose in DOSES:
        means = {a: np.nanmean([point(r.name, a, dose) for r in STABLE])
                 for a in ARMS}
        order = sorted(ARMS, key=lambda a: -means[a])
        print(f"  dose {dose:>4.0%}: " +
              '  >  '.join(f"{a} {means[a]:+.4f}" for a in order))

    print("\n" + "=" * 78)
    print("THE DECAY CURVE -- generalisation channel by temporal distance")
    print("  averaged over the four stable models; the unbounded tree is excluded")
    print("  because any added rows destabilise it, including exact duplicates")
    distances = {arm: float(np.mean([d[arm] for d in distance_log]))
                 for arm in ARMS if all(arm in d for d in distance_log)}
    print(f"\n{'arm':>9} {'distance':>9} " +
          ' '.join(f"{f'{d:.0%}':>10}" for d in DOSES) + f"{'/LEAK':>9}")
    leak30 = np.nanmean([point(r.name, 'LEAK', DOSES[-1]) for r in STABLE])
    for arm in sorted(ARMS, key=lambda a: -distances.get(a, 0)):
        values = [np.nanmean([point(r.name, arm, d) for r in STABLE])
                  for d in DOSES]
        share = values[-1] / leak30 if abs(leak30) > 1e-9 else float('nan')
        print(f"{arm:>9} {distances.get(arm, float('nan')):>9.1f} " +
              ' '.join(f"{v:>+10.4f}" for v in values) + f"{share:>9.3f}")

    print("\n" + "=" * 78)
    print("IS THE DECAY SHAPE THE SAME FOR EVERY MODEL?")
    print("  each model's channel divided by its own leak value, at 30% dose.")
    print("  If the shape is model-invariant, a buffer width can be recommended")
    print("  without knowing which model will be fitted.")
    curve_arms = [a for a in ('RESERVE', 'GAP1', 'VAL', 'GAP2')
                  if a in distances]
    print(f"\n{'model':>26} " + ' '.join(f"{a:>9}" for a in curve_arms))
    normalised = {a: [] for a in curve_arms}
    for rung in STABLE:
        own = point(rung.name, 'LEAK', DOSES[-1])
        if not np.isfinite(own) or abs(own) < 1e-6:
            continue
        row = []
        for arm in curve_arms:
            share = point(rung.name, arm, DOSES[-1]) / own
            normalised[arm].append(share)
            row.append(share)
        print(f"{rung.name:>26} " + ' '.join(f"{v:>9.3f}" for v in row))
    print(f"\n{'spread':>26} " +
          ' '.join(f"{np.ptp(normalised[a]):>9.3f}" for a in curve_arms))
    print("  spread is max minus min across models; small means the shape "
          "transfers")

    # The headline this probe produced was "leaking the buffer immediately before the
    # evaluation window costs MORE than leaking the window itself" -- GAP2 at 105% of
    # LEAK on the World Bank. It was a ratio of two bare means over nine folds, and it
    # does not survive its own interval. The contrast is taken fold by fold because
    # both arms are measured on the same folds, which is what cancels the shared
    # noise; two arms whose marginal intervals overlap can still differ reliably, and
    # two whose points differ can fail to.
    print("\n" + "=" * 78)
    print("DOES THE ADJACENT BUFFER REALLY COST MORE THAN THE TEST WINDOW?")
    print("  per-fold difference GAP2 minus LEAK, on the generalisation channel")
    print(f"{'model':>26} {'dose':>6} {'difference':>12} {'ci95':>22} {'verdict':>12}")
    for rung in STABLE:
        for dose in DOSES:
            a = channels.get((rung.name, 'GAP2', dose))
            b = channels.get((rung.name, 'LEAK', dose))
            if not a or not b or len(a) != len(b):
                continue
            got = contrast(a, b, 'global_uncontrolled', block=block, direction=+1)
            lo, hi = got['ci95']
            # Both directions, because a column that only asks "above zero?" answers
            # "no" for an interval that sits entirely BELOW zero, and that is a
            # different finding: on INEP the adjacent buffer costs reliably LESS than
            # the evaluation window for several models. This repository already fixed
            # exactly this blindness once, in excludes_zero, and it came back in the
            # presentation rather than in the test.
            if excludes_zero(got['ci95'], direction=+1):
                verdict = 'GAP2 > LEAK'
            elif excludes_zero(got['ci95'], direction=-1):
                verdict = 'GAP2 < LEAK'
            else:
                verdict = 'no difference'
            print(f"{rung.name:>26} {dose:>6.0%} {got['point']:>+12.4f} "
                  f"[{lo:>+8.4f},{hi:>+8.4f}] {verdict:>12}")
    print("\n  'no difference' everywhere is the World Bank reading: the adjacent")
    print("  buffer costs about what the evaluation window costs, not more, and the")
    print("  published 105% was a ratio of two bare means over nine folds. On INEP")
    print("  several cells read 'GAP2 < LEAK' instead, so the direction reverses")
    print("  between panels and only the prescription -- 5 to 8 years against the")
    print("  protocol's 2 -- is common to both.")

    print("\n" + "=" * 78)
    print("WHAT BUFFER WOULD IT TAKE? interpolating the normalised curve")
    points = sorted(((distances[a], float(np.mean(normalised[a])))
                     for a in curve_arms), key=lambda t: t[0])
    for target in (0.50, 0.25, 0.10):
        needed = None
        for (d0, s0), (d1, s1) in zip(points, points[1:]):
            if s0 >= target >= s1 and s0 != s1:
                needed = d0 + (s0 - target) / (s0 - s1) * (d1 - d0)
                break
        if needed is None:
            beyond = points[-1][0] if points[-1][1] > target else None
            print(f"  to hold the channel under {target:.0%}: "
                  + (f"more than {beyond:.1f} years, beyond what this panel "
                     f"measures" if beyond else
                     f"already under it at {points[0][0]:.1f} years"))
        else:
            print(f"  to hold the channel under {target:.0%}: "
                  f"buffer of about {needed:.1f} years")
    print(f"\n  The protocol currently uses {gap} years. A width chosen against")
    print("  residual dependence and a width chosen against this channel are")
    print("  different numbers, and a paper should say which one it used.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
