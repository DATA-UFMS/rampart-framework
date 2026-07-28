#!/usr/bin/env python3
"""Does the aggregate measure mislead everywhere, or only where entities spill over?

Every channel result so far comes from one panel: World Bank, 768 rows, 32
entities, nine folds. On it, the aggregate severity measure inverts the capacity
ordering -- the ridge appears more severely affected than the random forest, while
the memorisation channel says the opposite. The question this settles is whether
that is a property of aggregate severity or a property of that panel.

**The moderator, computable before running anything.** The generalisation channel
runs through two routes. One is within-entity: a contaminated row shifts its own
entity's mean, and that reaches the *other* evaluation rows of the same entity. The
other is an entity-independent coefficient shift of order 1/n_train. So the size of
the channel should track how many other evaluation rows a contaminated row can
reach -- the spillover degree:

    World Bank   test_len 2, one row per entity-year  ->  spillover +1.00
    INEP         test_len 1, one row per entity-year  ->  spillover  0.00

A binary contrast, not a gradient, and neither panel was designed for it.

Predictions were registered in `PRE_ESPECIFICACAO_amplificacao.md` §4.2g before
this ran. R1: |global| < 0.05 everywhere on INEP. R2: the aggregate inversion
disappears, so the random forest's aggregate is at least the ridge's at every
dose. R3: the memorisation channel replicates, with absorption predicting it at
r > 0.9.

Run: python scripts/validation/probe_regime.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_harness import (DOSES, fold_rng, folds, panel, prepared,
                           spillover_degree)

from core.models.absorption import absorption_coefficient  # noqa: E402
from core.models.ladder import LADDER, entity_effect_frames  # noqa: E402
from core.scientific_config import RANDOM_SEED  # noqa: E402
from statistical_validation.dependent_bootstrap import (  # noqa: E402
    fold_dependence_span)
from statistical_validation.leakage_channels import (  # noqa: E402
    decompose_fold, summarise)

PANELS = ('worldbank', 'inep_censo')


def run_panel(dataset):
    df, columns, cfg = panel(dataset)
    windows = folds(cfg)
    block = fold_dependence_span(cfg.walk_forward_config)
    spill = spillover_degree(df, cfg)
    print(f"\n{dataset}: {len(df)} rows, {df['entity_id'].nunique()} entities, "
          f"{len(windows)} folds, block {block}, spillover {spill:+.3f}",
          flush=True)

    channels, absorptions, sizes = {}, {}, []
    for fold, (a, b, test_start, test_end) in enumerate(windows):
        made = prepared(df, columns, a, b, test_start, test_end)
        if made is None:
            continue
        (X_train, y_train, e_train, _yr,
         X_test, y_test, e_test, _yr_test) = made
        sizes.append((len(X_train), len(X_test)))
        fit_frame, eval_frame, _m, _g = entity_effect_frames(
            X_train, X_test, y_train, e_train, e_test)
        truth = np.asarray(y_test, dtype=float)

        clean = {}
        for rung in LADDER:
            model = rung.make()
            model.fit(fit_frame, y_train)
            clean[rung.name] = np.asarray(model.predict(eval_frame), dtype=float)
            absorptions.setdefault(rung.name, []).append(
                absorption_coefficient(rung.make, fit_frame, y_train,
                                       eval_frame, y_test,
                                       seed=RANDOM_SEED + fold,
                                       baseline=clean[rung.name])['absorption'])

        for dose in DOSES:
            rng = fold_rng(fold, dose)
            count = max(1, int(round(dose * len(X_test))))
            handed = np.sort(rng.choice(len(X_test), size=count, replace=False))
            echoed = np.sort(rng.choice(len(X_train), size=count, replace=True))
            mask = np.zeros(len(X_test), dtype=bool)
            mask[handed] = True

            leak = entity_effect_frames(
                pd.concat([X_train, X_test.iloc[handed]], ignore_index=True),
                X_test,
                pd.concat([y_train, y_test.iloc[handed]], ignore_index=True),
                pd.concat([e_train, e_test.iloc[handed]], ignore_index=True),
                e_test)
            leak_y = pd.concat([y_train, y_test.iloc[handed]], ignore_index=True)
            control = entity_effect_frames(
                pd.concat([X_train, X_train.iloc[echoed]], ignore_index=True),
                X_test,
                pd.concat([y_train, y_train.iloc[echoed]], ignore_index=True),
                pd.concat([e_train, e_train.iloc[echoed]], ignore_index=True),
                e_test)
            control_y = pd.concat([y_train, y_train.iloc[echoed]],
                                  ignore_index=True)

            for rung in LADDER:
                fitted = {}
                for label, (frames, ya) in (('leak', (leak, leak_y)),
                                            ('control', (control, control_y))):
                    model = rung.make()
                    model.fit(frames[0], ya)
                    fitted[label] = np.asarray(model.predict(frames[1]),
                                               dtype=float)
                channels.setdefault((rung.name, dose), []).append(
                    decompose_fold(truth, clean[rung.name], fitted['leak'],
                                   mask=mask, control=fitted['control']))
        print(f"  fold {fold}: done", flush=True)

    return {
        'spillover': spill, 'block': block,
        'absorption': {n: float(np.nanmean(v)) for n, v in absorptions.items()},
        'channels': channels,
        'n_train': int(np.mean([s[0] for s in sizes])),
        'n_test': int(np.mean([s[1] for s in sizes])),
    }


def main():
    results = {name: run_panel(name) for name in PANELS}

    for name, got in results.items():
        print("\n" + "=" * 84)
        print(f"{name} -- spillover {got['spillover']:+.3f}, "
              f"n_train {got['n_train']}, n_test {got['n_test']}")
        for dose in DOSES:
            print(f"\n  dose {dose:.0%}")
            print(f"{'model':>26} {'absorp':>8} {'local':>9} {'global':>9} "
                  f"{'size':>8} {'aggregate':>10}")
            for rung in LADDER:
                summary = summarise(got['channels'][(rung.name, dose)],
                                    block=got['block'], iters=2000)
                print(f"{rung.name:>26} {got['absorption'][rung.name]:>8.4f} "
                      f"{summary['local']['point']:>+9.4f} "
                      f"{summary['global']['point']:>+9.4f} "
                      f"{summary['sample_size_effect']['point']:>+8.4f} "
                      f"{summary['aggregate']['point']:>+10.4f}")

    def point(name, rung, dose, channel):
        return summarise(results[name]['channels'][(rung, dose)],
                         block=results[name]['block'],
                         iters=2000)[channel]['point']

    print("\n" + "=" * 84)
    print("R1 -- does the global channel shrink where there is no spillover?")
    print(f"{'panel':>12} {'spillover':>10} {'max |global|':>13} {'verdict':>10}")
    for name, got in results.items():
        worst = max(abs(point(name, r.name, d, 'global'))
                    for r in LADDER for d in DOSES)
        verdict = ('< 0.05' if worst < 0.05 else 'large')
        print(f"{name:>12} {got['spillover']:>+10.3f} {worst:>13.4f} "
              f"{verdict:>10}")

    print("\nR2 -- does the aggregate inversion disappear without spillover?")
    print(f"{'panel':>12} " + ' '.join(f"{f'{d:.0%}':>22}" for d in DOSES))
    for name in results:
        cells = []
        for dose in DOSES:
            ridge = point(name, 'ladder_ridge', dose, 'aggregate')
            forest = point(name, 'ladder_random_forest', dose, 'aggregate')
            cells.append(f"RF{forest:+.4f} vs R{ridge:+.4f}"
                         + ('!' if ridge > forest else ' '))
        print(f"{name:>12} " + ' '.join(f"{c:>22}" for c in cells))
    print("   '!' marks an inversion: the ridge scoring as more severely affected")

    print("\nR3 -- does the memorisation channel replicate?")
    print(f"{'panel':>12} " + ' '.join(f"{f'r at {d:.0%}':>12}" for d in DOSES))
    for name in results:
        cells = []
        for dose in DOSES:
            axis = [results[name]['absorption'][r.name] for r in LADDER]
            local = [point(name, r.name, dose, 'local') for r in LADDER]
            frame = pd.DataFrame({'a': axis, 'l': local}).replace(
                [np.inf, -np.inf], np.nan).dropna()
            cells.append(frame['a'].corr(frame['l']))
        print(f"{name:>12} " + ' '.join(f"{c:>+12.3f}" for c in cells))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
