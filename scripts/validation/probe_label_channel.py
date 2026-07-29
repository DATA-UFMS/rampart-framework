#!/usr/bin/env python3
"""Is leakage severity set by where a statistic was fitted, or by how much label
information it carries?

Roth classifies leakage by mechanism -- where the statistic was fitted relative to
the split -- and that classification produces an exception he marks himself:
target encoding *"has Class-I mechanism but Class-II-magnitude inflation"*, so he
assigns it by behaviour and flags *"a partial exception to the mechanism-first
rule"*.

This tests a different axis. Not where the statistic was fitted, but how much of
the evaluation labels reach the model through it, attenuated by the width of the
aggregation. A statistic averaging `m` training rows into which one evaluation row
falls carries that row's label at weight `1/(m+1)`. On that axis his published
landscape is monotone with no exception:

    ordinal encoding        weight 0        d_z = +0.01
    global scaler           weight ~1/n     d_z = -0.02
    target encoding         weight 1/(m+1)  d_z = +0.46
    duplication (10%, RF)   weight 1        d_z = +0.90
    duplication (30%, DT)   weight 1        d_z = +1.38

Three arms, same folds, same models:

  **C1-F, feature statistics.** The scaler is fitted on the training window
  widened with evaluation rows. No label crosses. On this panel the imputation is
  inert -- zero missing cells, measured -- so this isolates the scaler.

  **C1-L, label-conditioned statistic.** The entity effect is the mean of the
  outcome per entity, which is a target encoding. It is computed including
  evaluation targets. No row enters the training frame; only label information
  does, attenuated by the number of training years per entity.

  **C3, duplication.** Evaluation rows with their labels in the training frame.
  Weight one. Already measured; included here so the three sit on one scale.

Predictions were pre-registered in `PRE_ESPECIFICACAO_amplificacao.md` §4.2e
before this ran: P1 no memorisation channel for C1-F, P2 trees near-immune to
C1-F, P3 C1-L small and positive at the attenuation the width implies, P4 the
ordering between families reverses between C1-F and C3.
"""

import sys
import warnings
from importlib.util import find_spec
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_harness import DOSES, fold_rng, folds, panel, prepared

from core.models.absorption import absorption_coefficient  # noqa: E402
from core.models.ladder import LADDER, entity_effect_frames  # noqa: E402
from core.scientific_config import (  # noqa: E402
    RANDOM_SEED, SCIENTIFIC_CONFIG)
from core.validation import scale_from_training_window  # noqa: E402
from statistical_validation.dependent_bootstrap import (  # noqa: E402
    fold_dependence_span)
from statistical_validation.leakage_channels import (  # noqa: E402
    decompose_fold, summarise)

ARMS = ('C1-F', 'C1-L', 'C3')


def candidates():
    entries = [(rung.name, rung.make, 'classical') for rung in LADDER]
    if find_spec('tabpfn') is not None or find_spec('tabicl') is not None:
        from core.models.icl import FAMILIES
        for family in FAMILIES:
            if find_spec(family.package) is not None:
                entries.append((family.name, family.make, 'in-context'))
    return entries


def scaled(fit_frame, eval_frame, *, widen_with=None):
    """Standardise, optionally fitting the statistics on a widened frame.

    The pipeline's own primitive, so the probe contaminates what the pipeline
    contaminates rather than an imitation of it.
    """
    frames, _report = scale_from_training_window(
        fit_frame, eval_frame,
        fit_on=(None if widen_with is None
                else pd.concat([fit_frame, widen_with], ignore_index=True)))
    return frames[0], frames[1]


def main(dataset='worldbank', entity_cap=None):
    df, columns, cfg = panel(dataset)
    if entity_cap is not None:
        keep = sorted(df['entity_id'].unique())[:int(entity_cap)]
        df = df[df['entity_id'].isin(keep)].reset_index(drop=True)
        print(f"subsampled to {len(keep)} entities, {len(df)} rows -- the full "
              f"panel does not finish locally; levels differ, the attenuation "
              f"ratio is what transfers")
    windows = folds(cfg)
    entries = candidates()
    block = fold_dependence_span(cfg.walk_forward_config)
    print(f"World Bank: {len(windows)} folds, {len(entries)} models, "
          f"block {block}")
    print(f"arms: {', '.join(ARMS)}\n")

    channels = {}
    absorptions = {}
    widths = []

    for fold, (a, b, test_start, test_end) in enumerate(windows):
        made = prepared(df, columns, a, b, test_start, test_end)
        if made is None:
            continue
        (X_train, y_train, e_train, yr_train,
         X_test, y_test, e_test, yr_test) = made
        truth = np.asarray(y_test, dtype=float)

        # How many training years each entity has: the attenuation width that
        # sets the label weight of the target encoding, 1/(m+1).
        width = float(e_train.value_counts().mean())
        widths.append(width)

        clean_fit, clean_eval, _m, _g = entity_effect_frames(
            X_train, X_test, y_train, e_train, e_test)
        clean_fit_s, clean_eval_s = scaled(clean_fit, clean_eval)

        clean_predictions = {}
        for name, make, _kind in entries:
            model = make()
            model.fit(clean_fit_s, y_train)
            clean_predictions[name] = np.asarray(
                model.predict(clean_eval_s), dtype=float)
            absorptions.setdefault(name, []).append(absorption_coefficient(
                make, clean_fit_s, y_train, clean_eval_s, y_test,
                seed=RANDOM_SEED + fold,
                baseline=clean_predictions[name])['absorption'])

        for dose in DOSES:
            rng = fold_rng(fold, dose)
            count = max(1, int(round(dose * len(X_test))))
            handed = np.sort(rng.choice(len(X_test), size=count, replace=False))
            mask = np.zeros(len(X_test), dtype=bool)
            mask[handed] = True

            # C1-F: the scaler sees the evaluation rows' features. No label.
            f_fit, f_eval = scaled(clean_fit, clean_eval,
                                   widen_with=clean_eval.iloc[handed])
            # C1-L: the entity means see the evaluation targets. No row moves.
            l_fit, l_eval, _m, _g = entity_effect_frames(
                X_train, X_test, y_train, e_train, e_test,
                contaminate_with=(y_test.iloc[handed], e_test.iloc[handed]))
            l_fit, l_eval = scaled(l_fit, l_eval)
            # C3: the rows themselves, with their labels, in the training frame.
            c3_fit, c3_eval, _m, _g = entity_effect_frames(
                pd.concat([X_train, X_test.iloc[handed]], ignore_index=True),
                X_test,
                pd.concat([y_train, y_test.iloc[handed]], ignore_index=True),
                pd.concat([e_train, e_test.iloc[handed]], ignore_index=True),
                e_test)
            c3_fit, c3_eval = scaled(c3_fit, c3_eval)
            c3_y = pd.concat([y_train, y_test.iloc[handed]], ignore_index=True)

            arms = {'C1-F': (f_fit, y_train, f_eval),
                    'C1-L': (l_fit, y_train, l_eval),
                    'C3': (c3_fit, c3_y, c3_eval)}

            arm_years = {'C1-F': yr_train, 'C1-L': yr_train,
                         'C3': pd.concat([yr_train, yr_test.iloc[handed]],
                                         ignore_index=True)}
            for name, make, _kind in entries:
                for arm, (Xa, ya, eval_a) in arms.items():
                    model = make()
                    model.fit(Xa, ya)
                    predicted = np.asarray(model.predict(eval_a), dtype=float)
                    channels.setdefault((name, arm, dose), []).append(
                        decompose_fold(truth, clean_predictions[name],
                                       predicted, mask=mask))
        print(f"  fold {fold}: done", flush=True)

    mean_absorption = {n: float(np.nanmean(v)) for n, v in absorptions.items()}
    width = float(np.mean(widths))
    print(f"\naggregation width of the target encoding: m = {width:.1f} training "
          f"rows per entity")
    print(f"implied label weight for C1-L: 1/(m+1) = {1.0/(width+1):.4f}")

    def cell(name, arm, dose, channel):
        got = summarise(channels[(name, arm, dose)], block=block, iters=2000)
        return got[channel]['point']

    for dose in DOSES:
        print("\n" + "=" * 82)
        print(f"dose {dose:.0%} -- local channel (memorisation) by arm")
        print(f"{'model':>26} {'absorp':>8} " +
              ' '.join(f"{arm:>10}" for arm in ARMS) + f"{'C1-L/C3':>10}")
        for name, _make, _kind in entries:
            values = [cell(name, arm, dose, 'local') for arm in ARMS]
            ratio = (values[1] / values[2] if abs(values[2]) > 1e-6
                     else float('nan'))
            print(f"{name:>26} {mean_absorption[name]:>8.4f} " +
                  ' '.join(f"{v:>+10.4f}" for v in values) +
                  f"{ratio:>10.3f}")

    print("\n" + "=" * 82)
    print("P1 -- does C1-F open a memorisation channel? (predicted |local| < 0.05)")
    worst = max(abs(cell(n, 'C1-F', d, 'local'))
                for n, _mk, _k in entries for d in DOSES)
    print(f"  largest |local| under C1-F across models and doses: {worst:.4f}"
          f"  -> {'HOLDS' if worst < 0.05 else 'FAILS'}")

    print("\nP2 -- are trees near-immune to C1-F? (predicted |aggregate| < 0.02)")
    trees = ('ladder_decision_tree', 'ladder_random_forest',
             'ladder_gradient_boosting')
    for name in trees:
        values = [cell(name, 'C1-F', d, 'aggregate') for d in DOSES]
        print(f"  {name:>26}: " + ' '.join(f"{v:>+8.4f}" for v in values) +
              f"  -> {'HOLDS' if max(abs(v) for v in values) < 0.02 else 'FAILS'}")
    knn = [cell('ladder_knn', 'C1-F', d, 'aggregate') for d in DOSES]
    print(f"  {'ladder_knn (scale-sensitive)':>26}: " +
          ' '.join(f"{v:>+8.4f}" for v in knn))

    print("\nP3 -- is C1-L intermediate, near the attenuation the width implies?")
    print(f"  predicted weight 1/(m+1) = {1.0/(width+1):.4f}")
    for dose in DOSES:
        ratios = [cell(n, 'C1-L', dose, 'local') / cell(n, 'C3', dose, 'local')
                  for n, _mk, _k in entries
                  if abs(cell(n, 'C3', dose, 'local')) > 1e-6]
        ratios = [r for r in ratios if np.isfinite(r)]
        if ratios:
            print(f"  dose {dose:>4.0%}: median C1-L/C3 local = "
                  f"{np.median(ratios):+.4f}  (n={len(ratios)} models)")

    print("\nP4 -- does the ordering between families reverse between C1-F and C3?")
    for dose in DOSES:
        f_values = [cell(n, 'C1-F', dose, 'aggregate') for n, _mk, _k in entries]
        c_values = [cell(n, 'C3', dose, 'local') for n, _mk, _k in entries]
        rho = pd.Series(f_values).corr(pd.Series(c_values), method='spearman')
        print(f"  dose {dose:>4.0%}: Spearman(C1-F aggregate, C3 local) = "
              f"{rho:+.3f}  -> {'reversed or unrelated' if rho < 0.3 else 'same order'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
