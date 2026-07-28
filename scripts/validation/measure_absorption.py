#!/usr/bin/env python3
"""How much of a handed answer does each model keep, in-context ones included?

Two things at once, and the second is the reason this is in the repository rather
than in a notebook.

**An axis for the comparison.** The capacity ladder was ranked by the class III
severities Roth reports. Measured on this panel that rank does not predict
inflation -- correlations of -0.72, -0.55 and +0.04 across the three doses -- so
the borrowed axis fails and the ladder needs one of its own. Absorption is
measured inside the same setup as the inflation it is meant to explain, so no
analogy between a classifier's AUC and a regressor's R^2 is required.

**The mechanism the study argues for, as a measurement.** The claim is that an
in-context learner has no fitting step to regularise, so a contaminated row
survives instead of being shrunk away. That is a statement about absorption, and
absorption is computable for an in-context model exactly as for a ridge: put the
row in the context and see whether the answer comes back. Measured, the claim
either holds or it does not, independently of the inflation experiment -- which
matters, because a mechanism supported only by the outcome it was invented to
explain is not supported.

Run with the optional extra installed for the in-context rows; without it the
classical rungs still report and the in-context ones are skipped.
"""

import sys
import warnings
from importlib.util import find_spec
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

# The harness first: importing it puts src/ on the path.
from probe_harness import folds, panel, prepared

from core.models.absorption import absorption_coefficient  # noqa: E402
from core.models.ladder import LADDER, entity_effect_frames  # noqa: E402
from core.scientific_config import RANDOM_SEED  # noqa: E402


def candidates():
    """Every model to measure: the five rungs, then whatever ICL is installed.

    The optional families are added by availability rather than skipped by
    exception, so a run without the extra says so in its table instead of
    failing halfway through the ninth fold.
    """
    entries = [(rung.name, rung.make, rung.roth_severity, 'classical')
               for rung in LADDER]
    if find_spec('tabpfn') is None and find_spec('tabicl') is None:
        return entries, False
    from core.models.icl import FAMILIES
    for family in FAMILIES:
        if find_spec(family.package) is not None:
            entries.append((family.name, family.make, None, 'in-context'))
    return entries, True


def main():
    df, columns, cfg = panel()
    windows = folds(cfg)
    entries, has_icl = candidates()
    print(f"World Bank: {len(df)} rows, {len(columns)} features, "
          f"{len(windows)} folds")
    print(f"models: {len(entries)}"
          + ('' if has_icl else '  (no in-context extra installed)') + "\n")

    rows = []
    for fold, (train_start, train_end, test_start, test_end) in enumerate(windows):
        made = prepared(df, columns, train_start, train_end, test_start, test_end)
        if made is None:
            continue
        X_train, y_train, entities_train, _years, X_test, y_test, entities_test = made
        # The entity effect is joined once, from the clean training window, and
        # held fixed. Recomputing it with the probe row included would let the
        # answer travel through a feature instead of through the model.
        fit_frame, eval_frame, _means, _global = entity_effect_frames(
            X_train, X_test, y_train, entities_train, entities_test)

        for name, make, roth, kind in entries:
            measured = absorption_coefficient(
                make, fit_frame, y_train, eval_frame, y_test,
                seed=RANDOM_SEED + fold)
            rows.append({'fold': fold, 'model': name, 'kind': kind,
                         'roth': roth, 'absorption': measured['absorption'],
                         'within_fold_sd': measured.get('absorption_sd'),
                         'probes': measured['probes_used'],
                         'skipped': measured['probes_skipped']})
        print(f"  fold {fold}: done", flush=True)

    out = pd.DataFrame(rows)
    print("\nabsorption -- how much of one handed answer the model keeps")
    print(f"{'model':>26} {'kind':>11} {'Roth':>6} {'mean':>8} "
          f"{'sd in fold':>11} {'sd of folds':>12}")
    for name, _make, roth, kind in entries:
        sub = out[out['model'] == name]
        label = f"{roth:.2f}" if roth is not None else '  --'
        print(f"{name:>26} {kind:>11} {label:>6} "
              f"{sub['absorption'].mean():>8.4f} "
              f"{sub['within_fold_sd'].mean():>11.4f} "
              f"{sub['absorption'].std():>12.4f}")

    print("\n  Above one means the prediction overshoots its own truth, which a\n"
          "  sequential residual fitter can do; the ratio stops reading as a\n"
          "  fraction there and is reported as measured rather than clipped.")

    tree = out[out['model'] == 'ladder_decision_tree']['absorption']
    if len(tree):
        print(f"\n  instrument check -- an unbounded decision tree should keep a\n"
              f"  training row exactly: {tree.mean():.4f} "
              f"(sd across folds {tree.std():.4f})")

    if has_icl:
        classical = (out[out['kind'] == 'classical']
                     .groupby('model', sort=False)['absorption'].mean())
        print("\n  where the in-context models sit against the classical rungs")
        for name in out[out['kind'] == 'in-context']['model'].unique():
            value = out[out['model'] == name]['absorption'].mean()
            above = int((classical < value).sum())
            bracketing = classical[(classical - value).abs() < 0.3]
            print(f"    {name}: {value:.4f}, above {above}/{len(classical)} rungs")
            if len(bracketing):
                print(f"      comparable absorption: {', '.join(bracketing.index)}")
        print("\n  Rungs of comparable absorption are what makes the comparison an\n"
              "  interpolation rather than an extrapolation: the question becomes\n"
              "  whether the in-context model inflates more than a classical model\n"
              "  that keeps just as much of a handed answer.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
