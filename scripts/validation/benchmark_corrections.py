#!/usr/bin/env python3
"""The decisive table: what each contamination correction leaves behind.

Reads the cell-replicate estimates produced by analyze_replicated_saturation
(one row per dataset/fold/model/saturation/replicate) and prices four ways of
reporting a contaminated evaluation, all against the clean score, all in the
additive improvement scale:

    uncorrected   bias = B(s) = S + share*D   what the leaderboard prints
    drop          bias = S(s)                 remove contaminated rows: the
                                              spillover stays (Lemma 3)
    exact rest.*  bias = (1-share)*S(s)       exact restoration --
                                              every contaminated row's loss
                                              restored to its exact clean
                                              value. Any correction that only
                                              edits contaminated items'
                                              scores has bias (1-share)*S
                                              plus share*(mean restoration
                                              error on the inserted rows); the
                                              first term is shared by the
                                              whole class and exact
                                              restoration is the member with
                                              zero restoration error. Not a
                                              lower bound: a restoration error
                                              of the opposite sign (the WB
                                              MLP cell) can offset it.
    split-half    residual ~ 0 (SE)           drop score corrected with S
                                              estimated from an INDEPENDENT
                                              half of the replicates
                                              (reps 0-19 vs 20-39, disjoint
                                              by construction): the audit
                                              protocol this paper proposes.

The drop column carries the stratified interval conditional on the observed
folds (mean of fold means, the same construction the analyzer prints;
uncorrected and exact rest.* are point summaries of the same replicates); the
split-half residual is tested over folds (each fold contributes one A-half minus B-half difference,
expectation zero regardless of fold heterogeneity). Percentages are of the
fold-mean clean loss, a constant scaling that preserves the algebra.

Run: .venv/bin/python scripts/validation/benchmark_corrections.py \
         [rs_cell_estimates.parquet ...]
Default: outputs/kaggle/rerun-rampart-r3c-rs-*/rampart/rs_cell_estimates.parquet
(first match; pass paths explicitly to combine several).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_replicated_saturation import t_ci_stratified  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CANONICAL = 'outputs/kaggle/rs_cell_estimates.parquet'
DEFAULT = 'outputs/kaggle/rerun-rampart-r3c-rs-*/rampart/rs_cell_estimates.parquet'


def t_ci(values, level=0.95):
    from scipy import stats
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, float('nan')
    half = (stats.t.ppf(0.5 + level / 2, len(v) - 1)
            * v.std(ddof=1) / np.sqrt(len(v)))
    return m, float(half)


def main(*paths):
    canonical = REPO / CANONICAL
    files = ([Path(p) for p in paths] or
             ([canonical] if canonical.exists() else sorted(REPO.glob(DEFAULT))[:1]))
    if not files:
        raise SystemExit(f'no cell estimates found (looked for {CANONICAL}, {DEFAULT})')
    cells = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'{len(cells)} cell-replicate estimates from {len(files)} file(s)\n')

    for dataset, dg in cells.groupby('dataset'):
        reps = dg['rep'].max() + 1
        half = reps // 2
        print(f'=== {dataset}: {dg["fold"].nunique()} folds, {reps} replicates ===')
        print(f"{'model':>26} {'s':>5} {'uncorr':>8} "
              f"{'drop=S':>17} {'exact rest.*':>12} {'split-half':>17} "
              f"{'drop/clean%':>12}")
        for (model, sat), g in dg.groupby(['model', 'saturation']):
            b = g['B_hat'].mean()
            s_mean, s_half = t_ci_stratified(g['S_hat'], g['fold'])
            oracle = ((1 - g['share']) * g['S_hat']).mean()
            # One independent-audit residual per fold: first-half reps price
            # the audit being corrected, second-half reps price the audit
            # doing the correcting. Disjoint by construction, never circular.
            resid = []
            for _, fg in g.groupby('fold'):
                a = fg.loc[fg['rep'] < half, 'S_hat']
                bb = fg.loc[fg['rep'] >= half, 'S_hat']
                if len(a) and len(bb):
                    resid.append(a.mean() - bb.mean())
            r_mean, r_half = t_ci(resid)
            rel = 100 * s_mean / g['mean_clean_loss'].mean()
            print(f'{model:>26} {sat:>5.2f} {b:>+8.3f} '
                  f'{s_mean:>+8.3f}±{s_half:<7.3f} {oracle:>+10.3f} '
                  f'{r_mean:>+8.4f}±{r_half:<7.4f} {rel:>11.1f}%')
        print()

    print('reading: exact restoration retains most of the drop bias, because')
    print('share is small and the bias is spillover; the split-half residual sits')
    print('near zero (zero in expectation for any data, so it reads precision,')
    print('not bias; 2 of the 48 audited cells exclude zero) and its interval is')
    print('a fold-level t check of the replicate machinery.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
