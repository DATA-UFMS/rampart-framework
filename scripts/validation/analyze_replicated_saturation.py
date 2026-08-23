#!/usr/bin/env python3
"""Design-based estimates of S(s), D(s), B(s) from replicated-saturation records.

Reads the per-row parquet files written by probe_replicated_saturation.py and
reduces them to the interference estimands:

    per replicate r (one independent draw of the inserted set):
        gain_i   = clean_loss_i - leak_loss_i          (additive, per row)
        L_hat(r) = mean gain over inserted rows        (estimates S + D)
        S_hat(r) = mean gain over uninserted rows      (estimates S)
        D_hat(r) = L_hat(r) - S_hat(r)                 (estimates D)
        B_hat(r) = mean gain over all rows             (= S + s*D, exactly)

Inference is design-based: replicates are iid draws from the design the audit
itself randomised, so a t interval over R replicates prices the assignment
noise with no assumption about dependence between folds. Cross-fold statements
are NOT made here -- the fold column is reported as spread, and whatever
generalisation claim the paper makes across folds goes through the dependent-
fold machinery it already carries.

The B = S + s*D line per cell is arithmetic, not a finding; it is printed as a
pipeline self-check and must hold to float precision.

Also printed: gains normalised by the fold's mean clean loss (a constant per
fold-model cell, so the scaling preserves the additive algebra), which is the
scale on which panels of different units can sit in one table.

Run: .venv/bin/python scripts/validation/analyze_replicated_saturation.py \
         [parquet-or-dir ...]
Default input: outputs/kaggle/rerun-rampart-r3c-rs-*/replicated_saturation_*.parquet
Writes rs_cell_estimates.parquet (one row per dataset/fold/model/saturation/rep)
next to the first input directory, for the paper generators to read.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = 'outputs/kaggle/rerun-rampart-r3c-rs-*/replicated_saturation_*.parquet'
NAME = re.compile(r'replicated_saturation_(.+)_fold(\d+)\.parquet$')


def discover(args):
    if not args:
        return sorted(REPO.glob(DEFAULT_GLOB))
    files = []
    for a in args:
        p = Path(a)
        files += sorted(p.rglob('replicated_saturation_*.parquet')) if p.is_dir() else [p]
    return files


def reduce_file(path):
    """One parquet (one dataset, one fold) -> per-replicate cell estimates."""
    m = NAME.search(path.name)
    if m is None:
        raise SystemExit(f'unrecognised file name: {path}')
    dataset, fold = m.group(1), int(m.group(2))
    df = pd.read_parquet(path)
    loss = (df['y_true'].astype('float64') - df['y_pred'].astype('float64')) ** 2
    df = df.assign(loss=loss)

    clean = (df[df['arm'] == 'clean']
             .set_index(['model', 'row'])['loss'].rename('clean_loss'))
    leak = df[df['arm'] == 'leak']
    rows = []
    for (model, sat, rep), g in leak.groupby(['model', 'saturation', 'rep'],
                                             observed=True):
        base = clean.loc[model].reindex(g['row']).to_numpy()
        gain = base - g['loss'].to_numpy()
        handed = g['handed'].to_numpy()
        l_hat = float(gain[handed].mean())
        s_hat = float(gain[~handed].mean())
        share = handed.mean()             # realised k/n, not the nominal dose
        rows.append({
            'dataset': dataset, 'fold': fold, 'model': str(model),
            'saturation': float(sat), 'rep': int(rep),
            'share': float(share),
            'L_hat': l_hat, 'S_hat': s_hat, 'D_hat': l_hat - s_hat,
            'B_hat': float(gain.mean()),
            'mean_clean_loss': float(clean.loc[model].mean()),
        })
    return pd.DataFrame(rows)


def t_ci(values, level=0.95):
    """t interval over iid replicate estimates: design-based by construction."""
    from scipy import stats
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), (float('nan'), float('nan'))
    se = v.std(ddof=1) / np.sqrt(len(v))
    half = stats.t.ppf(0.5 + level / 2, len(v) - 1) * se
    return float(v.mean()), (float(v.mean() - half), float(v.mean() + half))


def write_canonical(df, name):
    """One canonical output under outputs/kaggle, guarded against a partial
    run silently clobbering the consolidated file (it happened in review)."""
    out = REPO / 'outputs' / 'kaggle' / name
    if out.exists():
        old = len(pd.read_parquet(out, columns=[df.columns[0]]))
        if old > len(df):
            out = out.with_name(out.stem + '.partial.parquet')
            print(f'NOTE: existing {name} has {old} rows > {len(df)} new;'
                  f' writing {out.name} to protect the fuller file')
    df.to_parquet(out, index=False)
    return out


def main(*args):
    files = discover(list(args))
    if not files:
        raise SystemExit(f'no parquet found (looked for {DEFAULT_GLOB})')
    cells = pd.concat([reduce_file(f) for f in files], ignore_index=True)

    out = write_canonical(cells, 'rs_cell_estimates.parquet')
    print(f'{len(files)} files -> {len(cells)} cell-replicate estimates -> {out}\n')

    check = (cells['B_hat']
             - (cells['S_hat'] + cells['share'] * cells['D_hat'])).abs().max()
    print(f'self-check  max |B - (S + share*D)| = {check:.2e}  '
          f'(arithmetic identity; anything above float noise is a bug)\n')

    for dataset, dgroup in cells.groupby('dataset'):
        n_folds = dgroup['fold'].nunique()
        reps = dgroup['rep'].nunique()
        print(f'=== {dataset}: {n_folds} folds, {reps} replicates per cell ===')
        print(f"{'model':>26} {'s':>5} {'S(s)':>22} {'D(s)':>22} "
              f"{'B(s)':>9} {'S/clean%':>9} {'fold sd(S)':>11}")
        for (model, sat), g in dgroup.groupby(['model', 'saturation']):
            # Design-based interval: pool replicates across folds after
            # centring nothing -- each fold contributes R iid draws around its
            # own estimand, so the pooled t prices assignment noise around the
            # mean of fold-level estimands. Fold spread is shown beside it.
            s_mean, (s_lo, s_hi) = t_ci(g['S_hat'])
            d_mean, (d_lo, d_hi) = t_ci(g['D_hat'])
            b_mean = g['B_hat'].mean()
            fold_means = g.groupby('fold')['S_hat'].mean()
            rel = 100 * s_mean / g['mean_clean_loss'].mean()
            print(f'{model:>26} {sat:>5.2f} '
                  f'{s_mean:>+9.4f}[{s_lo:+.3f},{s_hi:+.3f}] '
                  f'{d_mean:>+9.4f}[{d_lo:+.3f},{d_hi:+.3f}] '
                  f'{b_mean:>+9.4f} {rel:>8.1f}% '
                  f'{fold_means.std(ddof=1) if len(fold_means) > 1 else float("nan"):>11.4f}')
        print()
        print('  intervals: t over replicate draws (assignment noise only);')
        print('  fold sd(S): spread of per-fold means -- cross-fold inference')
        print('  is NOT claimed here and goes through the dependent-fold tools.')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
