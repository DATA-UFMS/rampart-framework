#!/usr/bin/env python3
"""Within-window exposure mapping: spillover by temporal distance, S(s, d).

For every uninserted evaluation row, d is the distance in years to the nearest
INSERTED row of the same draw. If interference travels through what the model
learns about nearby years, S should be largest at d = 0 (an inserted row from
the row's own year) and decay with d; if the spillover is a global shift (the
ridge's mechanism), S should be flat in d.

Scope guard, stated up front: this reads distances WITHIN the evaluation
window only (a span of a few years), because that is what the recorded draws
vary. It is the microscale seed of the interference radius, not the buffer
curve -- the full exposure mapping S(s, d) over withheld distances needs the
buffer arms rerun with lag features rebuilt inside each arm, which this
script does not do.

Input: the per-row parquets written by probe_replicated_saturation.py.
Inference: t over replicate-level distance means, pooled across folds -- the
same design-based construction as the other readers. Cells where a replicate
has no row at distance d simply contribute nothing for that d.

Run: .venv/bin/python scripts/validation/analyze_spillover_distance.py \
         [parquet-or-dir ...]
Default: outputs/kaggle/rerun-rampart-r3c-rs-*/ (recursive).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = 'outputs/kaggle/rerun-rampart-r3c-rs-*/**/replicated_saturation_*.parquet'
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
    """One fold's rows -> replicate-level mean gain per (model, sat, distance)."""
    m = NAME.search(path.name)
    dataset, fold = m.group(1), int(m.group(2))
    df = pd.read_parquet(
        path, columns=['model', 'saturation', 'rep', 'arm', 'row', 'year',
                       'handed', 'y_true', 'y_pred'])
    loss = (df['y_true'].astype('float64') - df['y_pred'].astype('float64')) ** 2
    df = df.assign(loss=loss)
    clean = (df[df['arm'] == 'clean']
             .set_index(['model', 'row'])['loss'].rename('clean_loss'))
    leak = df[df['arm'] == 'leak']

    out = []
    # The draw is shared across models within (saturation, rep) -- seeded by
    # fold_rng(fold, dose, rep) -- so handed years are computed once per draw.
    for (sat, rep), draw in leak.groupby(['saturation', 'rep'], observed=True):
        one = draw[draw['model'] == draw['model'].iloc[0]]
        handed_years = np.sort(one.loc[one['handed'], 'year'].unique())
        for model, g in draw.groupby('model', observed=True):
            un = g[~g['handed']]
            base = clean.loc[model].reindex(un['row']).to_numpy()
            gain = base - un['loss'].to_numpy()
            dist = np.min(np.abs(un['year'].to_numpy()[:, None]
                                 - handed_years[None, :]), axis=1)
            for d in np.unique(dist):
                sel = dist == d
                out.append({'dataset': dataset, 'fold': fold,
                            'model': str(model), 'saturation': float(sat),
                            'rep': int(rep), 'distance': int(d),
                            'gain': float(gain[sel].mean()),
                            'n_rows': int(sel.sum())})
    return pd.DataFrame(out)


def t_ci(values, level=0.95):
    from scipy import stats
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, float('nan')
    half = (stats.t.ppf(0.5 + level / 2, len(v) - 1)
            * v.std(ddof=1) / np.sqrt(len(v)))
    return m, float(half)


def main(*args):
    files = discover(list(args))
    if not files:
        raise SystemExit(f'no parquet found (looked for {DEFAULT_GLOB})')
    parts = []
    for f in files:
        parts.append(reduce_file(f))
        print(f'  reduced {f.name}', flush=True)
    cells = pd.concat(parts, ignore_index=True)
    from analyze_replicated_saturation import write_canonical
    out = write_canonical(cells, 'rs_distance_estimates.parquet')
    print(f'\n{len(cells)} replicate-distance estimates -> {out}\n')

    for dataset, dg in cells.groupby('dataset'):
        dists = sorted(dg['distance'].unique())
        reps_per_cell = dg.groupby(['model', 'saturation', 'fold'])['rep'].nunique().max()
        print(f'=== {dataset}: S by distance to nearest inserted year '
              f'(folds pooled, t over replicate means; n = replicate-cells) ===')
        if len(dists) == 1:
            print('  SUPPORT WARNING: every evaluation window in this panel spans')
            print('  a single year, so d=0 is the only realisable distance and no')
            print('  flat-vs-decay reading exists here by construction.')
        head = ' '.join(f"{'d=' + str(d):>25}" for d in dists)
        print(f"{'model':>26} {'s':>5} {head}")
        for (model, sat), g in dg.groupby(['model', 'saturation']):
            row = []
            for d in dists:
                vals = g.loc[g['distance'] == d, 'gain']
                if len(vals) == 0:
                    row.append(f"{'--':>25}")
                else:
                    m, h = t_ci(vals)
                    row.append(f'{m:>+9.3f}±{h:<7.3f}(n={len(vals):>4})')
            print(f'{model:>26} {sat:>5.2f} ' + ' '.join(row))
        print()

    print('reading guard: a flat-vs-decay statement needs at least two distances')
    print('with real support. In these panels the windows span 1-2 years, larger')
    print('distances exist only at low saturation, and where both d=0 and d=1')
    print('exist their intervals overlap -- the within-window range does NOT')
    print('discriminate global from local spillover. That discrimination, and')
    print('the interference radius, need the buffer arms rerun with lag features')
    print('rebuilt inside each arm; this reader is the microscale seed only.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
