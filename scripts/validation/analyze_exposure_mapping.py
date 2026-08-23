#!/usr/bin/env python3
"""S(s, d) curves and the interference radius, from the exposure-mapping runs.

Reads the per-row parquets written by probe_exposure_mapping.py and reduces
them to the exposure mapping: per (dataset, model, saturation, distance), the
mean per-row improvement of evaluation rows when share-matched rows from the
year at temporal distance d entered training -- with the design-based t
interval over replicate draws that every reader in this chain uses.

Interference radius, with a PRACTICAL EQUIVALENCE REGION rather than a bare
null test: on the normalised scale (S as a share of the arm's own clean
loss -- each distance owns its baseline, since interior arms withhold their
year and rebuild lags), a cell's verdict against the margin eps is

    EQUIV      the 95% CI lies entirely inside (-eps, +eps)
    NON-EQUIV  the 95% CI lies entirely outside the region
    UNDECIDED  the CI straddles a boundary -- more replicates or folds
               would be needed to call it

and the radius at eps is the smallest d whose verdict is EQUIV with no
larger measured d NON-EQUIV or UNDECIDED. Three margins are declared up
front -- 1%, 2%, 5% of clean loss -- and the radius is reported under each,
so the reading carries its own sensitivity to the margin instead of a single
tuned threshold. (CI-in-region at 95% is the two-one-sided-tests reading at
alpha = 0.025 per side.)

Also printed: the d = 0 cross-check against the replicated-saturation audit
(prediction P2 of the probe): agreement is expected for encoding-insensitive
models (random forest, gradient boosting) and NOT for the ridge, whose
entity-effect encoding is recomputed with the leak here and was frozen
there -- a declared difference in treatment.

Scope notes printed with the tables: replicate intervals price assignment
noise only; fold spread is shown, cross-fold inference is not claimed; and
each distance's S is measured against that arm's own honest baseline, which
is the counterfactual question each distance asks.

Run: .venv/bin/python scripts/validation/analyze_exposure_mapping.py \
         [parquet-or-dir ...]
Default: outputs/kaggle/rerun-rampart-r3c-em-*/ (recursive).
Writes outputs/kaggle/em_cell_estimates.parquet (canonical, guarded).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = 'outputs/kaggle/rerun-rampart-r3c-em-*/**/exposure_mapping_*.parquet'
NAME = re.compile(r'exposure_mapping_(.+)_fold(\d+)_r(\d+)\.parquet$')
MARGINS = (0.01, 0.02, 0.05)


def discover(args):
    if not args:
        return sorted(REPO.glob(DEFAULT_GLOB))
    files = []
    for a in args:
        p = Path(a)
        files += (sorted(p.rglob('exposure_mapping_*.parquet'))
                  if p.is_dir() else [p])
    return files


def reduce_file(path):
    """One (fold, replicate-block) parquet -> per-replicate cell estimates."""
    m = NAME.search(path.name)
    if m is None:
        raise SystemExit(f'unrecognised file name: {path}')
    dataset, fold = m.group(1), int(m.group(2))
    df = pd.read_parquet(path)
    loss = (df['y_true'].astype('float64') - df['y_pred'].astype('float64')) ** 2
    df = df.assign(loss=loss)

    clean = (df[df['arm'] == 'clean']
             .drop_duplicates(subset=['model', 'distance', 'row'])
             .set_index(['model', 'distance', 'row'])['loss']
             .rename('clean_loss').sort_index())
    rows = []
    leak = df[df['arm'] == 'leak']
    for (model, dist, sat, rep), g in leak.groupby(
            ['model', 'distance', 'saturation', 'rep'], observed=True):
        base = clean.loc[(model, dist)].reindex(g['row']).to_numpy()
        if np.isnan(base).any():
            raise SystemExit(f'{path}: leak rows without a clean twin in '
                             f'(model={model}, d={dist}) -- refusing to guess')
        gain = base - g['loss'].to_numpy()
        handed = g['handed'].to_numpy()
        un = ~handed
        rows.append({
            'dataset': dataset, 'fold': fold, 'model': str(model),
            'distance': int(dist),
            'inserted_year': int(g['inserted_year'].iloc[0]),
            'saturation': float(sat), 'rep': int(rep),
            'share': float(handed.mean()),
            'S_hat': float(gain[un].mean()),
            'L_hat': float(gain[handed].mean()) if handed.any() else np.nan,
            'mean_clean_loss': float(clean.loc[(model, dist)].mean()),
        })
    return pd.DataFrame(rows)


def t_ci(values, level=0.95):
    from scipy import stats
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, float('nan')
    half = (stats.t.ppf(0.5 + level / 2, len(v) - 1)
            * v.std(ddof=1) / np.sqrt(len(v)))
    return m, float(half)


def verdict(lo, hi, eps):
    if -eps < lo and hi < eps:
        return 'EQUIV'
    if hi < -eps or lo > eps:
        return 'NON-EQUIV'
    return 'UNDECIDED'


def radius(verdicts_by_d, eps_verdicts):
    """Smallest d EQUIV with every larger measured d also EQUIV."""
    ds = sorted(verdicts_by_d)
    for i, d in enumerate(ds):
        tail = [eps_verdicts[dd] for dd in ds[i:]]
        if all(v == 'EQUIV' for v in tail):
            return str(d)
    return f'>{ds[-1]}'


def main(*args):
    files = discover(list(args))
    if not files:
        raise SystemExit(f'no parquet found (looked for {DEFAULT_GLOB})')
    parts = []
    for f in files:
        parts.append(reduce_file(f))
        print(f'  reduced {f.name}', flush=True)
    cells = (pd.concat(parts, ignore_index=True)
             .drop_duplicates(subset=['dataset', 'fold', 'model', 'distance',
                                      'saturation', 'rep']))
    from analyze_replicated_saturation import write_canonical
    out = write_canonical(cells, 'em_cell_estimates.parquet')
    print(f'\n{len(cells)} replicate-cell estimates -> {out}\n')

    rs_path = REPO / 'outputs' / 'kaggle' / 'rs_cell_estimates.parquet'
    rs = pd.read_parquet(rs_path) if rs_path.exists() else None

    for dataset, dg in cells.groupby('dataset'):
        print(f"=== {dataset}: S(s, d), {dg['fold'].nunique()} folds, "
              f"{dg.groupby(['fold', 'model', 'distance', 'saturation'])['rep'].nunique().max()} "
              f"replicates per cell ===")
        print(f"{'model':>26} {'s':>5} {'d':>3} {'S(s,d)':>19} "
              f"{'S/clean%':>19} {'verdito@1%':>10} {'clean loss':>11}")
        norm_ci = {}
        for (model, sat, dist), g in dg.groupby(['model', 'saturation',
                                                 'distance']):
            s_mean, s_half = t_ci(g['S_hat'])
            rel = g['S_hat'] / g['mean_clean_loss']
            r_mean, r_half = t_ci(rel)
            v = verdict(r_mean - r_half, r_mean + r_half, MARGINS[0])
            norm_ci[(model, sat, dist)] = (r_mean - r_half, r_mean + r_half)
            print(f'{model:>26} {sat:>5.2f} {dist:>3} '
                  f'{s_mean:>+9.3f}±{s_half:<8.3f} '
                  f'{100 * r_mean:>+8.2f}±{100 * r_half:<7.2f} '
                  f'{v:>10} {g["mean_clean_loss"].mean():>11.2f}')
        print()

        print(f"--- interference radius, {dataset} "
              f"(smallest d EQUIV with all larger d EQUIV) ---")
        print(f"{'model':>26} {'s':>5} " +
              ' '.join(f'{f"eps={100*e:g}%":>9}' for e in MARGINS))
        for (model, sat), g in dg.groupby(['model', 'saturation']):
            ds = sorted(g['distance'].unique())
            cols = []
            for eps in MARGINS:
                vs = {d: verdict(*norm_ci[(model, sat, d)], eps) for d in ds}
                cols.append(f'{radius(vs, vs):>9}')
            print(f'{model:>26} {sat:>5.2f} ' + ' '.join(cols))
        print()

        if rs is not None:
            both = rs[(rs['dataset'] == dataset)
                      & rs['saturation'].isin(dg['saturation'].unique())]
            if len(both):
                print(f'--- P2 cross-check at d = 0 vs the rs audit '
                      f'(agreement expected only for encoding-insensitive '
                      f'models) ---')
                d0 = dg[dg['distance'] == 0]
                for (model, sat), g in d0.groupby(['model', 'saturation']):
                    twin = both[(both['model'] == model)
                                & (both['saturation'] == sat)]
                    if not len(twin):
                        continue
                    a, ah = t_ci(g['S_hat'])
                    b, bh = t_ci(twin['S_hat'])
                    print(f'{model:>26} {sat:>5.2f}  em {a:>+8.3f}±{ah:<7.3f}'
                          f'  rs {b:>+8.3f}±{bh:<7.3f}')
                print()

    print('reading: intervals are t over replicate draws (assignment noise');
    print('only; fold spread not separately shown here -- the cell parquet')
    print('carries fold for any cross-fold treatment). Each distance is')
    print('measured against its own honest baseline: interior arms withhold')
    print('their year and rebuild lags, so their clean loss differs by design.')
    print('UNDECIDED cells block the radius; they are a replicate budget')
    print('statement, not a finding.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
