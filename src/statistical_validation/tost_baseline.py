#!/usr/bin/env python3
"""
Equivalência por Estimativa (SESOI + IC) com robustez (Wilcoxon + Hodges–Lehmann)

Substitui o TOST e é adequada a n pequeno entre folds:
- Define δ (SESOI) por métrica
- Estima efeito pareado DL vs DW com bootstrap (IC95%)
- Decide equivalência/superioridade/inferioridade/inconclusivo
- Aplica Wilcoxon pareado e Hodges–Lehmann como robustez

Saídas:
- JSON: outputs/statistics/equivalence_estimation.json
- LaTeX (opcional): outputs/statistics/equivalence_estimation.tex
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple, Optional
import math
import numpy as np
import pandas as pd
from scipy import stats

# Raiz do projeto para importar configuração científica
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
try:
    from src.core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED  # type: ignore
    DEFAULT_BOOTSTRAP_ITERS = int(SCIENTIFIC_CONFIG.get('bootstrap_iters', 3000))
    DEFAULT_SEED = RANDOM_SEED
except Exception:
    DEFAULT_BOOTSTRAP_ITERS = 3000
    DEFAULT_SEED = 42


def _median_hodges_lehmann(deltas: np.ndarray) -> float:
    n = len(deltas)
    if n == 0:
        return float('nan')
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append(0.5 * (deltas[i] + deltas[j]))
    return float(np.median(walsh))


def _bootstrap_ci(values: np.ndarray, iters: int = DEFAULT_BOOTSTRAP_ITERS, seed: int = DEFAULT_SEED, ci: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    if len(values) == 0 or np.all(np.isnan(values)):
        return float('nan'), (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return float('nan'), (float('nan'), float('nan'))
    means = np.empty(iters, dtype=float)
    n = len(clean)
    for b in range(iters):
        idx = rng.integers(0, n, size=n)
        means[b] = np.mean(clean[idx])
    point = float(np.mean(clean))
    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
    return point, (lo, hi)


def _decision_equivalence(ci_lo: float, ci_hi: float, delta: float) -> str:
    if math.isnan(ci_lo) or math.isnan(ci_hi):
        return 'insufficient_data'
    if -delta <= ci_lo and ci_hi <= delta:
        return 'equivalent'
    if ci_lo > delta:
        return 'superior'
    if ci_hi < -delta:
        return 'inferior'
    return 'inconclusive'


def _load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _extract_fold_metrics(d: Dict) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    candidates = []
    if isinstance(d, dict):
        candidates.append(d)
        for _, v in d.items():
            if isinstance(v, dict):
                candidates.append(v)
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            if not (isinstance(k, str) and k.startswith('fold_') and isinstance(v, dict)):
                continue
            try:
                fid = int(k.split('_')[1])
            except Exception:
                continue
            metrics = {}
            for model_key in ['naive_with_lag', 'naive', 'baseline', 'global_mean', 'summary', 'test_metrics']:
                if model_key in v and isinstance(v[model_key], dict):
                    leaf = v[model_key]
                    if 'test_metrics' in leaf and isinstance(leaf['test_metrics'], dict):
                        leaf = leaf['test_metrics']
                    for key, val in leaf.items():
                        if not isinstance(val, (int, float)):
                            continue
                        lk = key.lower()
                        if 'r2' in lk:
                            metrics['r2'] = float(val)
                        elif 'nrmse' in lk:
                            metrics['nrmse'] = float(val)
                        elif 'mase' in lk:
                            metrics['mase'] = float(val)
            if not metrics:
                for key, val in v.items():
                    if not isinstance(val, (int, float)):
                        continue
                    lk = str(key).lower()
                    if 'r2' in lk:
                        metrics['r2'] = float(val)
                    elif 'nrmse' in lk:
                        metrics['nrmse'] = float(val)
                    elif 'mase' in lk:
                        metrics['mase'] = float(val)
            if metrics:
                out[fid] = metrics
    return out


def _load_baseline_pairs() -> Dict[str, Dict[int, Dict[str, float]]]:
    paths = {
        'dw': 'outputs/ml_pipeline/architectures/data_warehouse/models/baseline_analysis_data_warehouse_consumer_results.json',
        'dl': 'outputs/ml_pipeline/architectures/data_lake/models/baseline_analysis_data_lake_results.json',
    }
    out = {}
    for arch, p in paths.items():
        d = _load_json(p)
        out[arch] = _extract_fold_metrics(d) if d else {}
    return out


def _paired_deltas_for_metric(pairs: Dict[str, Dict[int, Dict[str, float]]], metric: str) -> np.ndarray:
    dl = pairs.get('dl', {})
    dw = pairs.get('dw', {})
    common_ids = sorted(set(dl.keys()) & set(dw.keys()))
    deltas = []
    for fid in common_ids:
        vdl = dl[fid].get(metric)
        vdw = dw[fid].get(metric)
        if isinstance(vdl, (int, float)) and isinstance(vdw, (int, float)):
            deltas.append(vdw - vdl)
    return np.array(deltas, dtype=float)


def _analyze_predictive_metrics(args) -> Dict:
    pairs = _load_baseline_pairs()
    results = {}
    cfg = {'r2': args.r2_delta, 'nrmse': args.nrmse_delta, 'mase': args.mase_delta}
    for metric, delta in cfg.items():
        deltas = _paired_deltas_for_metric(pairs, metric)
        point, (lo, hi) = _bootstrap_ci(deltas, iters=args.bootstrap, seed=args.seed, ci=0.95)
        decision = _decision_equivalence(lo, hi, delta)
        wilcoxon_p = None
        hl = None
        if len(deltas) >= 1 and not np.all(np.isnan(deltas)):
            try:
                w = stats.wilcoxon(deltas, zero_method='wilcox', alternative='two-sided', method='auto')
                wilcoxon_p = float(w.pvalue)
            except Exception:
                wilcoxon_p = None
            hl = _median_hodges_lehmann(deltas)
        results[metric] = {
            'delta': delta,
            'n_pairs': int(len(deltas)),
            'point_estimate': point,
            'ci95': [lo, hi],
            'decision': decision,
            'wilcoxon_p': wilcoxon_p,
            'hodges_lehmann': hl,
        }
    return results


def _load_benchmark_csv() -> Optional[pd.DataFrame]:
    p = 'outputs/benchmarks/architectural_benchmark_results.csv'
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _parse_latency_profile(s: Optional[str], default_total: float) -> Dict[str, float]:
    profile = {'setup': 0.15, 'processing': 0.10, 'baseline': 0.10, 'hierarchical': 0.05, 'total': default_total}
    if not s:
        return profile
    try:
        for part in s.split(','):
            k, v = part.split(':')
            profile[k.strip().lower()] = float(v.strip())
    except Exception:
        pass
    return profile


def _analyze_latency(args) -> Dict:
    df = _load_benchmark_csv()
    if df is None or df.empty:
        return {'status': 'missing_benchmark'}
    cols = {c.lower(): c for c in df.columns}
    needed = ['architecture', 'phase', 'run_id', 'duration_s']
    if not all(c in cols for c in needed):
        return {'status': 'invalid_benchmark_columns', 'missing': [c for c in needed if c not in cols]}
    sub = df[[cols['architecture'], cols['phase'], cols['run_id'], cols['duration_s']]].copy()
    sub.columns = ['architecture', 'phase', 'run_id', 'duration_s']
    sub['architecture'] = sub['architecture'].str.lower()
    piv = sub.pivot_table(index=['phase', 'run_id'], columns='architecture', values='duration_s', aggfunc='mean')
    results: Dict[str, Dict] = {}
    profile = _parse_latency_profile(args.latency_delta_profile, args.latency_delta)
    for phase, pdf in piv.groupby(level=0):
        vals = pdf.droplevel(0)
        if 'data_warehouse' not in vals.columns or 'data_lake' not in vals.columns:
            results[phase] = {'status': 'missing_architectures'}
            continue
        x = vals['data_warehouse'].to_numpy(dtype=float)
        y = vals['data_lake'].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        lr = np.log(x[mask] / y[mask])
        if lr.size == 0:
            results[phase] = {'status': 'insufficient_data'}
            continue
        point, (lo, hi) = _bootstrap_ci(lr, iters=args.bootstrap, seed=args.seed, ci=0.95)
        delta_pct = profile.get(str(phase).lower(), profile['total'])
        delta_lr = math.log(1.0 + delta_pct)
        decision = _decision_equivalence(lo, hi, delta_lr)
        try:
            w = stats.wilcoxon(lr, zero_method='wilcox', alternative='two-sided', method='auto')
            p = float(w.pvalue)
        except Exception:
            p = None
        hl = _median_hodges_lehmann(lr)
        results[phase] = {
            'delta_pct': float(delta_pct),
            'n_pairs': int(lr.size),
            'point_estimate_lr': float(point),
            'ci95_lr': [float(lo), float(hi)],
            'decision': decision,
            'wilcoxon_p': p,
            'hodges_lehmann_lr': hl,
            'interpretation': {
                'pct_effect': float((math.exp(point) - 1.0) * 100.0),
                'ci95_pct': [float((math.exp(lo) - 1.0) * 100.0), float((math.exp(hi) - 1.0) * 100.0)],
            }
        }
    return results


def _save_outputs(obj: Dict, write_tex: bool = False) -> None:
    os.makedirs('outputs/statistics', exist_ok=True)
    with open('outputs/statistics/equivalence_estimation.json', 'w') as f:
        json.dump(obj, f, indent=2)
    if write_tex:
        lines = [
            '% Equivalência por Estimativa (SESOI + IC) — Gerado automaticamente',
            '\\begin{table}[htb]',
            '\\centering',
            '\\caption{Equivalência prática por estimativa — predição}',
            '\\begin{tabular}{lrrrrl}',
            '\\toprule',
            'Métrica & n & Estim. & IC95\\% & $\\delta$ & Decisão \\\\ ',
            '\\midrule',
        ]
        pred = obj.get('predictive', {})
        for m, r in pred.items():
            if not isinstance(r, dict):
                continue
            n = r.get('n_pairs', 0)
            est = r.get('point_estimate', float('nan'))
            ci = r.get('ci95', [float('nan'), float('nan')])
            d = r.get('delta', float('nan'))
            dec = r.get('decision', '')
            lines.append(f"{m} & {n} & {est:.3f} & [{ci[0]:.3f},{ci[1]:.3f}] & {d:.3f} & {dec} \\\\")
        lines += [
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}',
            '',
            '\\begin{table}[htb]',
            '\\centering',
            '\\caption{Equivalência prática por estimativa — latência (log‑ratio)}',
            '\\begin{tabular}{lrrrrl}',
            '\\toprule',
            'Fase & n & Estim. (LR) & IC95\\% & $\\delta$(%) & Decisão \\\\ ',
            '\\midrule',
        ]
        lat = obj.get('latency', {})
        for phase, r in lat.items():
            if not isinstance(r, dict) or 'point_estimate_lr' not in r:
                continue
            n = r.get('n_pairs', 0)
            est = r.get('point_estimate_lr', float('nan'))
            ci = r.get('ci95_lr', [float('nan'), float('nan')])
            d_pct = r.get('delta_pct', float('nan')) * 100.0
            dec = r.get('decision', '')
            lines.append(f"{phase} & {n} & {est:.3f} & [{ci[0]:.3f},{ci[1]:.3f}] & {d_pct:.1f} & {dec} \\\\")
        lines += [
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}',
        ]
        with open('outputs/statistics/equivalence_estimation.tex', 'w') as f:
            f.write("\n".join(lines))


def run(args: argparse.Namespace) -> int:
    predictive = _analyze_predictive_metrics(args)
    latency = _analyze_latency(args)
    out = {
        'method': 'equivalence_by_estimation',
        'seed': args.seed,
    'bootstrap': args.bootstrap,
        'predictive': predictive,
        'latency': latency,
    }
    _save_outputs(out, write_tex=args.latex)
    print(json.dumps(out))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Equivalência por Estimativa (SESOI + IC) para baselines e latência.')
    p.add_argument('--bootstrap', type=int, default=DEFAULT_BOOTSTRAP_ITERS, help=f'Iterações de bootstrap (default: {DEFAULT_BOOTSTRAP_ITERS})')
    p.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Seed (default: 42)')
    p.add_argument('--r2-delta', type=float, default=0.01, help='SESOI δ para R² (default: 0.01)')
    p.add_argument('--nrmse-delta', type=float, default=0.05, help='SESOI δ para nRMSE (default: 0.05)')
    p.add_argument('--mase-delta', type=float, default=0.05, help='SESOI δ para MASE (default: 0.05)')
    p.add_argument('--latency-delta', type=float, default=0.10, help='SESOI δ para latência TOTAL (default: 0.10)')
    p.add_argument('--latency-delta-profile', type=str, default='setup:0.15,processing:0.10,baseline:0.10,hierarchical:0.05,total:0.10',
                   help='Perfil de δ por fase (ex.: setup:0.15,processing:0.10,baseline:0.10,hierarchical:0.05,total:0.10)')
    p.add_argument('--latex', action='store_true', help='Gerar também tabelas LaTeX')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()

