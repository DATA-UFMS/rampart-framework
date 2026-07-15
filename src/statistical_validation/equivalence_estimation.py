#!/usr/bin/env python3
"""
Equivalência por Estimativa (SESOI + IC) com robustez (Wilcoxon + Hodges-Lehmann)

Substitui o TOST formal e é adequada a n pequeno entre folds:
- Define delta (SESOI) por métrica
- Estima efeito pareado DL vs DW com bootstrap (IC95%)
- Decide equivalência/superioridade/inferioridade/inconclusivo
- Aplica Wilcoxon pareado e Hodges-Lehmann como robustez

Limiares SESOI definidos a priori (ver scientific_config.py):
  R2=0.01  — metade do efeito pequeno de Cohen (1988, f2=0.02)
  MASE=0.05 — 5% relativo ao baseline naive (Hyndman & Koehler 2006)
  WAPE=0.05 — 5pp de erro ponderado

Abordagem hibrida distribution-based + anchor-based
conforme Lakens, Scheel & Isager (2018).

Nota sobre poder estatístico:
  O walk-forward com gaps de 2 anos produz n=9 folds (máximo sem
  comprometer anti-leakage). Wilcoxon pareado com n=9 tem poder
  ~30% para efeitos médios (d~0.5), insuficiente como teste primário.
  Por isso a decisão principal usa bootstrap CI: não depende de
  premissas assintóticas e fornece intervalo diretamente interpretável.
  Wilcoxon e Hodges-Lehmann são complementos de robustez.

  Interpretação dos desfechos com n pequeno:
    - "equivalent": forte — difícil de atingir com pouca precisão
    - "inconclusive": esperado — reflete a precisão disponível, não
      falha metodológica (Lakens et al. 2018)
    - "superior"/"inferior": requer corroboração pela sensibilidade

  A análise de sensibilidade (bootstrap_sensitivity.py) varia SESOI
  (0.5x, 1.0x, 1.5x) e iterações (1000, 3000, 10000, 15000) para
  verificar estabilidade das decisões.

O método que produziu cada IC é registrado em 'ci95_method': BCa,
fallback percentil (com a razão) ou degenerado por variância zero.
Os três não são intercambiáveis, e um IC sem essa informação não
permite ao leitor distinguir entre eles.

Saidas:
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

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
# No fallback: silently substituting the resample count or a SESOI would make
# the reported protocol differ from the executed one. The SESOI values in
# particular are the decision thresholds, defined a priori.
from core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED

DEFAULT_BOOTSTRAP_ITERS = int(SCIENTIFIC_CONFIG['bootstrap_iters'])
DEFAULT_SEED = RANDOM_SEED
DEFAULT_SESOI_R2 = float(SCIENTIFIC_CONFIG['sesoi_r2'])
DEFAULT_SESOI_MASE = float(SCIENTIFIC_CONFIG['sesoi_mase'])
DEFAULT_SESOI_WAPE = float(SCIENTIFIC_CONFIG['sesoi_wape'])

# Comparações par-a-par
PREDICTIVE_PAIRS = [("dl", "dw"), ("dl", "pl"), ("dw", "pl")]
LATENCY_PAIRS = [
    ("task_graph", "sql_engine", "dl", "dw"),
    ("task_graph", "dataframe_lib", "dl", "pl"),
    ("sql_engine", "dataframe_lib", "dw", "pl"),
]


def _median_hodges_lehmann(deltas: np.ndarray) -> float:
    n = len(deltas)
    if n == 0:
        return float('nan')
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append(0.5 * (deltas[i] + deltas[j]))
    return float(np.median(walsh))


def _bootstrap_ci(values: np.ndarray, iters: int = DEFAULT_BOOTSTRAP_ITERS, seed: int = DEFAULT_SEED, ci: float = 0.95) -> Tuple[float, Tuple[float, float], str]:
    """IC bootstrap (BCa com fallback percentil).

    Returns the point estimate, the interval, and which method produced it.
    Three methods can produce an interval here and they are not interchangeable,
    so reporting an interval without naming its method leaves the reader unable
    to tell BCa from a percentile fallback.
    """
    if len(values) == 0 or np.all(np.isnan(values)):
        return float('nan'), (float('nan'), float('nan')), 'insufficient_data'
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return float('nan'), (float('nan'), float('nan')), 'insufficient_data'
    point = float(np.mean(clean))

    # Zero variance: every resample has the same mean, so the interval is the
    # point estimate exactly. Not an approximation, and not optional -- BCa's
    # acceleration divides by the spread.
    if np.std(clean) < np.finfo(float).eps * 100:
        return point, (point, point), 'degenerate_zero_variance'

    try:
        from scipy.stats import bootstrap as scipy_bootstrap
        res = scipy_bootstrap(
            (clean,), np.mean, n_resamples=iters,
            confidence_level=ci, method='BCa',
            random_state=np.random.default_rng(seed)
        )
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"BCa returned a non-finite interval: [{lo}, {hi}]")
        return point, (lo, hi), 'bca'
    except Exception as exc:
        # BCa needs enough distinct resample values to estimate acceleration;
        # small n makes it fail or return non-finite endpoints. The percentile
        # interval is the documented fallback, recorded as such rather than
        # reported as BCa.
        reason = type(exc).__name__
        rng = np.random.default_rng(seed)
        n = len(clean)
        means = np.array([np.mean(clean[rng.integers(0, n, size=n)]) for _ in range(iters)])
        alpha = (1 - ci) / 2
        lo, hi = float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
        return point, (lo, hi), f'percentile_fallback:{reason}'


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
    """Extrai métricas de teste por fold do JSON de resultados de baseline.

    Usa o modelo 'best_baseline' quando disponível, recorrendo a 'naive_with_lag'.
    Extrai test_r2, test_mase, test_wape explicitamente por nome de chave.

    Compatível com ambas as estruturas de JSON:
      - DW/DL (v2+): baseline_model_results -> fold_X -> {best_baseline, naive_with_lag, ...}
      - PL (v1 legacy): baseline_models -> fold_X -> {naive_with_lag, cross_country_average, ...}
    """
    out: Dict[int, Dict[str, float]] = {}
    # Folds ficam em 'baseline_model_results', 'baseline_models', ou diretamente no dict
    folds_container = d.get('baseline_model_results') or d.get('baseline_models') or d
    if not isinstance(folds_container, dict):
        return out
    for k, v in folds_container.items():
        if not (isinstance(k, str) and k.startswith('fold_') and isinstance(v, dict)):
            continue
        try:
            fid = int(k.split('_')[1])
        except Exception:
            continue
        # Usa best_baseline se disponível, senão recorre a naive_with_lag
        best = v.get('best_baseline', {})
        best_name = best.get('model', '') if isinstance(best, dict) else ''
        model = v.get(best_name) if best_name else None
        if not isinstance(model, dict):
            model = v.get('naive_with_lag')
        if not isinstance(model, dict):
            for sub in v.values():
                if isinstance(sub, dict) and 'test_r2' in sub:
                    model = sub
                    break
        if not isinstance(model, dict):
            continue
        metrics = {}
        if 'test_r2' in model:
            metrics['r2'] = float(model['test_r2'])
        if 'test_mase' in model:
            metrics['mase'] = float(model['test_mase'])
        if 'test_wape' in model:
            metrics['wape'] = float(model['test_wape'])
        if metrics:
            out[fid] = metrics
    return out


def _load_baseline_pairs() -> Dict[str, Dict[int, Dict[str, float]]]:
    paths = {
        'dw': 'outputs/ml_pipeline/architectures/sql_engine/models/baseline_analysis_sql_engine_consumer_results.json',
        'dl': 'outputs/ml_pipeline/architectures/task_graph/models/baseline_results/baseline_analysis_task_graph_results.json',
        'pl': 'outputs/ml_pipeline/architectures/dataframe_lib/models/baseline_results/baseline_analysis_dataframe_lib_results.json',
    }
    out = {}
    for arch, p in paths.items():
        d = _load_json(p)
        out[arch] = _extract_fold_metrics(d) if d else {}
    return out


def _paired_deltas_for_metric(pairs: Dict[str, Dict[int, Dict[str, float]]], metric: str, arch_a: str = 'dl', arch_b: str = 'dw') -> np.ndarray:
    a = pairs.get(arch_a, {})
    b = pairs.get(arch_b, {})
    common_ids = sorted(set(a.keys()) & set(b.keys()))
    deltas = []
    for fid in common_ids:
        va = a[fid].get(metric)
        vb = b[fid].get(metric)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            deltas.append(vb - va)
    return np.array(deltas, dtype=float)


def _analyze_predictive_metrics(args) -> Dict:
    pairs = _load_baseline_pairs()
    results = {}
    cfg = {'r2': args.r2_delta, 'mase': args.mase_delta, 'wape': args.wape_delta}
    for arch_a, arch_b in PREDICTIVE_PAIRS:
        pair_key = f"{arch_a}_vs_{arch_b}"
        pair_results = {}
        for metric, delta in cfg.items():
            deltas = _paired_deltas_for_metric(pairs, metric, arch_a, arch_b)
            point, (lo, hi), ci_method = _bootstrap_ci(deltas, iters=args.bootstrap, seed=args.seed, ci=0.95)
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
            pair_results[metric] = {
                'delta': delta,
                'n_pairs': int(len(deltas)),
                'point_estimate': point,
                'ci95': [lo, hi],
                'ci95_method': ci_method,
                'decision': decision,
                'wilcoxon_p': wilcoxon_p,
                'hodges_lehmann': hl,
            }
        results[pair_key] = pair_results
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
        phase_results = {}
        for arch_name_a, arch_name_b, arch_a, arch_b in LATENCY_PAIRS:
            if arch_name_a not in vals.columns or arch_name_b not in vals.columns:
                continue
            pair_key = f"{arch_a}_vs_{arch_b}"
            x = vals[arch_name_a].to_numpy(dtype=float)
            y = vals[arch_name_b].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            lr = np.log(x[mask] / y[mask])
            if lr.size == 0:
                phase_results[pair_key] = {'status': 'insufficient_data'}
                continue
            point, (lo, hi), ci_method = _bootstrap_ci(lr, iters=args.bootstrap, seed=args.seed, ci=0.95)
            delta_pct = profile.get(str(phase).lower(), profile['total'])
            delta_lr = math.log(1.0 + delta_pct)
            decision = _decision_equivalence(lo, hi, delta_lr)
            try:
                w = stats.wilcoxon(lr, zero_method='wilcox', alternative='two-sided', method='auto')
                p = float(w.pvalue)
            except Exception:
                p = None
            hl = _median_hodges_lehmann(lr)
            phase_results[pair_key] = {
                'delta_pct': float(delta_pct),
                'n_pairs': int(lr.size),
                'point_estimate_lr': float(point),
                'ci95_lr': [float(lo), float(hi)],
                'ci95_method': ci_method,
                'decision': decision,
                'wilcoxon_p': p,
                'hodges_lehmann_lr': hl,
                'interpretation': {
                    'pct_effect': float((math.exp(point) - 1.0) * 100.0),
                    'ci95_pct': [float((math.exp(lo) - 1.0) * 100.0), float((math.exp(hi) - 1.0) * 100.0)],
                }
            }
        results[phase] = phase_results
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
            '\\caption{Equivalência prática por estimativa — predição (3-way pairwise)}',
            '\\begin{tabular}{llrrrrl}',
            '\\toprule',
            'Par & Métrica & n & Estim. & IC95\\% & $\\delta$ & Decisão \\\\ ',
            '\\midrule',
        ]
        pred = obj.get('predictive', {})
        for pair_key, pair_data in pred.items():
            if not isinstance(pair_data, dict):
                continue
            for m, r in pair_data.items():
                if not isinstance(r, dict):
                    continue
                n = r.get('n_pairs', 0)
                est = r.get('point_estimate', float('nan'))
                ci = r.get('ci95', [float('nan'), float('nan')])
                d = r.get('delta', float('nan'))
                dec = r.get('decision', '')
                lines.append(f"{pair_key} & {m} & {n} & {est:.3f} & [{ci[0]:.3f},{ci[1]:.3f}] & {d:.3f} & {dec} \\\\")
        lines += [
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}',
            '',
            '\\begin{table}[htb]',
            '\\centering',
            '\\caption{Equivalência prática por estimativa — latência (log‑ratio, 3-way pairwise)}',
            '\\begin{tabular}{llrrrrl}',
            '\\toprule',
            'Fase & Par & n & Estim. (LR) & IC95\\% & $\\delta$(%) & Decisão \\\\ ',
            '\\midrule',
        ]
        lat = obj.get('latency', {})
        for phase, phase_data in lat.items():
            if not isinstance(phase_data, dict):
                continue
            for pair_key, r in phase_data.items():
                if not isinstance(r, dict) or 'point_estimate_lr' not in r:
                    continue
                n = r.get('n_pairs', 0)
                est = r.get('point_estimate_lr', float('nan'))
                ci = r.get('ci95_lr', [float('nan'), float('nan')])
                d_pct = r.get('delta_pct', float('nan')) * 100.0
                dec = r.get('decision', '')
                lines.append(f"{phase} & {pair_key} & {n} & {est:.3f} & [{ci[0]:.3f},{ci[1]:.3f}] & {d_pct:.1f} & {dec} \\\\")
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
    # Extrair n_pairs para nota de poder (máximo entre todos os pares preditivos)
    n_pairs = max(
        (r.get('n_pairs', 0) for pair_data in predictive.values()
         if isinstance(pair_data, dict)
         for r in pair_data.values() if isinstance(r, dict)),
        default=0
    )
    out = {
        'method': 'equivalence_by_estimation',
        'seed': args.seed,
        'bootstrap': args.bootstrap,
        'n_folds': n_pairs,
        'power_note': (
            f'n={n_pairs} folds (maximo sem comprometer anti-leakage temporal). '
            f'Analise 3-way pairwise: dl_vs_dw, dl_vs_pl, dw_vs_pl. '
            f'Wilcoxon pareado com n={n_pairs} tem poder limitado (~30% para d=0.5); '
            f'decisao principal via bootstrap CI. Resultado "inconclusive" e esperado '
            f'e reflete precisao disponivel (Lakens et al. 2018).'
        ) if n_pairs > 0 else 'Sem dados para analise de poder.',
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
    p.add_argument('--r2-delta', type=float, default=DEFAULT_SESOI_R2, help=f'SESOI δ para R² (default: {DEFAULT_SESOI_R2})')
    p.add_argument('--mase-delta', type=float, default=DEFAULT_SESOI_MASE, help=f'SESOI δ para MASE (default: {DEFAULT_SESOI_MASE})')
    p.add_argument('--wape-delta', type=float, default=DEFAULT_SESOI_WAPE, help=f'SESOI δ para WAPE (default: {DEFAULT_SESOI_WAPE})')
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

