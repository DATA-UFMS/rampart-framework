#!/usr/bin/env python3
"""
Equivalence by Estimation (SESOI + CI) with robustness (Wilcoxon + Hodges-Lehmann)

Replaces the formal TOST and is suited to the small n across folds:
- Defines delta (SESOI) per metric
- Estimates the paired DL vs DW effect with bootstrap (95% CI)
- Decides equivalence, or which side exceeds the other, or inconclusive
- Applies paired Wilcoxon and Hodges-Lehmann as robustness checks

Why there is no explicit TOST:
  Two one-sided tests at alpha=0.05 are identical to checking whether the
  90% CI (1-2*alpha) falls entirely within +-delta. The decision
  here uses a 95% CI, which is wider and therefore harder to
  contain within +-delta: concluding equivalence by this criterion is
  strictly more conservative than by TOST at alpha=0.05, and the
  interval reports the magnitude, which the TOST p-value does not
  (Lakens, Scheel & Isager, 2018).

SESOI thresholds defined a priori (see scientific_config.py):
  R2=0.01  — half of Cohen's small effect (1988, f2=0.02)
  MASE=0.05 — 5% relative to the naive baseline (Hyndman & Koehler 2006)
  WAPE=0.05 — 5pp of weighted error

Hybrid distribution-based + anchor-based approach
following Lakens, Scheel & Isager (2018).

Note on statistical power:
  Walk-forward with 2-year gaps yields n=9 folds (the maximum without
  compromising anti-leakage). Paired Wilcoxon with n=9 has
  ~30% power for medium effects (d~0.5), insufficient as a primary test.
  The main decision therefore uses the bootstrap CI: it does not depend on
  asymptotic assumptions and provides a directly interpretable interval.
  Wilcoxon and Hodges-Lehmann are robustness complements.

  Interpreting the outcomes with small n:
    - "equivalent": strong — hard to reach with little precision
    - "inconclusive": expected — reflects the available precision, not a
      methodological failure (Lakens et al. 2018)
    - "a_exceeds_b"/"b_exceeds_a": requires corroboration by the sensitivity analysis

  The sensitivity analysis (bootstrap_sensitivity.py) varies SESOI
  (0.5x, 1.0x, 1.5x) and iterations (1000, 3000, 10000, 15000) to
  check the stability of the decisions.

The decision is directional, not meritorious: the effect is measured as A-B
(or log(A/B)), and whether a positive effect favours A depends on the metric
— it does for R2, it does not for latency, MASE and WAPE. The field
'advantage' names the favoured side, already accounting for that direction.

The method that produced each CI is recorded in 'ci95_method': BCa,
percentile fallback (with the reason) or degenerate from zero variance.
The three are not interchangeable, and a CI without that information does
not let the reader tell them apart.

Outputs:
- JSON: outputs/statistics/equivalence_estimation.json
- LaTeX (optional): outputs/statistics/equivalence_estimation.tex
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple
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
from core.config import get_absolute_output_path
from core.paradigm_registry import baseline_results_paths, paradigm_pairs
from core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED

DEFAULT_BOOTSTRAP_ITERS = int(SCIENTIFIC_CONFIG['bootstrap_iters'])
DEFAULT_SEED = RANDOM_SEED
DEFAULT_SESOI_R2 = float(SCIENTIFIC_CONFIG['sesoi_r2'])
DEFAULT_SESOI_MASE = float(SCIENTIFIC_CONFIG['sesoi_mase'])
DEFAULT_SESOI_WAPE = float(SCIENTIFIC_CONFIG['sesoi_wape'])

# Pairwise comparisons, derived from the registry. The dl/dw/pl abbreviations
# encoded the pre-rename names (data_lake, data_warehouse, polars)
# and stopped naming anything after it.
PREDICTIVE_PAIRS = paradigm_pairs()
LATENCY_PAIRS = paradigm_pairs()


def _median_hodges_lehmann(deltas: np.ndarray) -> float:
    n = len(deltas)
    if n == 0:
        return float('nan')
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append(0.5 * (deltas[i] + deltas[j]))
    return float(np.median(walsh))


def bootstrap_ci(values: np.ndarray, iters: int = DEFAULT_BOOTSTRAP_ITERS, seed: int = DEFAULT_SEED, ci: float = 0.95) -> Tuple[float, Tuple[float, float], str]:
    """Bootstrap CI (BCa with percentile fallback).

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


# Whether a larger value of the metric is the better outcome. Latency and the
# error measures improve downwards; R2 improves upwards. The decision function
# cannot know this, which is why it does not name a winner.
HIGHER_IS_BETTER = {'r2': True, 'mase': False, 'wape': False, 'latency': False}


def _decision_equivalence(ci_lo: float, ci_hi: float, delta: float) -> str:
    """Where the interval sits relative to +-delta, in neutral terms.

    Labelled by direction rather than by merit. The effect is measured as A - B
    (or log(A/B)), and whether a positive effect favours A depends on the metric:
    it does for R2, and it does not for latency, MASE or WAPE. Calling a positive
    effect 'superior' reported Dask as superior on the stages where it was
    slower, and inferior on the stages where it was faster.
    """
    if math.isnan(ci_lo) or math.isnan(ci_hi):
        return 'insufficient_data'
    if -delta <= ci_lo and ci_hi <= delta:
        return 'equivalent'
    if ci_lo > delta:
        return 'a_exceeds_b'
    if ci_hi < -delta:
        return 'b_exceeds_a'
    return 'inconclusive'


def _advantage(decision: str, metric: str, label_a: str, label_b: str) -> Optional[str]:
    """Which side the interval favours, once the metric's direction is known.

    None when the comparison does not name a winner: equivalent, inconclusive or
    without data.
    """
    if decision not in ('a_exceeds_b', 'b_exceeds_a'):
        return None
    higher_is_better = HIGHER_IS_BETTER[metric]
    a_is_greater = decision == 'a_exceeds_b'
    return label_a if a_is_greater == higher_is_better else label_b


def _load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _extract_fold_metrics(d: Dict) -> Dict[int, Dict[str, float]]:
    """Extracts per-fold test metrics from the baseline results JSON.

    Uses the 'best_baseline' model when available, falling back to 'naive_with_lag'.
    Extracts test_r2, test_mase, test_wape explicitly by key name.

    Compatible with both JSON structures:
      - DW/DL (v2+): baseline_model_results -> fold_X -> {best_baseline, naive_with_lag, ...}
      - PL (v1 legacy): baseline_models -> fold_X -> {naive_with_lag, cross_country_average, ...}
    """
    out: Dict[int, Dict[str, float]] = {}
    # Folds live under 'baseline_model_results', 'baseline_models', or directly in the dict
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
        # Uses best_baseline if available, otherwise falls back to naive_with_lag
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
    # Declared in PARADIGM_META: the paradigms write to distinct layouts.
    paths = baseline_results_paths()
    out = {}
    for arch, p in paths.items():
        d = _load_json(p)
        out[arch] = _extract_fold_metrics(d) if d else {}
    return out


def _paired_deltas_for_metric(pairs: Dict[str, Dict[int, Dict[str, float]]], metric: str, arch_a: str, arch_b: str) -> np.ndarray:
    a = pairs.get(arch_a, {})
    b = pairs.get(arch_b, {})
    common_ids = sorted(set(a.keys()) & set(b.keys()))
    deltas = []
    for fid in common_ids:
        va = a[fid].get(metric)
        vb = b[fid].get(metric)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            # A minus B, like the rest of the module: the docstring,
            # _decision_equivalence and paradigm_pairs assume that order, and the
            # latency path already uses log(A/B). With B-A the 'advantage' field
            # named the worse paradigm on every predictive metric.
            deltas.append(va - vb)
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
            point, (lo, hi), ci_method = bootstrap_ci(deltas, iters=args.bootstrap, seed=args.seed, ci=0.95)
            decision = _decision_equivalence(lo, hi, delta)
            advantage = _advantage(decision, metric, arch_a, arch_b)
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
                'advantage': advantage,
                'wilcoxon_p': wilcoxon_p,
                'hodges_lehmann': hl,
            }
        results[pair_key] = pair_results
    return results


def _load_benchmark_csv() -> Optional[pd.DataFrame]:
    p = get_absolute_output_path('outputs/benchmarks/architectural_benchmark_results.csv')
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _parse_latency_profile(s: Optional[str], default_total: float) -> Dict[str, float]:
    profile = {'setup': 0.15, 'processing': 0.10, 'baseline': 0.10, 'hierarchical': 0.05, 'total': default_total}
    if not s:
        return profile
    # A malformed item used to be dropped silently, and the run went on with
    # a SESOI different from the one the operator asked for -- which changes the
    # equivalence verdicts without leaving a trace.
    for part in s.split(','):
        if not part.strip():
            continue
        try:
            key, value = part.split(':')
            profile[key.strip().lower()] = float(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"Malformed SESOI profile at {part.strip()!r}: expected "
                f"'metric:value' separated by commas. Ignoring the item would "
                f"make the run decide equivalence with a threshold nobody "
                f"asked for."
            ) from exc
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
        for arch_a, arch_b in LATENCY_PAIRS:
            if arch_a not in vals.columns or arch_b not in vals.columns:
                continue
            pair_key = f"{arch_a}_vs_{arch_b}"
            x = vals[arch_a].to_numpy(dtype=float)
            y = vals[arch_b].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            lr = np.log(x[mask] / y[mask])
            if lr.size == 0:
                phase_results[pair_key] = {'status': 'insufficient_data'}
                continue
            point, (lo, hi), ci_method = bootstrap_ci(lr, iters=args.bootstrap, seed=args.seed, ci=0.95)
            delta_pct = profile.get(str(phase).lower(), profile['total'])
            delta_lr = math.log(1.0 + delta_pct)
            decision = _decision_equivalence(lo, hi, delta_lr)
            advantage = _advantage(decision, 'latency', arch_a, arch_b)
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
                'advantage': advantage,
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
    stats_dir = get_absolute_output_path('outputs/statistics')
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, 'equivalence_estimation.json'), 'w') as f:
        json.dump(obj, f, indent=2)
    if write_tex:
        lines = [
            '% Equivalence by Estimation (SESOI + CI) — Automatically generated',
            '\\begin{table}[htb]',
            '\\centering',
            '\\caption{Practical equivalence by estimation — prediction (3-way pairwise)}',
            '\\begin{tabular}{llrrrrll}',
            '\\toprule',
            'Pair & Metric & n & Est. & 95\\% CI & $\\delta$ & Decision & Advantage \\\\ ',
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
                adv = r.get('advantage') or '--'
                lines.append(
                    f"{_tex(pair_key)} & {_tex(m)} & {n} & {est:.3f} & "
                    f"[{ci[0]:.3f},{ci[1]:.3f}] & {d:.3f} & "
                    f"{_tex(dec)} & {_tex(adv)} \\\\")
        lines += [
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}',
            '',
            '\\begin{table}[htb]',
            '\\centering',
            '\\caption{Practical equivalence by estimation — latency (log‑ratio, 3-way pairwise)}',
            '\\begin{tabular}{llrrrrll}',
            '\\toprule',
            'Phase & Pair & n & Est. (LR) & 95\\% CI & $\\delta$(\\%) & Decision & Advantage \\\\ ',
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
                adv = r.get('advantage') or '--'
                lines.append(
                    f"{_tex(phase)} & {_tex(pair_key)} & {n} & "
                    f"{est:.3f} & [{ci[0]:.3f},{ci[1]:.3f}] & "
                    f"{d_pct:.1f} & {_tex(dec)} & {_tex(adv)} \\\\")
        lines += [
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}',
        ]
        with open(os.path.join(stats_dir, 'equivalence_estimation.tex'), 'w') as f:
            f.write("\n".join(lines))


def run(args: argparse.Namespace) -> int:
    predictive = _analyze_predictive_metrics(args)
    latency = _analyze_latency(args)
    # Extract n_pairs for the power note (maximum across all predictive pairs)
    n_pairs = max(
        (r.get('n_pairs', 0) for pair_data in predictive.values()
         if isinstance(pair_data, dict)
         for r in pair_data.values() if isinstance(r, dict)),
        default=0
    )
    # Derived from the registry. The previous list was a literal naming
    # pre-rename pairs, so the note described an analysis that was not run.
    pairs = [f'{a}_vs_{b}' for a, b in paradigm_pairs()]

    out = {
        'method': 'equivalence_by_estimation',
        'seed': args.seed,
        'bootstrap': args.bootstrap,
        'n_folds': n_pairs,
        'pairs': pairs,
        'power_note': power_note(n_pairs, pairs),
        'predictive': predictive,
        'latency': latency,
    }
    _save_outputs(out, write_tex=args.latex)
    print(json.dumps(out))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Equivalence by Estimation (SESOI + CI) for baselines and latency.')
    p.add_argument('--bootstrap', type=int, default=DEFAULT_BOOTSTRAP_ITERS, help=f'Bootstrap iterations (default: {DEFAULT_BOOTSTRAP_ITERS})')
    p.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Seed (default: 42)')
    p.add_argument('--r2-delta', type=float, default=DEFAULT_SESOI_R2, help=f'SESOI δ for R² (default: {DEFAULT_SESOI_R2})')
    p.add_argument('--mase-delta', type=float, default=DEFAULT_SESOI_MASE, help=f'SESOI δ for MASE (default: {DEFAULT_SESOI_MASE})')
    p.add_argument('--wape-delta', type=float, default=DEFAULT_SESOI_WAPE, help=f'SESOI δ for WAPE (default: {DEFAULT_SESOI_WAPE})')
    p.add_argument('--latency-delta', type=float, default=0.10, help='SESOI δ for TOTAL latency (default: 0.10)')
    p.add_argument('--latency-delta-profile', type=str, default='setup:0.15,processing:0.10,baseline:0.10,hierarchical:0.05,total:0.10',
                   help='Per-phase δ profile (e.g.: setup:0.15,processing:0.10,baseline:0.10,hierarchical:0.05,total:0.10)')
    p.add_argument('--latex', action='store_true', help='Also generate LaTeX tables')
    return p.parse_args()



def _tex(text) -> str:
    """Escape a text cell for LaTeX.

    Every text column in these two tables carries underscores: the pair key
    (dataframe_lib_vs_sql_engine), the phase (total_architectural), the
    decision (a_exceeds_b, insufficient_data) and the advantage, which is a
    paradigm name. None was escaped, so neither file compiled -- and the error
    surfaces to whoever assembles the paper, not to whoever ran the pipeline.
    """
    return (str(text).replace('\\', r'\textbackslash{}')
            .replace('_', r'\_').replace('%', r'\%')
            .replace('&', r'\&').replace('#', r'\#'))



def power_note(n_pairs: int, pairs: List[str]) -> str:
    """Prose accompanying the equivalence decisions.

    Kept as a function so the pair list it names can be checked against the
    registry. It used to be an f-string inside main() naming three pre-rename
    pairs, which no longer exist in the artifacts the note accompanies.
    """
    if n_pairs <= 0:
        return 'No data for power analysis.'
    return (
        f'n={n_pairs} folds (maximum without compromising temporal anti-leakage). '
        f'Pairwise analysis: {", ".join(pairs)}. '
        f'Paired Wilcoxon with n={n_pairs} has limited power (~30% for '
        f'd=0.5); main decision via bootstrap CI. An '
        f'"inconclusive" result is expected and reflects the available precision '
        f'(Lakens et al. 2018).'
    )


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()

