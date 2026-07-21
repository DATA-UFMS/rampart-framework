#!/usr/bin/env python3
"""
Tamanhos de Efeito e Comparações Múltiplas para Benchmark Arquitetural

Calcula tamanhos de efeito e correções de múltiplas comparações para as
diferenças pareadas 3-way (DL vs DW, DL vs PL, DW vs PL), por fase e total.

Métricas:
  - Cohen's d (paired; d_z = mean(diff)/sd(diff))
  - Hedges' g (correção small-sample) opcional
  - Eta squared (para t pareado: eta2 = t^2 / (t^2 + df))
  - Wilcoxon signed-rank pareado, ao lado do t pareado
  - Correções: Bonferroni e FDR (Benjamini-Hochberg), por família de teste
  - Power observada via Monte Carlo no alpha corrigido (Hoenig & Heisey, 2001)

Saídas:
  - outputs/statistics/effect_sizes_summary.json
  - outputs/statistics/effect_sizes_summary.csv
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_SRC_DIR = os.path.join(BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import get_absolute_output_path
from core.paradigm_registry import paradigm_pairs
from core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED

DEFAULT_BOOTSTRAP_ITERS = int(SCIENTIFIC_CONFIG['bootstrap_iters'])
DEFAULT_SEED = RANDOM_SEED

STATS_DIR = get_absolute_output_path("outputs/statistics")

# Derived from the registry, so a fourth paradigm enters the comparison without
# this module being edited. The abbreviations dl/dw/pl are gone: they encoded the
# pre-rename names (data_lake, data_warehouse, polars) and named nothing after it.
ALL_PAIRS = paradigm_pairs()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_benchmark(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "run_id" not in df.columns and "rep" in df.columns:
        df = df.rename(columns={"rep": "run_id"})

    # A failed phase used to be recorded as duration_ns = -1, which reached this
    # point as a latency of -1e-09 and was averaged in as a measurement. The
    # benchmark now aborts instead, but a CSV produced before that must not be
    # consumed silently either.
    if "duration_s" in df.columns:
        invalid = df[~(df["duration_s"] > 0)]
        if not invalid.empty:
            offenders = invalid[["run_id", "phase", "architecture",
                                 "duration_s"]].head(5).to_dict("records")
            raise ValueError(
                f"{csv_path}: {len(invalid)} rows carry a non-positive "
                f"duration, which cannot be a latency measurement. A failed "
                f"phase must not enter the comparison: {offenders}"
            )
    return df


def paired_vectors_for_phase(df: pd.DataFrame, phase: str, arch_a: str, arch_b: str) -> Tuple[np.ndarray, np.ndarray]:
    a = df[(df["phase"] == phase) & (df["architecture"] == arch_a)]
    b = df[(df["phase"] == phase) & (df["architecture"] == arch_b)]
    la, lb = arch_a, arch_b
    merged = pd.merge(
        a[["run_id", "duration_s"]],
        b[["run_id", "duration_s"]],
        on="run_id",
        suffixes=(f"_{la}", f"_{lb}"),
        how="inner",
    ).sort_values("run_id")
    return merged[f"duration_s_{la}"].to_numpy(), merged[f"duration_s_{lb}"].to_numpy()


def paired_vectors_total(df: pd.DataFrame, exclude_phases: List[str], arch_a: str, arch_b: str) -> Tuple[np.ndarray, np.ndarray]:
    filt = ~df["phase"].isin(exclude_phases)
    sub = df[filt & df["architecture"].isin([arch_a, arch_b])]
    tot = (
        sub.groupby(["run_id", "architecture"])['duration_s']
        .sum()
        .reset_index()
    )
    la, lb = arch_a, arch_b
    a = tot[tot["architecture"] == arch_a][["run_id", "duration_s"]]
    b = tot[tot["architecture"] == arch_b][["run_id", "duration_s"]]
    merged = pd.merge(a, b, on="run_id", suffixes=(f"_{la}", f"_{lb}"), how="inner").sort_values("run_id")
    return merged[f"duration_s_{la}"].to_numpy(), merged[f"duration_s_{lb}"].to_numpy()


def cohens_dz(diff: np.ndarray) -> float:
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
    return float(np.mean(diff)) / sd if sd and sd > 0 else float("nan")


def hedges_g(d: float, n: int) -> float:
    if not math.isfinite(d):
        return d
    J = 1 - (3 / (4 * (n - 1) - 1)) if n > 2 else 1.0
    return d * J


def _effect_size_ci(diff: np.ndarray, n_boot: int = DEFAULT_BOOTSTRAP_ITERS,
                    seed: int = DEFAULT_SEED, ci: float = 0.95) -> Tuple[float, float]:
    """IC bootstrap percentil para Cohen's d_z."""
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diff), size=len(diff))
        d_boot = cohens_dz(diff[idx])
        if math.isfinite(d_boot):
            ds.append(d_boot)
    if not ds:
        return (float('nan'), float('nan'))
    alpha = (1 - ci) / 2
    return (float(np.quantile(ds, alpha)), float(np.quantile(ds, 1 - alpha)))


def eta_squared_from_t(t_stat: float, n: int) -> float:
    # t pareado: df = n-1
    if not math.isfinite(t_stat) or n <= 1:
        return float("nan")
    df = n - 1
    return float((t_stat * t_stat) / ((t_stat * t_stat) + df))


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values.

    Delegates to SciPy's reference implementation rather than reimplementing the
    step-up. The monotonicity pass has to run in the order of the sorted
    p-values; running it in the order the tests happen to be listed produces
    adjusted values below the raw ones, which the procedure cannot produce.

    A test without a p-value is not part of the family and comes back as NaN, so
    the family size reflects the tests actually performed.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if valid.any():
        out[valid] = stats.false_discovery_control(p[valid], method='bh')
    return np.minimum(out, 1.0).tolist()


def _observed_power_wilcoxon(n: int, effect_size: float, alpha: float,
                             n_sim: int = 5000,
                             seed: int = DEFAULT_SEED) -> float:
    """Power of the signed-rank test at an observed effect, by simulation.

    Observed power, not prospective: the effect comes from the same data as the
    test, which makes it a monotone transform of the p-value rather than
    independent evidence (Hoenig & Heisey, 2001). Reported for the record, and
    not to be read as support for a null result.

    The samples are drawn as paired differences directly. Drawing two
    independent groups and subtracting them gives differences with standard
    deviation sqrt(2), so the simulated paired effect would be effect_size/sqrt(2)
    -- at d_z = 0.8 and n = 10 that reports 0.33 where the power is 0.59.

    alpha must be the threshold the decision actually uses, so that reported
    power and reported significance refer to the same test.

      Hoenig, J. M., & Heisey, D. M. (2001). The Abuse of Power: The Pervasive
        Fallacy of Power Calculations for Data Analysis. The American
        Statistician, 55(1), 19-24.
    """
    # A non-finite effect propagates NaN into every simulated replicate, the
    # signed-rank test raises on all of them, and the rejection count stays at
    # zero -- so the record reports power 0.0, which asserts the test had no
    # chance of detecting anything. What is known is that the power is
    # undefined.
    if n < 4 or not np.isfinite(effect_size) or abs(effect_size) < 1e-10:
        return float('nan')
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sim):
        diff = rng.normal(effect_size, 1.0, n)
        try:
            p = stats.wilcoxon(diff).pvalue
            if p < alpha:
                rejections += 1
        except Exception:
            pass
    return rejections / n_sim


def _signed_rank(diff: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank on the paired differences.

    Reported alongside the paired t-test: it is the test the reported power
    refers to, and it does not assume normality of the differences, which n=10
    cannot establish.
    """
    try:
        res = stats.wilcoxon(diff)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float('nan'), float('nan')


def analyze(csv_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    df = load_benchmark(csv_path)
    phases = [p for p in sorted(df['phase'].unique()) if p != 'collection']
    results = {}

    # Por par
    for arch_a, arch_b in ALL_PAIRS:
        la, lb = arch_a, arch_b
        pair_key = f"{la}_vs_{lb}"
        res: Dict[str, Dict[str, float]] = {}

        # Por fase
        for p in phases:
            x, y = paired_vectors_for_phase(df, p, arch_a, arch_b)
            diff = x - y
            n = len(diff)
            if n < 2:
                continue
            # t-test pareado (one-sample em diff)
            t_stat, t_p = stats.ttest_rel(x, y)
            w_stat, w_p = _signed_rank(diff)
            d = cohens_dz(diff)
            g = hedges_g(d, n)
            eta2 = eta_squared_from_t(float(t_stat), n)
            d_ci_lo, d_ci_hi = _effect_size_ci(diff)
            rec = dict(
                n=n,
                mean_diff_s=float(np.mean(diff)),
                cohen_dz=float(d),
                cohen_dz_ci=(d_ci_lo, d_ci_hi),
                hedges_g=float(g),
                eta_squared=float(eta2),
                t_stat=float(t_stat),
                t_p=float(t_p),
                wilcoxon_stat=w_stat,
                wilcoxon_p=w_p,
            )
            res[p] = rec

        # Total (exclui collection)
        x, y = paired_vectors_total(df, exclude_phases=["collection"], arch_a=arch_a, arch_b=arch_b)
        diff = x - y
        n = len(diff)
        if n >= 2:
            t_stat, t_p = stats.ttest_rel(x, y)
            w_stat, w_p = _signed_rank(diff)
            d = cohens_dz(diff)
            g = hedges_g(d, n)
            eta2 = eta_squared_from_t(float(t_stat), n)
            d_ci_lo, d_ci_hi = _effect_size_ci(diff)
            res['total_architectural'] = dict(
                n=n,
                mean_diff_s=float(np.mean(diff)),
                cohen_dz=float(d),
                cohen_dz_ci=(d_ci_lo, d_ci_hi),
                hedges_g=float(g),
                eta_squared=float(eta2),
                t_stat=float(t_stat),
                t_p=float(t_p),
                wilcoxon_stat=w_stat,
                wilcoxon_p=w_p,
            )

        results[pair_key] = res

    all_p = []
    all_refs = []  # (pair_key, phase_key)
    for pk, res in results.items():
        for fk, rec in res.items():
            if 't_p' in rec:
                all_p.append(rec['t_p'])
                all_refs.append((pk, fk))

    if all_p:
        n_tests = len(all_p)
        # Recorded rather than left implicit: the family size determines the
        # threshold, and a reader cannot recover it from the adjusted values.
        alpha_family = 0.05 / n_tests

        bonf = [min(1.0, p * n_tests) for p in all_p]
        # Already clamped to 1, and NaN where the test produced no p-value;
        # min(1.0, nan) would report it as 1.0.
        fdr = benjamini_hochberg(all_p)

        # Wilcoxon is corrected over its own family. Mixing p-values from two
        # tests into one family would correct neither.
        all_w = [results[pk][fk].get('wilcoxon_p', float('nan'))
                 for pk, fk in all_refs]
        w_bonf = [min(1.0, p * n_tests) if np.isfinite(p) else float('nan')
                  for p in all_w]
        w_fdr = benjamini_hochberg(all_w)

        for i, (pk, fk) in enumerate(all_refs):
            rec = results[pk][fk]
            rec['family_size'] = n_tests
            rec['alpha_bonferroni'] = alpha_family
            rec['p_bonferroni'] = float(bonf[i])
            rec['p_fdr_bh'] = float(fdr[i])
            rec['wilcoxon_p_bonferroni'] = float(w_bonf[i])
            rec['wilcoxon_p_fdr_bh'] = float(w_fdr[i])
            # Computed here so it uses the threshold the decision uses, which is
            # only known once the family is closed.
            rec['observed_power'] = _observed_power_wilcoxon(
                rec['n'], rec['cohen_dz'], alpha=alpha_family)

    return results


def write_outputs(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    ensure_dir(STATS_DIR)
    json_path = os.path.join(STATS_DIR, 'effect_sizes_summary.json')
    csv_path = os.path.join(STATS_DIR, 'effect_sizes_summary.csv')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    rows = []
    cols = [
        'n', 'mean_diff_s', 'cohen_dz', 'hedges_g', 'eta_squared',
        't_stat', 't_p', 'p_bonferroni', 'p_fdr_bh',
        'wilcoxon_stat', 'wilcoxon_p', 'wilcoxon_p_bonferroni',
        'wilcoxon_p_fdr_bh', 'family_size', 'alpha_bonferroni',
        'observed_power'
    ]
    for pair_key, pair_results in results.items():
        for phase, metrics in pair_results.items():
            row = {'pair': pair_key, 'phase': phase}
            for c in cols:
                row[c] = metrics.get(c, float('nan'))
            rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Effect sizes 3-way pairwise")
    parser.add_argument("--csv", default=get_absolute_output_path(
                            'outputs/benchmarks/architectural_benchmark_results.csv'),
                        help="Caminho para o CSV do benchmark")
    args = parser.parse_args()
    csv_path = args.csv
    results = analyze(csv_path)
    write_outputs(results)
    print(json.dumps(results, indent=2))

    # Gera resumo interpretativo mínimo em Markdown
    md_path = os.path.join(STATS_DIR, 'effect_sizes_interpretation.md')
    lines = [
        "# Interpretação de Tamanhos de Efeito (comparações pareadas 3-way)",
        "",
    ]
    for pair_key, pair_results in results.items():
        lines.append(f"## {pair_key.upper()}")
        lines.append("")
        for phase, m in pair_results.items():
            d = m.get('cohen_dz')
            eta2 = m.get('eta_squared')
            interp = (
                'negligible' if not isinstance(d, float) or not math.isfinite(d) else
                ('negligible' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large')
            )
            lines.append(f"### {phase}")
            lines.append(f"- Cohen's d_z: {d:.4f} ({interp})")
            lines.append(f"- Eta-squared: {eta2:.4f}")
            lines.append("")
    os.makedirs(STATS_DIR, exist_ok=True)
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()


