#!/usr/bin/env python3
"""
Tamanhos de Efeito e Comparações Múltiplas para Benchmark Arquitetural

Calcula tamanhos de efeito e correções de múltiplas comparações para as
diferenças pareadas 3-way (DL vs DW, DL vs PL, DW vs PL), por fase e total.

Métricas:
  - Cohen's d (paired; d_z = mean(diff)/sd(diff))
  - Hedges' g (correção small-sample) opcional
  - Eta squared (para t pareado: eta2 = t^2 / (t^2 + df))
  - Correções: Bonferroni e FDR (Benjamini-Hochberg)
  - Power analysis post-hoc (aproximação via one-sample sobre diff):
       statsmodels.stats.power.TTestPower

Saídas:
  - outputs/statistics/effect_sizes_summary.json
  - outputs/statistics/effect_sizes_summary.csv
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STATS_DIR = os.path.join(BASE_DIR, "outputs", "statistics")

ALL_PAIRS = [
    ("data_lake", "data_warehouse"),
    ("data_lake", "polars_dataframe"),
    ("data_warehouse", "polars_dataframe"),
]
PAIR_LABELS = {
    ("data_lake", "data_warehouse"): ("dl", "dw"),
    ("data_lake", "polars_dataframe"): ("dl", "pl"),
    ("data_warehouse", "polars_dataframe"): ("dw", "pl"),
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_benchmark(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "run_id" not in df.columns and "rep" in df.columns:
        df = df.rename(columns={"rep": "run_id"})
    return df


def paired_vectors_for_phase(df: pd.DataFrame, phase: str, arch_a: str, arch_b: str) -> Tuple[np.ndarray, np.ndarray]:
    a = df[(df["phase"] == phase) & (df["architecture"] == arch_a)]
    b = df[(df["phase"] == phase) & (df["architecture"] == arch_b)]
    la, lb = PAIR_LABELS[(arch_a, arch_b)]
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
    la, lb = PAIR_LABELS[(arch_a, arch_b)]
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
    # Correção para amostras pequenas
    J = 1 - (3 / (4 * (n - 1) - 1)) if n > 2 else 1.0
    return d * J


def eta_squared_from_t(t_stat: float, n: int) -> float:
    # t pareado: df = n-1
    if not math.isfinite(t_stat) or n <= 1:
        return float("nan")
    df = n - 1
    return float((t_stat * t_stat) / ((t_stat * t_stat) + df))


def benjamini_hochberg(pvals: List[float], alpha: float = 0.05) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(m)
    for rank, idx in enumerate(order, start=1):
        ranked[idx] = pvals[idx] * m / rank
    # Monotonicidade (não-decrescente)
    for i in range(m - 2, -1, -1):
        ranked[i] = min(ranked[i], ranked[i + 1])
    return ranked.tolist()


def analyze(csv_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    df = load_benchmark(csv_path)
    phases = [p for p in sorted(df['phase'].unique()) if p != 'collection']
    results = {}
    power = TTestPower()

    # Loop over each pair
    for arch_a, arch_b in ALL_PAIRS:
        la, lb = PAIR_LABELS[(arch_a, arch_b)]
        pair_key = f"{la}_vs_{lb}"
        res: Dict[str, Dict[str, float]] = {}

        # Por fase
        p_values = []
        keys = []
        for p in phases:
            x, y = paired_vectors_for_phase(df, p, arch_a, arch_b)
            diff = x - y
            n = len(diff)
            if n < 2:
                continue
            # t-test pareado (one-sample em diff)
            t_stat, t_p = stats.ttest_rel(x, y)
            d = cohens_dz(diff)
            g = hedges_g(d, n)
            eta2 = eta_squared_from_t(float(t_stat), n)
            # power post-hoc (aprox.)
            try:
                power_est = float(power.solve_power(effect_size=d, nobs=n, alpha=0.05, alternative='two-sided'))
            except Exception:
                power_est = float("nan")
            rec = dict(
                n=n,
                mean_diff_s=float(np.mean(diff)),
                cohen_dz=float(d),
                hedges_g=float(g),
                eta_squared=float(eta2),
                t_stat=float(t_stat),
                t_p=float(t_p),
                power_est=power_est,
            )
            res[p] = rec
            p_values.append(float(t_p))
            keys.append(p)

        # Total (exclui collection)
        x, y = paired_vectors_total(df, exclude_phases=["collection"], arch_a=arch_a, arch_b=arch_b)
        diff = x - y
        n = len(diff)
        if n >= 2:
            t_stat, t_p = stats.ttest_rel(x, y)
            d = cohens_dz(diff)
            g = hedges_g(d, n)
            eta2 = eta_squared_from_t(float(t_stat), n)
            try:
                power_est = float(power.solve_power(effect_size=d, nobs=n, alpha=0.05, alternative='two-sided'))
            except Exception:
                power_est = float("nan")
            res['total_architectural'] = dict(
                n=n,
                mean_diff_s=float(np.mean(diff)),
                cohen_dz=float(d),
                hedges_g=float(g),
                eta_squared=float(eta2),
                t_stat=float(t_stat),
                t_p=float(t_p),
                power_est=power_est,
            )
            p_values.append(float(t_p))
            keys.append('total_architectural')

        # Múltiplas comparações (Bonferroni e FDR por pair)
        if p_values:
            bonf = [min(1.0, p * len(p_values)) for p in p_values]
            fdr = benjamini_hochberg(p_values)
            for k, p_raw, p_bonf, p_fdr in zip(keys, p_values, bonf, fdr):
                res[k]['p_bonferroni'] = float(p_bonf)
                res[k]['p_fdr_bh'] = float(min(1.0, p_fdr))

        results[pair_key] = res

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
        't_stat', 't_p', 'p_bonferroni', 'p_fdr_bh', 'power_est'
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
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, 'outputs', 'benchmarks', 'architectural_benchmark_results.csv'),
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


