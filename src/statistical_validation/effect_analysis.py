#!/usr/bin/env python3
"""
Tamanhos de Efeito e Comparações Múltiplas para Benchmark Arquitetural

Calcula tamanhos de efeito e correções de múltiplas comparações para as
diferenças pareadas Data Lake (DL) vs Data Warehouse (DW), por fase e total.

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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_benchmark(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "run_id" not in df.columns and "rep" in df.columns:
        df = df.rename(columns={"rep": "run_id"})
    return df


def paired_vectors_for_phase(df: pd.DataFrame, phase: str) -> Tuple[np.ndarray, np.ndarray]:
    a = df[(df["phase"] == phase) & (df["architecture"] == "data_lake")]
    b = df[(df["phase"] == phase) & (df["architecture"] == "data_warehouse")]
    merged = pd.merge(
        a[["run_id", "duration_s"]],
        b[["run_id", "duration_s"]],
        on="run_id",
        suffixes=("_dl", "_dw"),
        how="inner",
    ).sort_values("run_id")
    return merged["duration_s_dl"].to_numpy(), merged["duration_s_dw"].to_numpy()


def paired_vectors_total(df: pd.DataFrame, exclude_phases: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    filt = ~df["phase"].isin(exclude_phases)
    sub = df[filt & df["architecture"].isin(["data_lake", "data_warehouse"])]
    tot = (
        sub.groupby(["run_id", "architecture"])['duration_s']
        .sum()
        .reset_index()
    )
    a = tot[tot["architecture"] == "data_lake"][["run_id", "duration_s"]]
    b = tot[tot["architecture"] == "data_warehouse"][["run_id", "duration_s"]]
    merged = pd.merge(a, b, on="run_id", suffixes=("_dl", "_dw"), how="inner").sort_values("run_id")
    return merged["duration_s_dl"].to_numpy(), merged["duration_s_dw"].to_numpy()


def cohens_dz(diff: np.ndarray) -> float:
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
    return float(np.mean(diff)) / sd if sd and sd > 0 else float("nan")


def hedges_g(d: float, n: int) -> float:
    if not math.isfinite(d):
        return d
    # small sample correction
    J = 1 - (3 / (4 * (n - 1) - 1)) if n > 2 else 1.0
    return d * J


def eta_squared_from_t(t_stat: float, n: int) -> float:
    # paired t: df = n-1
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
    # monotonicity (non-decreasing)
    for i in range(m - 2, -1, -1):
        ranked[i] = min(ranked[i], ranked[i + 1])
    return ranked.tolist()


def analyze(csv_path: str) -> Dict[str, Dict[str, float]]:
    df = load_benchmark(csv_path)
    phases = [p for p in sorted(df['phase'].unique()) if p != 'collection']
    res: Dict[str, Dict[str, float]] = {}
    power = TTestPower()

    # Por fase
    p_values = []
    keys = []
    tmp_stats = {}
    for p in phases:
        x, y = paired_vectors_for_phase(df, p)
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
        tmp_stats[p] = rec

    # Total (exclui collection)
    x, y = paired_vectors_total(df, exclude_phases=["collection"])
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

    # Múltiplas comparações
    if p_values:
        bonf = [min(1.0, p * len(p_values)) for p in p_values]
        fdr = benjamini_hochberg(p_values)
        for k, p_raw, p_bonf, p_fdr in zip(keys, p_values, bonf, fdr):
            res[k]['p_bonferroni'] = float(p_bonf)
            res[k]['p_fdr_bh'] = float(min(1.0, p_fdr))

    return res


def write_outputs(results: Dict[str, Dict[str, float]]) -> None:
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
    for phase, metrics in results.items():
        row = {'phase': phase}
        for c in cols:
            row[c] = metrics.get(c, float('nan'))
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main() -> None:
    csv_path = os.path.join(BASE_DIR, 'outputs', 'benchmarks', 'architectural_benchmark_results.csv')
    results = analyze(csv_path)
    write_outputs(results)
    print(json.dumps(results, indent=2))

    # Gera resumo interpretativo mínimo em Markdown
    md_path = os.path.join(STATS_DIR, 'effect_sizes_interpretation.md')
    lines = [
        "# Interpretação de Tamanhos de Efeito (DL vs DW pareado)",
        "",
    ]
    for phase, m in results.items():
        d = m.get('cohen_dz')
        eta2 = m.get('eta_squared')
        interp = (
            'negligible' if not isinstance(d, float) or not math.isfinite(d) else
            ('small' if abs(d) < 0.2 else 'medium' if abs(d) < 0.5 else 'large' if abs(d) < 0.8 else 'very_large')
        )
        lines.append(f"## {phase}")
        lines.append(f"- Cohen's d_z: {d:.4f} ({interp})")
        lines.append(f"- Eta-squared: {eta2:.4f}")
        lines.append("")
    os.makedirs(STATS_DIR, exist_ok=True)
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()


