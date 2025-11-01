#!/usr/bin/env python3
"""
Statistical Significance Tests for Architectural Benchmark

Executa testes estatísticos pareados entre Data Lake e Data Warehouse usando
os tempos do benchmark. Para cada fase e para o total arquitetural, o script
alinha execuções por run_id e aplica:

- Paired t-test (diferença de médias DL vs DW)
- Wilcoxon signed-rank (não-paramétrico)
- Testes de normalidade (Shapiro-Wilk e Anderson-Darling) no vetor de diferenças
- Bootstrap (intervalos de confiança) para diferença de médias e speedup

Saídas:
  - outputs/statistics/significance_summary.json
  - outputs/statistics/significance_summary.csv

Definições:
  - Diferença (diff_s) = DL - DW (em segundos)
  - Speedup (DW_vs_DL) = DL_mean / DW_mean (maior que 1 favorece DW)
  - Total arquitetural: soma das fases específicas por arquitetura
    (exclui 'collection' que é comum a ambas)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Raiz do projeto para importar configuração científica
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
try:
    from src.core.scientific_config import SCIENTIFIC_CONFIG  # type: ignore
    DEFAULT_BOOTSTRAP_ITERS = int(SCIENTIFIC_CONFIG.get('bootstrap_iters', 3000))
except Exception:
    DEFAULT_BOOTSTRAP_ITERS = 3000


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "outputs",
    "statistics",
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_benchmark(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "run_id" not in df.columns and "rep" in df.columns:
        df = df.rename(columns={"rep": "run_id"})
    return df


def paired_vectors_for_phase(df: pd.DataFrame, phase: str) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna vetores alinhados DL e DW por run_id para uma fase."""
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


def paired_vectors_total(df: pd.DataFrame, exclude_phases: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Soma durations por run_id e arquitetura nas fases especificadas (excluindo collection)."""
    if exclude_phases is None:
        exclude_phases = ["collection"]
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


def bootstrap_ci(x: np.ndarray, y: np.ndarray, iters: int = DEFAULT_BOOTSTRAP_ITERS, rng: Optional[np.random.Generator] = None) -> Dict[str, Tuple[float, float]]:
    """Bootstrap 95% CI para diferença de médias (DL-DW) e speedup (DL_mean/DW_mean)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    diffs = np.empty(iters)
    ratios = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        diffs[i] = float(np.mean(xb) - np.mean(yb))
        yb_mean = float(np.mean(yb))
        ratios[i] = float(np.mean(xb)) / yb_mean if yb_mean > 0 else math.inf
    lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])
    lo_r, hi_r = np.percentile(ratios, [2.5, 97.5])
    return {
        "diff_mean_ci95": (float(lo_d), float(hi_d)),
        "speedup_dw_vs_dl_ci95": (float(lo_r), float(hi_r)),
    }


def run_tests(x: np.ndarray, y: np.ndarray, bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS) -> Dict[str, float]:
    """Executa testes de significância e normalidade sobre as diferenças (x - y)."""
    res: Dict[str, float] = {}
    diff = x - y
    # Normalidade
    if len(diff) >= 3:
        sh_w, sh_p = stats.shapiro(diff)
    else:
        sh_w, sh_p = float("nan"), float("nan")
    try:
        ad = stats.anderson(diff, dist='norm')
        ad_stat = float(ad.statistic)
    except Exception:
        ad_stat = float("nan")

    # Testes pareados
    try:
        t_stat, t_p = stats.ttest_rel(x, y)
    except Exception:
        t_stat, t_p = float("nan"), float("nan")
    try:
        w_stat, w_p = stats.wilcoxon(x, y)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")

    # Bootstrap CIs
    ci = bootstrap_ci(x, y, iters=bootstrap_iters)

    res.update(
        dict(
            n=len(diff),
            mean_dl_s=float(np.mean(x)),
            mean_dw_s=float(np.mean(y)),
            mean_diff_s=float(np.mean(diff)),
            shapiro_W=float(sh_w),
            shapiro_p=float(sh_p),
            anderson_stat=ad_stat,
            t_stat=float(t_stat),
            t_p=float(t_p),
            wilcoxon_stat=float(w_stat),
            wilcoxon_p=float(w_p),
            speedup_dw_vs_dl=(float(np.mean(x)) / float(np.mean(y))) if float(np.mean(y)) > 0 else float("inf"),
            diff_mean_ci95_lo=ci["diff_mean_ci95"][0],
            diff_mean_ci95_hi=ci["diff_mean_ci95"][1],
            speedup_ci95_lo=ci["speedup_dw_vs_dl_ci95"][0],
            speedup_ci95_hi=ci["speedup_dw_vs_dl_ci95"][1],
        )
    )
    return res


def analyze(csv_path: str, bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS) -> Dict[str, Dict[str, float]]:
    df = load_benchmark(csv_path)
    phases = sorted([p for p in df['phase'].unique() if p != 'collection'])
    results: Dict[str, Dict[str, float]] = {}

    # Por fase
    for p in phases:
        x, y = paired_vectors_for_phase(df, p)
        if len(x) >= 2 and len(y) >= 2:
            results[p] = run_tests(x, y, bootstrap_iters=bootstrap_iters)

    # Total arquitetural (exclui collection)
    x_tot, y_tot = paired_vectors_total(df, exclude_phases=["collection"])
    if len(x_tot) >= 2 and len(y_tot) >= 2:
        results["total_architectural"] = run_tests(x_tot, y_tot, bootstrap_iters=bootstrap_iters)

    return results


def _format_markdown_table(rows: List[Dict[str, float]], cols: List[str]) -> str:
    """Cria uma tabela Markdown estável sem depender de tabulate."""
    headers = ['phase'] + cols
    # Header
    out = ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|']
    # Rows
    for r in rows:
        line = [str(r.get('phase', ''))]
        for c in cols:
            v = r.get(c, float('nan'))
            if isinstance(v, float):
                line.append(f"{v:.4g}")
            else:
                line.append(str(v))
        out.append('|' + '|'.join(line) + '|')
    return '\n'.join(out) + '\n'


def write_outputs(results: Dict[str, Dict[str, float]]) -> None:
    ensure_dir(RESULTS_DIR)
    json_path = os.path.join(RESULTS_DIR, "significance_summary.json")
    csv_path = os.path.join(RESULTS_DIR, "significance_summary.csv")
    md_path = os.path.join(RESULTS_DIR, "significance_summary.md")
    tex_path = os.path.join(RESULTS_DIR, "significance_summary.tex")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Tabular CSV ordenado
    cols = [
        "n",
        "mean_dl_s",
        "mean_dw_s",
        "mean_diff_s",
        "speedup_dw_vs_dl",
        "diff_mean_ci95_lo",
        "diff_mean_ci95_hi",
        "speedup_ci95_lo",
        "speedup_ci95_hi",
        "shapiro_W",
        "shapiro_p",
        "anderson_stat",
        "t_stat",
        "t_p",
        "wilcoxon_stat",
        "wilcoxon_p",
    ]
    rows = []
    for phase, metrics in results.items():
        row = {"phase": phase}
        for c in cols:
            row[c] = metrics.get(c, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # Markdown (columns curated and rounded)
    md_cols = [
        "n",
        "mean_dl_s",
        "mean_dw_s",
        "mean_diff_s",
        "speedup_dw_vs_dl",
        "diff_mean_ci95_lo",
        "diff_mean_ci95_hi",
        "speedup_ci95_lo",
        "speedup_ci95_hi",
        "t_stat",
        "t_p",
        "p_bonferroni",
        "p_fdr_bh",
        "shapiro_p",
    ]
    with open(md_path, "w") as fmd:
        fmd.write("# Statistical Significance Summary\n\n")
        fmd.write(_format_markdown_table(rows, md_cols))

    # LaTeX table (uses pandas to_latex)
    try:
        df_latex = df[["phase"] + md_cols].copy()
        # Round selected float columns for readability
        for c in df_latex.columns:
            if c != "phase" and pd.api.types.is_numeric_dtype(df_latex[c]):
                df_latex[c] = df_latex[c].astype(float).round(4)
        latex = df_latex.to_latex(index=False, escape=True)
        with open(tex_path, "w") as ftx:
            ftx.write(latex)
    except Exception:
        # Silent fallback if LaTeX export is not possible
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical significance tests for architectural benchmark")
    p.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "outputs",
            "benchmarks",
            "architectural_benchmark_results.csv",
        ),
        help="Caminho para o CSV do benchmark",
    )
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_ITERS, help=f"Iterações bootstrap (default={DEFAULT_BOOTSTRAP_ITERS})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = analyze(args.csv, bootstrap_iters=args.bootstrap)
    write_outputs(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


