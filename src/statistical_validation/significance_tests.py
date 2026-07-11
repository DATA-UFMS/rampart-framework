#!/usr/bin/env python3
"""
Testes de Significância Estatística para Benchmark Arquitetural (3-Way)

Executa testes estatísticos pareados entre os 3 paradigmas arquiteturais:
Data Lake (DL), Data Warehouse (DW) e Polars DataFrame (PL) em comparações
pairwise (DL×DW, DL×PL, DW×PL). Para cada fase e para o total arquitetural,
o script alinha execuções por run_id e aplica:

- Paired t-test (diferença de médias entre arquiteturas)
- Wilcoxon signed-rank (não-paramétrico)
- Testes de normalidade (Shapiro-Wilk e Anderson-Darling) no vetor de diferenças
- Bootstrap (intervalos de confiança) para diferença de médias e speedup

Saídas:
  - outputs/statistics/significance_summary.json (estrutura aninhada por pair)
  - outputs/statistics/significance_summary.csv (estrutura plana com coluna 'pair')
  - outputs/statistics/significance_summary.md

Definições:
  - Diferença (diff_s) = arch_a - arch_b (em segundos)
  - Speedup (arch_b_vs_arch_a) = arch_a_mean / arch_b_mean (maior que 1 favorece arch_b)
  - Total arquitetural: soma das fases específicas por arquitetura
    (exclui 'collection' que é comum a todas)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Raiz do projeto para importar configuração
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
try:
    from core.scientific_config import SCIENTIFIC_CONFIG
    DEFAULT_BOOTSTRAP_ITERS = int(SCIENTIFIC_CONFIG.get('bootstrap_iters', 3000))
except Exception:
    DEFAULT_BOOTSTRAP_ITERS = 3000


# Pares de arquiteturas para comparação (3-way)
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


def paired_vectors_for_phase(
    df: pd.DataFrame, phase: str, arch_a: str = "data_lake", arch_b: str = "data_warehouse"
) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna vetores alinhados para dois arquiteturas por run_id em uma fase."""
    a = df[(df["phase"] == phase) & (df["architecture"] == arch_a)]
    b = df[(df["phase"] == phase) & (df["architecture"] == arch_b)]

    pair_key = (arch_a, arch_b)
    if pair_key in PAIR_LABELS:
        suffix_a, suffix_b = PAIR_LABELS[pair_key]
    else:
        suffix_a, suffix_b = ("a", "b")

    merged = pd.merge(
        a[["run_id", "duration_s"]],
        b[["run_id", "duration_s"]],
        on="run_id",
        suffixes=(f"_{suffix_a}", f"_{suffix_b}"),
        how="inner",
    ).sort_values("run_id")

    col_a = f"duration_s_{suffix_a}"
    col_b = f"duration_s_{suffix_b}"
    return merged[col_a].to_numpy(), merged[col_b].to_numpy()


def paired_vectors_total(
    df: pd.DataFrame,
    arch_a: str = "data_lake",
    arch_b: str = "data_warehouse",
    exclude_phases: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Soma durations por run_id e arquitetura nas fases especificadas (excluindo collection)."""
    if exclude_phases is None:
        exclude_phases = ["collection"]
    filt = ~df["phase"].isin(exclude_phases)
    sub = df[filt & df["architecture"].isin([arch_a, arch_b])]
    tot = (
        sub.groupby(["run_id", "architecture"])['duration_s']
        .sum()
        .reset_index()
    )
    a = tot[tot["architecture"] == arch_a][["run_id", "duration_s"]]
    b = tot[tot["architecture"] == arch_b][["run_id", "duration_s"]]

    pair_key = (arch_a, arch_b)
    if pair_key in PAIR_LABELS:
        suffix_a, suffix_b = PAIR_LABELS[pair_key]
    else:
        suffix_a, suffix_b = ("a", "b")

    merged = pd.merge(a, b, on="run_id", suffixes=(f"_{suffix_a}", f"_{suffix_b}"), how="inner").sort_values("run_id")
    col_a = f"duration_s_{suffix_a}"
    col_b = f"duration_s_{suffix_b}"
    return merged[col_a].to_numpy(), merged[col_b].to_numpy()


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    label_a: str = "dl",
    label_b: str = "dw",
    iters: int = DEFAULT_BOOTSTRAP_ITERS,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap 95% CI para diferença de médias (x-y) e speedup (x_mean/y_mean)."""
    if rng is None:
        rng = np.random.default_rng(SCIENTIFIC_CONFIG.get('random_seed', 42))
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
        f"speedup_{label_b}_vs_{label_a}_ci95": (float(lo_r), float(hi_r)),
    }


def run_tests(
    x: np.ndarray,
    y: np.ndarray,
    label_a: str = "dl",
    label_b: str = "dw",
    bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS,
) -> Dict[str, float]:
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
    ci = bootstrap_ci(x, y, label_a=label_a, label_b=label_b, iters=bootstrap_iters)

    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    speedup_key = f"speedup_{label_b}_vs_{label_a}"
    speedup_ci_key = f"speedup_{label_b}_vs_{label_a}_ci95"

    res.update(
        dict(
            n=len(diff),
            **{f"mean_{label_a}_s": mean_x},
            **{f"mean_{label_b}_s": mean_y},
            mean_diff_s=float(np.mean(diff)),
            shapiro_W=float(sh_w),
            shapiro_p=float(sh_p),
            anderson_stat=ad_stat,
            t_stat=float(t_stat),
            t_p=float(t_p),
            wilcoxon_stat=float(w_stat),
            wilcoxon_p=float(w_p),
            **{speedup_key: mean_x / mean_y if mean_y > 0 else float("inf")},
            diff_mean_ci95_lo=ci["diff_mean_ci95"][0],
            diff_mean_ci95_hi=ci["diff_mean_ci95"][1],
            **{f"{speedup_ci_key}_lo": ci[speedup_ci_key][0]},
            **{f"{speedup_ci_key}_hi": ci[speedup_ci_key][1]},
        )
    )
    return res


def analyze(csv_path: str, bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analisa todas as comparações pairwise (DL×DW, DL×PL, DW×PL)."""
    df = load_benchmark(csv_path)
    phases = sorted([p for p in df['phase'].unique() if p != 'collection'])
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Para cada par de arquiteturas
    for arch_a, arch_b in ALL_PAIRS:
        la, lb = PAIR_LABELS[(arch_a, arch_b)]
        pair_key = f"{la}_vs_{lb}"
        pair_results: Dict[str, Dict[str, float]] = {}

        # Por fase
        for p in phases:
            x, y = paired_vectors_for_phase(df, p, arch_a, arch_b)
            if len(x) >= 2 and len(y) >= 2:
                pair_results[p] = run_tests(x, y, label_a=la, label_b=lb, bootstrap_iters=bootstrap_iters)

        # Total arquitetural (exclui collection)
        x_tot, y_tot = paired_vectors_total(df, arch_a, arch_b, exclude_phases=["collection"])
        if len(x_tot) >= 2 and len(y_tot) >= 2:
            pair_results["total_architectural"] = run_tests(
                x_tot, y_tot, label_a=la, label_b=lb, bootstrap_iters=bootstrap_iters
            )

        results[pair_key] = pair_results

    return results


def _format_markdown_table(rows: List[Dict[str, float]], cols: List[str], include_pair: bool = False) -> str:
    """Cria uma tabela Markdown estável sem depender de tabulate."""
    headers = (['pair'] if include_pair else []) + ['phase'] + cols
    out = ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        line = []
        if include_pair:
            line.append(str(r.get('pair', '')))
        line.append(str(r.get('phase', '')))
        for c in cols:
            v = r.get(c, float('nan'))
            if isinstance(v, float):
                line.append(f"{v:.4g}")
            else:
                line.append(str(v))
        out.append('|' + '|'.join(line) + '|')
    return '\n'.join(out) + '\n'


def write_outputs(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    ensure_dir(RESULTS_DIR)
    json_path = os.path.join(RESULTS_DIR, "significance_summary.json")
    csv_path = os.path.join(RESULTS_DIR, "significance_summary.csv")
    md_path = os.path.join(RESULTS_DIR, "significance_summary.md")

    # JSON: estrutura aninhada por pair
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for pair_key, pair_results in results.items():
        for phase, metrics in pair_results.items():
            row = {"pair": pair_key, "phase": phase}
            row.update(metrics)
            rows.append(row)

    possible_cols = [
        "n",
        "mean_dl_s",
        "mean_dw_s",
        "mean_pl_s",
        "mean_diff_s",
        "speedup_dw_vs_dl",
        "speedup_pl_vs_dl",
        "speedup_pl_vs_dw",
        "diff_mean_ci95_lo",
        "diff_mean_ci95_hi",
        "speedup_dw_vs_dl_ci95_lo",
        "speedup_dw_vs_dl_ci95_hi",
        "speedup_pl_vs_dl_ci95_lo",
        "speedup_pl_vs_dl_ci95_hi",
        "speedup_pl_vs_dw_ci95_lo",
        "speedup_pl_vs_dw_ci95_hi",
        "shapiro_W",
        "shapiro_p",
        "anderson_stat",
        "t_stat",
        "t_p",
        "wilcoxon_stat",
        "wilcoxon_p",
    ]
    # Determina quais colunas realmente existem
    cols_present = set()
    for row in rows:
        cols_present.update(k for k in row.keys() if k not in ["pair", "phase"])
    cols = [c for c in possible_cols if c in cols_present]

    # CSV
    df = pd.DataFrame(rows)
    df_csv = df[["pair", "phase"] + cols]
    df_csv.to_csv(csv_path, index=False)

    # Markdown (colunas selecionadas e arredondadas)
    md_cols = [
        "n",
        "mean_diff_s",
        "shapiro_p",
        "t_stat",
        "t_p",
        "wilcoxon_stat",
        "wilcoxon_p",
    ]
    # Filtra apenas colunas que existem
    md_cols = [c for c in md_cols if c in cols_present]

    with open(md_path, "w") as fmd:
        fmd.write("# Resumo de Significância Estatística (3-Way Pairwise)\n\n")
        for pair_key in sorted(results.keys()):
            fmd.write(f"## {pair_key.upper()}\n\n")
            pair_rows = [r for r in rows if r["pair"] == pair_key]
            fmd.write(_format_markdown_table(pair_rows, md_cols, include_pair=False))
            fmd.write("\n")

    # LaTeX
    try:
        for pair_key in sorted(results.keys()):
            pair_rows = [r for r in rows if r["pair"] == pair_key]
            pair_df = pd.DataFrame(pair_rows)
            latex_cols = ["phase"] + [c for c in md_cols if c in pair_df.columns]
            df_latex = pair_df[latex_cols].copy()
            for c in df_latex.columns:
                if c != "phase" and pd.api.types.is_numeric_dtype(df_latex[c]):
                    df_latex[c] = df_latex[c].astype(float).round(4)
            latex = df_latex.to_latex(index=False, escape=True)
            tex_file = os.path.join(RESULTS_DIR, f"significance_summary_{pair_key}.tex")
            with open(tex_file, "w") as ftx:
                ftx.write(latex)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Testes de significância estatística para benchmark arquitetural")
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


