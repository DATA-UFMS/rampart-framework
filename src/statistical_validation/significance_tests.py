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
# No fallback: a local default would let the analysis run with a resample count
# other than the configured one, which is how the reported figure drifts from
# the executed one. Without the configuration the run is not reproducible.
from core.config import get_absolute_output_path
from core.paradigm_registry import COMPARABLE_PHASES, paradigm_pairs
from core.scientific_config import SCIENTIFIC_CONFIG

DEFAULT_BOOTSTRAP_ITERS = int(SCIENTIFIC_CONFIG['bootstrap_iters'])


# Derived from the registry, so a fourth paradigm enters the comparison without
# this module being edited. The abbreviations dl/dw/pl are gone: they encoded the
# pre-rename names (data_lake, data_warehouse, polars) and named nothing after it.
ALL_PAIRS = paradigm_pairs()


RESULTS_DIR = get_absolute_output_path("outputs/statistics")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Ordem de leitura das colunas do CSV. É ordenação, não filtragem: a versão
# anterior era uma lista branca de nomes pré-rename e descartava em silêncio
# tudo o que não reconhecia -- as médias por paradigma e todos os speedups com
# seus IC95, que são justamente as colunas cujo nome contém o paradigma.
# Sobreviviam quatro: n, mean_diff_s e o IC da diferença.
_COLUMN_ORDER = ("n", "mean_", "speedup_", "diff_mean_ci95_",
                 "shapiro_", "anderson_", "t_", "wilcoxon_")


def column_rank(name: str) -> tuple:
    """Posição de leitura de uma coluna; desconhecidas vão ao fim, em ordem."""
    for index, prefix in enumerate(_COLUMN_ORDER):
        if name == prefix or name.startswith(prefix):
            return (index, name)
    return (len(_COLUMN_ORDER), name)


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


def paired_vectors_for_phase(
    df: pd.DataFrame, phase: str, arch_a: str, arch_b: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna vetores alinhados para dois arquiteturas por run_id em uma fase."""
    a = df[(df["phase"] == phase) & (df["architecture"] == arch_a)]
    b = df[(df["phase"] == phase) & (df["architecture"] == arch_b)]

    suffix_a, suffix_b = arch_a, arch_b

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
    arch_a: str,
    arch_b: str,
    exclude_phases: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Soma durations por run_id e arquitetura nas fases comparáveis.

    Quais são elas vem de core.paradigm_registry: quatro arquivos enumeravam a
    política, e um deles já havia esquecido de aplicá-la.
    """
    if exclude_phases is None:
        filt = df["phase"].isin(COMPARABLE_PHASES)
    else:
        filt = ~df["phase"].isin(exclude_phases)
    sub = df[filt & df["architecture"].isin([arch_a, arch_b])]
    tot = (
        sub.groupby(["run_id", "architecture"])['duration_s']
        .sum()
        .reset_index()
    )
    a = tot[tot["architecture"] == arch_a][["run_id", "duration_s"]]
    b = tot[tot["architecture"] == arch_b][["run_id", "duration_s"]]

    suffix_a, suffix_b = arch_a, arch_b

    merged = pd.merge(a, b, on="run_id", suffixes=(f"_{suffix_a}", f"_{suffix_b}"), how="inner").sort_values("run_id")
    col_a = f"duration_s_{suffix_a}"
    col_b = f"duration_s_{suffix_b}"
    return merged[col_a].to_numpy(), merged[col_b].to_numpy()


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    label_a: str,
    label_b: str,
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
    label_a: str,
    label_b: str,
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

    # O signed-rank descarta diferenças exatamente nulas, então o n do teste
    # não é o número de pares. É esse n que determina o menor p alcançável
    # (2/2^n bilateral), e o piso reportado saía calculado sobre os pares:
    # com três empates em dez, o piso real é oito vezes maior.
    n_nonzero = int(np.count_nonzero(np.asarray(diff, dtype=float)))

    res.update(
        dict(
            n=len(diff),
            n_nonzero_diffs=n_nonzero,
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
    phases = sorted([p for p in df['phase'].unique()
                     if p in COMPARABLE_PHASES])
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Para cada par de arquiteturas
    for arch_a, arch_b in ALL_PAIRS:
        la, lb = arch_a, arch_b
        pair_key = f"{la}_vs_{lb}"
        pair_results: Dict[str, Dict[str, float]] = {}

        # Por fase
        for p in phases:
            x, y = paired_vectors_for_phase(df, p, arch_a, arch_b)
            if len(x) >= 2 and len(y) >= 2:
                pair_results[p] = run_tests(x, y, label_a=la, label_b=lb, bootstrap_iters=bootstrap_iters)

        # Total arquitetural sobre as fases comparáveis; a lista vem do registro.
        x_tot, y_tot = paired_vectors_total(df, arch_a, arch_b)
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

    cols_present = set()
    for row in rows:
        cols_present.update(k for k in row.keys() if k not in ["pair", "phase"])
    cols = sorted(cols_present, key=column_rank)

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
        default=get_absolute_output_path(
            "outputs/benchmarks/architectural_benchmark_results.csv"),
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


