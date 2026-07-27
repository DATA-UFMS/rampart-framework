#!/usr/bin/env python3
"""
Statistical Significance Tests for the Architectural Benchmark (3-Way)

Runs paired statistical tests between the 3 architectural paradigms:
Data Lake (DL), Data Warehouse (DW) and Polars DataFrame (PL) in pairwise
comparisons (DL×DW, DL×PL, DW×PL). For each phase and for the architectural
total, the script aligns runs by run_id and applies:

- Paired t-test (difference of means between architectures)
- Wilcoxon signed-rank (non-parametric)
- Normality tests (Shapiro-Wilk and Anderson-Darling) on the difference vector
- Bootstrap (confidence intervals) for the difference of means and the speedup

Outputs:
  - outputs/statistics/significance_summary.json (nested structure by pair)
  - outputs/statistics/significance_summary.csv (flat structure with a 'pair' column)
  - outputs/statistics/significance_summary.md

Definitions:
  - Difference (diff_s) = arch_a - arch_b (in seconds)
  - Speedup (arch_b_vs_arch_a) = arch_a_mean / arch_b_mean (greater than 1 favors arch_b)
  - Architectural total: sum of the architecture-specific phases
    (excludes 'collection', which is common to all)
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

# Project root, so the configuration can be imported
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


# Read order for the CSV columns. It is ordering, not filtering: the previous
# version was a whitelist of pre-rename names and silently discarded everything
# it did not recognize -- the per-paradigm means and every speedup with its
# 95% CI, which are precisely the columns whose name contains the paradigm.
# Four survived: n, mean_diff_s and the CI of the difference.
_COLUMN_ORDER = ("n", "mean_", "speedup_", "diff_mean_ci95_",
                 "shapiro_", "anderson_", "t_", "wilcoxon_")


def column_rank(name: str) -> tuple:
    """Read position of a column; unknown ones go to the end, in order."""
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
    """Return aligned vectors for two architectures, by run_id, within one phase."""
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
    """Sum durations by run_id and architecture over the comparable phases.

    Which ones those are comes from core.paradigm_registry: four files used to
    enumerate the policy, and one of them had already forgotten to apply it.
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
    """Bootstrap 95% CI for the difference of means (x-y) and the speedup (x_mean/y_mean)."""
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
    """Run the significance and normality tests on the differences (x - y)."""
    res: Dict[str, float] = {}
    diff = x - y
    # Normality
    if len(diff) >= 3:
        sh_w, sh_p = stats.shapiro(diff)
    else:
        sh_w, sh_p = float("nan"), float("nan")
    try:
        ad = stats.anderson(diff, dist='norm')
        ad_stat = float(ad.statistic)
    except Exception:
        ad_stat = float("nan")

    # Paired tests
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

    # The signed-rank test discards exactly-zero differences, so the test's n
    # is not the number of pairs. It is that n which determines the smallest
    # attainable p (2/2^n two-sided), and the reported floor came out computed
    # over the pairs: with three ties out of ten, the real floor is eight times
    # larger.
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
    """Analyze every pairwise comparison (DL×DW, DL×PL, DW×PL)."""
    df = load_benchmark(csv_path)
    phases = sorted([p for p in df['phase'].unique()
                     if p in COMPARABLE_PHASES])
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # For each pair of architectures
    for arch_a, arch_b in ALL_PAIRS:
        la, lb = arch_a, arch_b
        pair_key = f"{la}_vs_{lb}"
        pair_results: Dict[str, Dict[str, float]] = {}

        # By phase
        for p in phases:
            x, y = paired_vectors_for_phase(df, p, arch_a, arch_b)
            if len(x) >= 2 and len(y) >= 2:
                pair_results[p] = run_tests(x, y, label_a=la, label_b=lb, bootstrap_iters=bootstrap_iters)

        # Architectural total over the comparable phases; the list comes from the registry.
        x_tot, y_tot = paired_vectors_total(df, arch_a, arch_b)
        if len(x_tot) >= 2 and len(y_tot) >= 2:
            pair_results["total_architectural"] = run_tests(
                x_tot, y_tot, label_a=la, label_b=lb, bootstrap_iters=bootstrap_iters
            )

        results[pair_key] = pair_results

    return results


def _format_markdown_table(rows: List[Dict[str, float]], cols: List[str], include_pair: bool = False) -> str:
    """Build a stable Markdown table without depending on tabulate."""
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

    # JSON: nested structure by pair
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

    # Markdown (selected and rounded columns)
    md_cols = [
        "n",
        "mean_diff_s",
        "shapiro_p",
        "t_stat",
        "t_p",
        "wilcoxon_stat",
        "wilcoxon_p",
    ]
    # Keep only the columns that exist
    md_cols = [c for c in md_cols if c in cols_present]

    with open(md_path, "w") as fmd:
        fmd.write("# Statistical Significance Summary (3-Way Pairwise)\n\n")
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
    except Exception as exc:
        # It used to be swallowed: the per-pair table simply did not appear, and
        # the step went on with exit 0. A published artifact that is missing is
        # indistinguishable from one that was never asked for.
        raise RuntimeError(
            f"Failed to write the significance tables to "
            f"{RESULTS_DIR}: {exc}"
        ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical significance tests for the architectural benchmark")
    p.add_argument(
        "--csv",
        default=get_absolute_output_path(
            "outputs/benchmarks/architectural_benchmark_results.csv"),
        help="Path to the benchmark CSV",
    )
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_ITERS, help=f"Bootstrap iterations (default={DEFAULT_BOOTSTRAP_ITERS})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = analyze(args.csv, bootstrap_iters=args.bootstrap)
    write_outputs(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


