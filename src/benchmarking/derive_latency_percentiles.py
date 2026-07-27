#!/usr/bin/env python3
"""
Summary of architectural benchmark latencies into evidence tables (PT-BR).

Inputs:
  - outputs/benchmarks/architectural_benchmark_results.csv

Outputs:
  - outputs/statistics/architectural_latency_percentiles.json
  - outputs/statistics/architectural_latency_percentiles.tex

Notes:
  - Compares only the phases the three paradigms execute; the list comes from
    core.paradigm_registry.COMPARABLE_PHASES.
  - Computes P50/P95/P99 per architecture and phase (seconds).
  - Computes per-phase speedup as (median_DL_s / median_DW_s) → higher is better for DW.
  - Also reports percentiles and speedup of the total time per run (sum of the non‑excluded phases).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


import os
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import get_absolute_output_path
from core.paradigm_registry import (comparable_rows, discover_paradigms,
                                    paradigm_pairs)

RESULTS_CSV = Path(get_absolute_output_path(
    "outputs/benchmarks/architectural_benchmark_results.csv"))
OUT_DIR = Path(get_absolute_output_path("outputs/statistics"))
OUT_JSON = OUT_DIR / "architectural_latency_percentiles.json"
OUT_TEX = OUT_DIR / "architectural_latency_percentiles.tex"



def _garantir_diretorio() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _speedups(por_arq: Dict[str, Dict]) -> Dict[str, float | None]:
    """Ratio of medians per paradigm pair, named a/b in registry order.

    A pair whose denominator is not positive returns None, and the consumer
    prints an em dash -- but because the datum is absent, not because a key was
    written under one name and read under another.
    """
    out: Dict[str, float | None] = {}
    for left, right in paradigm_pairs():
        a = por_arq.get(left, {}).get("p50")
        b = por_arq.get(right, {}).get("p50")
        out[f"{left}_vs_{right}"] = (
            (a / b) if (a is not None and b is not None and b > 0) else None)
    return out


def _fmt_segundos(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.2f}s"


def _pct(arr: np.ndarray, q: float) -> float | None:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def resumir_percentis(df: pd.DataFrame) -> Dict:
    # Expects columns: run_id, phase, architecture, step, duration_ns (or duration_s), records
    if df.empty:
        return {"erro": "resultados_vazios"}

    df = df.copy()
    if "duration_ns" in df.columns:
        df["duration_s"] = df["duration_ns"].astype(float) / 1e9
    elif "duration_s" in df.columns:
        df["duration_s"] = df["duration_s"].astype(float)
    else:
        raise SystemExit("Results CSV without a duration column (duration_ns/duration_s)")

    # The previous fallback reinstated exactly the excluded rows whenever the
    # filter emptied the frame: with only collection left, the latency table
    # came out built on it.
    df_filt = comparable_rows(df)

    fases = sorted(df_filt["phase"].unique())
    arq = sorted(df_filt["architecture"].unique())

    resumo: Dict = {"per_phase": {}, "architectures": arq, "fases": fases}

    # Percentiles per phase
    for fase in fases:
        dff = df_filt[df_filt["phase"] == fase]
        por_arq: Dict[str, Dict[str, float | None]] = {}
        for a in arq:
            vals = dff[dff["architecture"] == a]["duration_s"].to_numpy()
            por_arq[a] = {
                "p50": _pct(vals, 50),
                "p95": _pct(vals, 95),
                "p99": _pct(vals, 99),
                "n": int(np.isfinite(vals).sum()),
            }

        resumo["per_phase"][fase] = {
            "architectures": por_arq,
            # Keys derived from the registry and named in the same order used in
            # the total. Before, per_phase wrote speedup_dl_vs_dw_p50 and the
            # LaTeX read speedup_dw_vs_dl_p50: the column came out empty on
            # every row.
            "speedups_p50": _speedups(por_arq),
        }

    # Totals per run (summing the non-excluded phases)
    totais = (
        df_filt.groupby(["run_id", "architecture"])
        ["duration_s"].sum().reset_index()
    )
    por_arq_total: Dict[str, Dict[str, float | None]] = {}
    for a in arq:
        arr = totais[totais["architecture"] == a]["duration_s"].to_numpy()
        por_arq_total[a] = {
            "p50": _pct(arr, 50),
            "p95": _pct(arr, 95),
            "p99": _pct(arr, 99),
            "n_runs": int(np.isfinite(arr).sum()),
        }
    resumo["total"] = {
        "architectures": por_arq_total,
        "speedups_p50": _speedups(por_arq_total),
    }
    return resumo


def para_latex(resumo: Dict) -> str:
    """Transposed table: one row per (phase, paradigm).

    The previous layout had one column per percentile of two paradigms, with the
    names written in the header -- the third paradigm appeared nowhere in the
    table, and a fourth would have required rewriting the header. Transposed,
    the table scales without modification and no paradigm can be forgotten,
    because the rows come from the registry.
    """
    paradigms = sorted(discover_paradigms())
    por_fase = resumo.get("per_phase", {})
    if not por_fase:
        return ("% No latency data\n"
                "\\begin{tabular}{llrrr}\n\\hline\n"
                "Phase & Paradigm & P50 & P95 & P99 \\\\ \n"
                "\\hline\n\\end{tabular}\n")

    linhas: List[str] = [
        "% Generated automatically by derive_latency_percentiles.py",
        "% P50/P95/P99 in seconds, by phase and paradigm",
        "\\begin{tabular}{llrrr}",
        "\\hline",
        "Phase & Paradigm & P50 & P95 & P99 \\\\",
        "\\hline",
    ]

    def _bloco(rotulo: str, arquiteturas: Dict) -> None:
        for paradigm in paradigms:
            stats = arquiteturas.get(paradigm, {})
            linhas.append(
                f"{rotulo} & {paradigm.replace('_', chr(92) + '_')}"
                f" & {_fmt_segundos(stats.get('p50'))}"
                f" & {_fmt_segundos(stats.get('p95'))}"
                f" & {_fmt_segundos(stats.get('p99'))} \\\\")
            rotulo = ""

    for fase in sorted(por_fase):
        _bloco(fase, por_fase[fase].get("architectures", {}))
        linhas.append("\\hline")
    _bloco("Total", resumo.get("total", {}).get("architectures", {}))
    linhas += ["\\hline", "\\end{tabular}", ""]

    # Speedups in a table of their own: one pair per row, derived from the registry.
    linhas += [
        "",
        "% Speedup de mediana por par (P50 de A dividido por P50 de B)",
        # One label column plus one per phase plus the total.
        "\\begin{tabular}{l" + "r" * (len(por_fase) + 1) + "}",
        "\\hline",
        "Par & " + " & ".join(sorted(por_fase)) + " & Total \\\\",
        "\\hline",
    ]
    for left, right in paradigm_pairs():
        key = f"{left}_vs_{right}"
        celulas = []
        for fase in sorted(por_fase):
            value = por_fase[fase].get("speedups_p50", {}).get(key)
            celulas.append(f"{value:.2f}" if value and np.isfinite(value) else "—")
        total_value = resumo.get("total", {}).get("speedups_p50", {}).get(key)
        celulas.append(f"{total_value:.2f}"
                       if total_value and np.isfinite(total_value) else "—")
        label = f"{left} / {right}".replace('_', chr(92) + '_')
        linhas.append(f"{label} & " + " & ".join(celulas) + " \\\\")
    linhas += ["\\hline", "\\end{tabular}"]
    return "\n".join(linhas) + "\n"


def main() -> None:
    _garantir_diretorio()
    if not RESULTS_CSV.exists():
        msg = {
            "erro": f"CSV not found: {str(RESULTS_CSV)}",
            "dica": "Run the architectural benchmark to generate results.",
        }
        OUT_JSON.write_text(json.dumps(msg, indent=2, ensure_ascii=False), encoding="utf-8")
        OUT_TEX.write_text("% Sem dados\n" + para_latex({}), encoding="utf-8")
        print(json.dumps(msg, indent=2, ensure_ascii=False))
        return

    df = pd.read_csv(RESULTS_CSV)
    resumo = resumir_percentis(df)
    OUT_JSON.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_TEX.write_text(para_latex(resumo), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(OUT_JSON), "tex": str(OUT_TEX)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
