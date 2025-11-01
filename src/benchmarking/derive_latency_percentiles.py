#!/usr/bin/env python3
"""
Resumo de latências do benchmark arquitetural em tabelas de evidência (PT-BR).

Entradas:
  - outputs/benchmarks/architectural_benchmark_results.csv

Saídas:
  - outputs/statistics/architectural_latency_percentiles.json
  - outputs/statistics/architectural_latency_percentiles.tex

Notas:
  - Exclui a fase 'collection' do cálculo de speedup por padrão.
  - Computa P50/P95/P99 por arquitetura e fase (segundos).
  - Computa speedup por fase como (mediana_DL_seg / mediana_DW_seg) → maior é melhor para DW.
  - Também reporta percentis e speedup do tempo total por execução (soma das fases não‑excluídas).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


RESULTS_CSV = Path("outputs/benchmarks/architectural_benchmark_results.csv")
OUT_DIR = Path("outputs/statistics")
OUT_JSON = OUT_DIR / "architectural_latency_percentiles.json"
OUT_TEX = OUT_DIR / "architectural_latency_percentiles.tex"

EXCLUDE_PHASES = {"collection"}


def _garantir_diretorio() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    # Espera colunas: run_id, phase, architecture, step, duration_ns (ou duration_s), records
    if df.empty:
        return {"erro": "resultados_vazios"}

    df = df.copy()
    if "duration_ns" in df.columns:
        df["duration_s"] = df["duration_ns"].astype(float) / 1e9
    elif "duration_s" in df.columns:
        df["duration_s"] = df["duration_s"].astype(float)
    else:
        raise SystemExit("CSV de resultados sem coluna de duração (duration_ns/duration_s)")

    # Filtrar fases
    df_filt = df[~df["phase"].isin(EXCLUDE_PHASES)].copy()
    if df_filt.empty:
        df_filt = df.copy()

    fases = sorted(df_filt["phase"].unique())
    arq = sorted(df_filt["architecture"].unique())

    resumo: Dict = {"por_fase": {}, "arquiteturas": arq, "fases": fases}

    # Percentis por fase
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

        dl_med = por_arq.get("data_lake", {}).get("p50")
        dw_med = por_arq.get("data_warehouse", {}).get("p50")
        speedup = (dl_med / dw_med) if (dl_med and dw_med and dw_med > 0) else None

        resumo["por_fase"][fase] = {"arquiteturas": por_arq, "speedup_dw_vs_dl_p50": speedup}

    # Totais por execução (somando fases não-excluídas)
    totais = (
        df_filt.groupby(["run_id", "architecture"])  # type: ignore
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
    dl_med = por_arq_total.get("data_lake", {}).get("p50")
    dw_med = por_arq_total.get("data_warehouse", {}).get("p50")
    speedup_total = (dl_med / dw_med) if (dl_med and dw_med and dw_med > 0) else None
    resumo["total"] = {"arquiteturas": por_arq_total, "speedup_dw_vs_dl_p50": speedup_total}
    return resumo


def para_latex(resumo: Dict) -> str:
    if not resumo or "por_fase" not in resumo:
        return (
            "\\begin{tabular}{lrrrrrr}\n"
            "\\hline\n"
            "Fase & DL P50 & DL P95 & DL P99 & DW P50 & DW P95 & DW P99 \\\\ \n"
            "\\hline\n"
            "Sem dados & -- & -- & -- & -- & -- & -- \\\\ \n"
            "\\hline\n"
            "\\end{tabular}\n"
        )

    linhas: List[str] = []
    linhas.append("% Gerado automaticamente por derive_latency_percentiles.py")
    linhas.append("% P50/P95/P99 em segundos e speedups DW vs DL (P50)")
    linhas.append("\\begin{tabular}{lrrrrrrr}")
    linhas.append("\\hline")
    linhas.append("Fase & DL P50 & DL P95 & DL P99 & DW P50 & DW P95 & DW P99 & Speedup DW (P50) \\ ")
    linhas.append("\\hline")

    por_fase = resumo.get("por_fase", {})
    for fase in sorted(por_fase.keys()):
        item = por_fase[fase]
        a = item.get("arquiteturas", {})
        dl = a.get("data_lake", {})
        dw = a.get("data_warehouse", {})
        sp = item.get("speedup_dw_vs_dl_p50")
        row = [
            fase,
            _fmt_segundos(dl.get("p50")),
            _fmt_segundos(dl.get("p95")),
            _fmt_segundos(dl.get("p99")),
            _fmt_segundos(dw.get("p50")),
            _fmt_segundos(dw.get("p95")),
            _fmt_segundos(dw.get("p99")),
            (f"{sp:.2f}×" if sp and np.isfinite(sp) else "—"),
        ]
        linhas.append(" ".join([f"{col}" for col in row]).replace(" ", " & ") + " \\ ")

    total = resumo.get("total", {})
    ta = total.get("arquiteturas", {})
    dl = ta.get("data_lake", {})
    dw = ta.get("data_warehouse", {})
    sp = total.get("speedup_dw_vs_dl_p50")
    linhas.append("\\hline")
    linhas.append(
        ("Total" +
         f" & {_fmt_segundos(dl.get('p50'))} & {_fmt_segundos(dl.get('p95'))} & {_fmt_segundos(dl.get('p99'))}" +
         f" & {_fmt_segundos(dw.get('p50'))} & {_fmt_segundos(dw.get('p95'))} & {_fmt_segundos(dw.get('p99'))}" +
         f" & {(f'{sp:.2f}×' if sp and np.isfinite(sp) else '—')} \\ ")
    )
    linhas.append("\\hline")
    linhas.append("\\end{tabular}")
    return "\n".join(linhas) + "\n"


def main() -> None:
    _garantir_diretorio()
    if not RESULTS_CSV.exists():
        msg = {
            "erro": f"CSV não encontrado: {str(RESULTS_CSV)}",
            "dica": "Execute o benchmark arquitetural para gerar resultados.",
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
