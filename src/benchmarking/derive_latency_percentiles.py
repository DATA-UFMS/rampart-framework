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


import os
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import get_absolute_output_path
from core.paradigm_registry import discover_paradigms, paradigm_pairs

RESULTS_CSV = Path(get_absolute_output_path(
    "outputs/benchmarks/architectural_benchmark_results.csv"))
OUT_DIR = Path(get_absolute_output_path("outputs/statistics"))
OUT_JSON = OUT_DIR / "architectural_latency_percentiles.json"
OUT_TEX = OUT_DIR / "architectural_latency_percentiles.tex"

EXCLUDE_PHASES = {"collection"}


def _garantir_diretorio() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _speedups(por_arq: Dict[str, Dict]) -> Dict[str, float | None]:
    """Razão de medianas por par de paradigmas, nomeada a/b na ordem do registro.

    Um par cujo denominador não é positivo devolve None, e o consumidor imprime
    travessão -- mas por ausência de dado, não por chave escrita com um nome e
    lida com outro.
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

    resumo: Dict = {"per_phase": {}, "architectures": arq, "fases": fases}

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

        resumo["per_phase"][fase] = {
            "architectures": por_arq,
            # Chaves derivadas do registro e nomeadas na mesma ordem usada no
            # total. Antes o per_phase gravava speedup_dl_vs_dw_p50 e o LaTeX lia
            # speedup_dw_vs_dl_p50: a coluna saía vazia em toda linha.
            "speedups_p50": _speedups(por_arq),
        }

    # Totais por execução (somando fases não-excluídas)
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
    """Tabela transposta: uma linha por (fase, paradigma).

    O layout anterior tinha uma coluna por percentil de dois paradigmas, com os
    nomes escritos no cabeçalho -- o terceiro paradigma não aparecia em lugar
    nenhum da tabela, e um quarto exigiria reescrever o cabeçalho. Transposta,
    a tabela escala sem alteração e nenhum paradigma pode ser esquecido, porque
    as linhas vêm do registro.
    """
    paradigms = sorted(discover_paradigms())
    por_fase = resumo.get("per_phase", {})
    if not por_fase:
        return ("% Sem dados de latência\n"
                "\\begin{tabular}{llrrr}\n\\hline\n"
                "Fase & Paradigma & P50 & P95 & P99 \\\\ \n"
                "\\hline\n\\end{tabular}\n")

    linhas: List[str] = [
        "% Gerado automaticamente por derive_latency_percentiles.py",
        "% P50/P95/P99 em segundos, por fase e paradigma",
        "\\begin{tabular}{llrrr}",
        "\\hline",
        "Fase & Paradigma & P50 & P95 & P99 \\\\",
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

    # Speedups em tabela própria: um par por linha, derivado do registro.
    linhas += [
        "",
        "% Speedup de mediana por par (P50 de A dividido por P50 de B)",
        # Uma coluna de rótulo mais uma por fase mais o total.
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
