#!/usr/bin/env python3
"""
Resumo de throughput (registros/segundo) do benchmark arquitetural (PT-BR).

Entradas:
  - outputs/benchmarks/architectural_benchmark_results.csv (colunas 'records' e duração)

Saídas:
  - outputs/statistics/architectural_throughput_percentiles.json
  - outputs/statistics/architectural_throughput_percentiles.tex

Notas:
  - Apenas fases com 'records' > 0 e duração > 0 são consideradas.
  - Computa P50/P95/P99 de throughput por arquitetura e fase.
  - Se não houver dados para uma fase, preenche com '—' na tabela LaTeX.
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
from core.paradigm_registry import comparable_rows, discover_paradigms

RESULTS_CSV = Path(get_absolute_output_path(
    "outputs/benchmarks/architectural_benchmark_results.csv"))
OUT_DIR = Path(get_absolute_output_path("outputs/statistics"))
OUT_JSON = OUT_DIR / "architectural_throughput_percentiles.json"
OUT_TEX = OUT_DIR / "architectural_throughput_percentiles.tex"


def _garantir_diretorio() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt(val: float | None, sufixo: str = " r/s") -> str:
    if val is None or not np.isfinite(val):
        return "—"
   
    base = f"{val:,.1f}".replace(",", r"\,")
    return f"{base}{sufixo}"


def _escape_latex(text: str) -> str:
    if text is None:
        return ""
    return text.replace("_", r"\_")


def _pct(arr: np.ndarray, q: float) -> float | None:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def resumir(df: pd.DataFrame) -> Dict:

    if df.empty:
        return {"erro": "resultados_vazios"}

    df = df.copy()
    if "duration_ns" in df.columns:
        df["duration_s"] = df["duration_ns"].astype(float) / 1e9
    elif "duration_s" in df.columns:
        df["duration_s"] = df["duration_s"].astype(float)
    else:
        raise SystemExit("CSV de resultados sem coluna de duração (duration_ns/duration_s)")

    if "records" not in df.columns:
        return {"erro": "coluna_records_ausente"}

    df = df[(df["records"].fillna(0) > 0) & (df["duration_s"].fillna(0) > 0)].copy()
    if df.empty:
        return {"erro": "sem_registros_para_throughput"}

    df["throughput_rps"] = df["records"].astype(float) / df["duration_s"].astype(float)

    df = comparable_rows(df)
    fases = sorted(df["phase"].unique())
    arq = sorted(df["architecture"].unique())

    out: Dict = {"per_phase": {}, "architectures": arq, "fases": fases}
    for fase in fases:
        dff = df[df["phase"] == fase]
        por_arq: Dict[str, Dict[str, float | None]] = {}
        for a in arq:
            arr = dff[dff["architecture"] == a]["throughput_rps"].to_numpy()
            por_arq[a] = {
                "p50": _pct(arr, 50),
                "p95": _pct(arr, 95),
                "p99": _pct(arr, 99),
                "n": int(np.isfinite(arr).sum()),
            }
        out["per_phase"][fase] = {"architectures": por_arq}

    return out


def para_latex(resumo: Dict) -> str:
    """Tabela transposta: uma linha por (fase, paradigma).

    O cabeçalho anterior nomeava dois paradigmas em colunas fixas, e o terceiro
    não aparecia na tabela publicada. Transposta, as linhas vêm do registro e
    nenhum paradigma pode ficar de fora.
    """
    paradigms = sorted(discover_paradigms())
    prefix = [
        r"\begingroup",
        r"\setlength{\tabcolsep}{4pt}",
        r"\scriptsize",
        r"\begin{center}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\hline",
        r"Fase & Paradigma & P50 & P95 & P99 \\",
        r"\hline",
    ]
    suffix = [r"\hline", r"\end{tabular}", r"\end{center}", r"\endgroup"]

    if not resumo or "per_phase" not in resumo:
        return "\n".join(prefix + [r"Sem dados & — & — & — & — \\"] + suffix) + "\n"

    linhas: List[str] = [
        "% Gerado automaticamente por derive_throughput_percentiles.py",
        "% P50/P95/P99 de throughput (registros/segundo), por fase e paradigma",
        "",
    ]
    linhas.extend(prefix)

    for fase in sorted(resumo["per_phase"]):
        arquiteturas = resumo["per_phase"][fase].get("architectures", {})
        rotulo = _escape_latex(fase)
        for paradigm in paradigms:
            stats = arquiteturas.get(paradigm, {})
            linhas.append(
                f"{rotulo} & {_escape_latex(paradigm)}"
                f" & {_fmt(stats.get('p50'))}"
                f" & {_fmt(stats.get('p95'))}"
                f" & {_fmt(stats.get('p99'))} \\\\")
            rotulo = ""
        linhas.append(r"\hline")

    if linhas[-1] == r"\hline":
        linhas.pop()
    linhas.extend(suffix)
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

    try:
        df = pd.read_csv(RESULTS_CSV)
        resumo = resumir(df)
        

        OUT_JSON.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
        OUT_TEX.write_text(para_latex(resumo), encoding="utf-8")
        
        print(json.dumps({
            "status": "ok", 
            "json": str(OUT_JSON), 
            "tex": str(OUT_TEX)
        }, indent=2, ensure_ascii=False))
        
    except Exception as e:
        erro_msg = {
            "erro": f"Falha no processamento: {str(e)}",
            "arquivo": str(RESULTS_CSV)
        }
        OUT_JSON.write_text(json.dumps(erro_msg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(erro_msg, indent=2, ensure_ascii=False))
        raise


if __name__ == "__main__":
    main()