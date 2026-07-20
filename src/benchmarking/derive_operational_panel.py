#!/usr/bin/env python3
"""
Painel operacional compacto (PT-BR): latência (P50) e recursos médios por fase.

Lê:
  - outputs/statistics/architectural_latency_percentiles.json
  - outputs/statistics/architectural_resource_usage.json

Gera:
  - outputs/statistics/architectural_operational_panel.tex

Colunas (por fase e total):
  Fase | DL P50 (s) | DW P50 (s) | Speedup P50 | CPU(proc)% DL | CPU(proc)% DW | RSS(MB) DL | RSS(MB) DW
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import os
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import get_absolute_output_path
from core.paradigm_registry import discover_paradigms

LAT_JSON = Path(get_absolute_output_path(
    "outputs/statistics/architectural_latency_percentiles.json"))
RES_JSON = Path(get_absolute_output_path(
    "outputs/statistics/architectural_resource_usage.json"))
OUT_TEX = Path(get_absolute_output_path(
    "outputs/statistics/architectural_operational_panel.tex"))


def _fmt_s(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x):.2f}s"


def _fmt_pct(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x):.1f}%"


def _fmt_mb(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x):.1f}"


def main() -> None:
    """Painel operacional: latência e recursos, uma linha por (fase, paradigma).

    O layout anterior tinha colunas fixas para dois paradigmas e lia a chave
    speedup_dw_vs_dl_p50, que o gerador de percentis nunca escreveu no bloco
    per_phase -- a coluna de speedup saía em travessão em todas as linhas, e o
    terceiro paradigma não aparecia. Transposto, as linhas vêm do registro.
    """
    paradigms = sorted(discover_paradigms())

    lat, res = {}, {}
    if LAT_JSON.exists():
        lat = json.loads(LAT_JSON.read_text())
    if RES_JSON.exists():
        payload = json.loads(RES_JSON.read_text())
        res = payload.get("per_phase", payload.get("por_fase", {}))

    per_phase = lat.get("per_phase", {})
    phases = sorted(per_phase)

    lines = [
        "% Painel operacional (P50 de latência e recursos médios)",
        "\\begin{tabular}{llrrr}",
        "\\hline",
        "Fase & Paradigma & P50 & CPU(proc) & RSS \\\\",
        "\\hline",
    ]

    def _bloco(rotulo: str, arquiteturas: dict, recursos: dict) -> None:
        for paradigm in paradigms:
            stats = arquiteturas.get(paradigm, {})
            usage = recursos.get(paradigm, {}) if recursos else {}
            lines.append(
                f"{rotulo} & {paradigm.replace('_', chr(92) + '_')}"
                f" & {_fmt_s(stats.get('p50'))}"
                f" & {_fmt_pct(usage.get('cpu_proc_mean'))}"
                f" & {_fmt_mb(usage.get('rss_mb_mean'))} \\\\")
            rotulo = ""

    for phase in phases:
        _bloco(phase, per_phase[phase].get("architectures", {}),
               res.get(phase, {}))
        lines.append("\\hline")

    _bloco("Total", lat.get("total", {}).get("architectures", {}), {})
    lines += ["\\hline", "\\end{tabular}"]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "tex": str(OUT_TEX),
                      "paradigms": paradigms}, ensure_ascii=False))


if __name__ == "__main__":
    main()
