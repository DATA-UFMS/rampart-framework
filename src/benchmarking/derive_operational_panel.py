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
    phases = []
    lat = {}
    res = {}
    if LAT_JSON.exists():
        j = json.loads(LAT_JSON.read_text())
        per_phase = j.get("per_phase", {})
        phases = sorted(per_phase.keys())
        lat = j
    if RES_JSON.exists():
        r = json.loads(RES_JSON.read_text())
        res = r.get("per_phase", r.get("por_fase", {}))

    lines = []
    lines.append("% Painel operacional compacto (P50 e recursos médios)")
    lines.append("\\begin{tabular}{lrrrrrrr}")
    lines.append("\\hline")
    lines.append("Fase & DL P50 & DW P50 & Speedup P50 & CPU(p) DL & CPU(p) DW & RSS DL & RSS DW \\\\")
    lines.append("\\hline")

    def row_for_phase(ph: str) -> str:
        per_phase = lat.get("per_phase", {})
        arch_key = "architectures"
        dl_p50 = per_phase.get(ph, {}).get(arch_key, {}).get("task_graph", {}).get("p50")
        dw_p50 = per_phase.get(ph, {}).get(arch_key, {}).get("sql_engine", {}).get("p50")
        sp = per_phase.get(ph, {}).get("speedup_dw_vs_dl_p50")
        # recursos
        rdl = res.get(ph, {}).get("task_graph", {}) if res else {}
        rdw = res.get(ph, {}).get("sql_engine", {}) if res else {}
        cpu_dl = rdl.get('cpu_proc_mean')
        cpu_dw = rdw.get('cpu_proc_mean')
        rss_dl = rdl.get('rss_mb_mean')
        rss_dw = rdw.get('rss_mb_mean')
        sp_s = f"{float(sp):.2f}×" if (sp is not None and np.isfinite(sp)) else "—"
        return (
            f"{ph} & {_fmt_s(dl_p50)} & {_fmt_s(dw_p50)} & {sp_s} & "
            f"{_fmt_pct(cpu_dl)} & {_fmt_pct(cpu_dw)} & {_fmt_mb(rss_dl)} & {_fmt_mb(rss_dw)} \\\\"
        )

    for ph in phases:
        lines.append(row_for_phase(ph))

    # total
    t_total = lat.get("total", {})
    t_arch_key = "architectures"
    t_dl = t_total.get(t_arch_key, {}).get("task_graph", {}).get("p50")
    t_dw = t_total.get(t_arch_key, {}).get("sql_engine", {}).get("p50")
    t_sp = t_total.get("speedup_dw_vs_dl_p50")
    t_sp_s = f"{float(t_sp):.2f}×" if (t_sp is not None and np.isfinite(t_sp)) else "—"
    lines.append("\\hline")
    lines.append(f"Total & {_fmt_s(t_dl)} & {_fmt_s(t_dw)} & {t_sp_s} & — & — & — & — \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status":"ok","tex":str(OUT_TEX)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
