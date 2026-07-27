#!/usr/bin/env python3
"""
Compact operational panel: latency (P50) and mean resources per phase.

Reads:
  - outputs/statistics/architectural_latency_percentiles.json
  - outputs/statistics/architectural_resource_usage.json

Generates:
  - outputs/statistics/architectural_operational_panel.tex

Columns (per phase and total):
  Phase | DL P50 (s) | DW P50 (s) | Speedup P50 | CPU(proc)% DL | CPU(proc)% DW | RSS(MB) DL | RSS(MB) DW
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
    """Percent, escaped for LaTeX.

    A bare % opens a comment: everything after it on the row disappeared,
    including the remaining columns and the \\ that ends the line. The table
    still rendered, with fewer columns than its own specification declares.
    """
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x):.1f}\\%"


def _fmt_mb(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x):.1f}"


def _escape(text) -> str:
    """Underscore and percent both need escaping; both reached the file raw."""
    return str(text).replace('_', r'\_').replace('%', r'\%')


def para_latex(lat: dict, res: dict, paradigms) -> str:
    """Operational panel: one row per (phase, paradigm).

    The previous layout had fixed columns for two paradigms and read the key
    speedup_dw_vs_dl_p50, which the percentile generator never wrote in the
    per_phase block -- the speedup column came out as an em dash on every row,
    and the third paradigm did not appear. Transposed, the rows come from the
    registry.

    Extracted from main so it could be exercised: it was inside main that the
    unescaped percent entered the row, and no test reached the table.
    """
    per_phase = lat.get("per_phase", {})
    phases = sorted(per_phase)

    lines = [
        "% Operational panel (P50 of latency and mean resources)",
        "\\begin{tabular}{llrrr}",
        "\\hline",
        "Phase & Paradigm & P50 & CPU(proc) & RSS \\\\",
        "\\hline",
    ]

    def _bloco(rotulo: str, arquiteturas: dict, recursos: dict) -> None:
        for paradigm in paradigms:
            stats = arquiteturas.get(paradigm, {})
            usage = recursos.get(paradigm, {}) if recursos else {}
            lines.append(
                f"{_escape(rotulo)} & {_escape(paradigm)}"
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
    return "\n".join(lines)


def main() -> None:
    paradigms = sorted(discover_paradigms())

    lat, res = {}, {}
    if LAT_JSON.exists():
        lat = json.loads(LAT_JSON.read_text())
    if RES_JSON.exists():
        payload = json.loads(RES_JSON.read_text())
        res = payload.get("per_phase", payload.get("por_fase", {}))

    lines = [para_latex(lat, res, paradigms)]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(lines[0] + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "tex": str(OUT_TEX),
                      "paradigms": paradigms}, ensure_ascii=False))


if __name__ == "__main__":
    main()
