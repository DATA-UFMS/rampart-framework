#!/usr/bin/env python3
"""
Geração automática do painel consolidado (scorecard) em LaTeX a partir dos
artefatos já produzidos pelo pipeline.

Entradas (melhor esforço, com fallbacks):
 - outputs/statistics/significance_summary.json ou .tex (speedups + IC95 por fase)
 - outputs/statistics/tost_baseline.json (equivalência em R^2, faixa de δ)
 - outputs/statistics/tost_baseline_mase.json (opcional)
 - outputs/statistics/tost_baseline_nrmse.json (opcional)
 - outputs/statistics/architectural_resource_usage.tex (CPU(proc) média e RSS para processing)

Saída:
 - outputs/statistics/architectural_scorecard.tex

Uso:
 - Execute diretamente: `python src/statistical_validation/make_scorecard.py`
 - Integrado ao pipeline: chamado automaticamente após o benchmark.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

BASE = Path('outputs/statistics')


def read_text(p: Path) -> str:
    return p.read_text(encoding='utf-8') if p.exists() else ''


def load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    except Exception:
        return None


def parse_significance_tex(tex: str) -> Dict[str, Tuple[float, float, float]]:
    """Extrai speedup e IC 95% de significance_summary.tex.
    Retorna dict fase -> (speedup, lo, hi).
    """
    res: Dict[str, Tuple[float, float, float]] = {}
    # Expect rows like: "Processing & 10.00 & 1.00 & 9.00 [8.00, 10.00] & 13.70 [10.32, 18.18] & ..."
    for line in tex.splitlines():
        if not line or '&' not in line or line.startswith('%'):
            continue
        parts = [p.strip() for p in line.split('&')]
        if len(parts) < 6:
            continue
        phase = parts[0].lower()
        # speedup field looks like: "7.10 [6.58, 7.67]"
        m = re.search(r"([0-9]+\.?[0-9]*)\s*\[\s*([0-9]+\.?[0-9]*),\s*([0-9]+\.?[0-9]*)\s*\]", parts[5])
        if m:
            val = float(m.group(1))
            lo = float(m.group(2))
            hi = float(m.group(3))
            res[phase] = (val, lo, hi)
    return res


def get_speedups() -> Dict[str, Tuple[float, float, float]]:
    # Prefer JSON if available
    j = load_json(BASE / 'significance_summary.json')
    if j:
        out: Dict[str, Tuple[float, float, float]] = {}
        for phase, metrics in j.items():
            if not isinstance(metrics, dict):
                continue
            if 'speedup_dw_vs_dl' in metrics and 'speedup_ci95_lo' in metrics and 'speedup_ci95_hi' in metrics:
                out[phase.lower()] = (
                    float(metrics['speedup_dw_vs_dl']),
                    float(metrics['speedup_ci95_lo']),
                    float(metrics['speedup_ci95_hi']),
                )
        if out:
            return out
    tex = read_text(BASE / 'significance_summary.tex')
    if tex:
        return parse_significance_tex(tex)
    # Last resort: empty
    return {}


def summarize_tost(name: str) -> Optional[str]:
    p = BASE / f'tost_baseline{name}.json' if name else BASE / 'tost_baseline.json'
    data = load_json(p)
    if not data:
        return None
    deltas = []
    flags = []
    for k, v in data.items():
        try:
            d = float(k.split('_')[1])
        except Exception:
            continue
        deltas.append(d)
        flags.append(bool(v.get('equivalent', False)))
    if not deltas:
        return None
    lo, hi = min(deltas), max(deltas)
    status = 'Sim' if all(flags) else 'Não'
    # Use LaTeX math for delta/in and escape braces
    return f"{status} ($\\delta\\in\\{{{lo:.2f},{hi:.2f}\\}})"


def get_resources_processing() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (cpu_dl, cpu_dw, rss_dl, rss_dw) from resource_usage.tex for processing phase."""
    tex = read_text(BASE / 'architectural_resource_usage.tex')
    if not tex:
        return (None, None, None, None)
    cpu_dl = cpu_dw = rss_dl = rss_dw = None
    for line in tex.splitlines():
        if line.startswith('%'):
            continue
        if 'processing & data_lake' in line:
            parts = [p.strip() for p in line.split('&')]
            try:
                cpu_dl = float(parts[2].rstrip('% '))
                rss_dl = float(parts[4])
            except Exception:
                pass
        if 'processing & data_warehouse' in line:
            parts = [p.strip() for p in line.split('&')]
            try:
                cpu_dw = float(parts[2].rstrip('% '))
                rss_dw = float(parts[4])
            except Exception:
                pass
    return (cpu_dl, cpu_dw, rss_dl, rss_dw)


def build_scorecard() -> str:
    sp = get_speedups()
    def row(k: str) -> Optional[str]:
        v = sp.get(k)
        if not v:
            return None
        return f"{v[0]:.2f}$\\times$ [{v[1]:.2f},\\,{v[2]:.2f}]"

    setup = row('setup')
    processing = row('processing')
    baseline = row('baseline')
    hierarchical = row('hierarchical')
    total = sp.get('total (sem collection)') or sp.get('total_architectural')
    total_s = None
    if total:
        total_s = f"{total[0]:.2f}$\\times$ [{total[1]:.2f},\\,{total[2]:.2f}]"

    r2 = summarize_tost('') or '—'
    mase = summarize_tost('_mase') or '—'
    nrmse = summarize_tost('_nrmse') or '—'

    cpu_dl, cpu_dw, rss_dl, rss_dw = get_resources_processing()
    cpu_s = '—'
    rss_s = '—'
    if cpu_dl is not None and cpu_dw is not None:
        cpu_s = f"CPU(proc) média: DL={cpu_dl:.1f}\\%, DW={cpu_dw:.1f}\\%"
    if rss_dl is not None and rss_dw is not None:
        rss_s = f"RSS (MB): DL={rss_dl:.1f}, DW={rss_dw:.1f}"

    parts = []
    parts.append('% Auto-generated scorecard')
    parts.append('\\begin{table}[htbp]')
    parts.append('\\centering')
    parts.append('\\caption{Painel de evidências (resumo consolidado)}')
    parts.append('\\label{tab:architectural-scorecard}')
    parts.append('\\begin{tabular}{p{0.22\\linewidth}p{0.73\\linewidth}}')
    parts.append('\\toprule')
    speed_lines = []
    if setup: speed_lines.append(f"Setup: {setup}")
    if processing: speed_lines.append(f"Processing: {processing}")
    if baseline: speed_lines.append(f"Baseline: {baseline}")
    if hierarchical: speed_lines.append(f"Hierarchical: {hierarchical}")
    if total_s: speed_lines.append(f"Total: {total_s}")
    parts.append('\\textbf{Speedup por fase} & ' + '; '.join(speed_lines) + '. \\\\ ')
    parts.append('\\textbf{Equivalência (TOST)} & ' + f"R$^2$: {r2}; MASE: {mase}; nRMSE: {nrmse}. Notas: quando diferenças por fold são numericamente nulas, $p$-valores podem aparecer como $<10^{{-12}}$ e ICs estreitos." + ' \\\\ ')
    parts.append('\\textbf{Recursos (processing)} & ' + f"{cpu_s}; {rss_s}." + ' \\\\ ')
    parts.append('\\bottomrule')
    parts.append('\\end{tabular}')
    parts.append('\\end{table}')
    return '\n'.join(parts) + '\n'


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    content = build_scorecard()
    (BASE / 'architectural_scorecard.tex').write_text(content, encoding='utf-8')
    print('saved', BASE / 'architectural_scorecard.tex')


if __name__ == '__main__':
    main()
