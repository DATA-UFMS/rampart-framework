#!/usr/bin/env python3
"""
Automatic generation of the consolidated panel (scorecard) in LaTeX from the
artifacts already produced by the pipeline.

Inputs (best effort, with fallbacks):
 - outputs/statistics/significance_summary.json or .tex (speedups + 95% CI per phase)
 - outputs/statistics/equivalence_estimation.json (equivalence by SESOI+CI estimation)
 - outputs/statistics/architectural_resource_usage.tex (mean CPU(proc) and RSS for processing)

Output:
 - outputs/statistics/architectural_scorecard.tex

Usage:
 - Run directly: `python src/statistical_validation/make_scorecard.py`
 - Integrated into the pipeline: called automatically after the benchmark.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import os
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import get_absolute_output_path
from core.paradigm_registry import discover_paradigms, paradigm_pairs

BASE = Path(get_absolute_output_path('outputs/statistics'))


def load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    except Exception:
        return None


def get_speedups() -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Returns {pair_key: {phase: (speedup, lo, hi)}}"""
    j = load_json(BASE / 'significance_summary.json')
    if j:
        out: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        for pair_key, phases in j.items():
            if not isinstance(phases, dict):
                continue
            pair_speedups: Dict[str, Tuple[float, float, float]] = {}
            for phase, metrics in phases.items():
                if not isinstance(metrics, dict):
                    continue
                # Detect the speedup key dynamically
                speedup_key = [k for k in metrics if k.startswith('speedup_') and not k.endswith('_lo') and not k.endswith('_hi')]
                ci_lo_key = [k for k in metrics if k.endswith('ci95_lo') and 'speedup' in k]
                ci_hi_key = [k for k in metrics if k.endswith('ci95_hi') and 'speedup' in k]
                if speedup_key and ci_lo_key and ci_hi_key:
                    pair_speedups[phase.lower()] = (
                        float(metrics[speedup_key[0]]),
                        float(metrics[ci_lo_key[0]]),
                        float(metrics[ci_hi_key[0]]),
                    )
            if pair_speedups:
                out[pair_key] = pair_speedups
        if out:
            return out
    # No LaTeX-parsing fallback. It recovered numbers from the table that
    # another script renders and keyed them under 'dl_vs_dw', a pair that
    # stopped existing at the rename -- so the result never matched anything and
    # the absence of the JSON showed up as an empty scorecard instead of as an
    # absence.
    return {}


def summarize_equivalence(metric: str, pair_key: str) -> Optional[str]:
    """Reads equivalence_estimation.json and returns a summary for the given metric."""
    data = load_json(BASE / 'equivalence_estimation.json')
    if not data:
        return None
    pred = data.get('predictive', {})
    pair_data = pred.get(pair_key, {})
    entry = pair_data.get(metric)
    if not entry or not isinstance(entry, dict):
        return None
    decision = entry.get('decision', '?')
    delta = entry.get('delta', float('nan'))
    ci = entry.get('ci95', [float('nan'), float('nan')])
    status = 'Yes' if 'equivalen' in decision.lower() else 'No'
    return f"{status} ($\\delta={delta:.3f}$, IC=[{ci[0]:.3f},{ci[1]:.3f}])"


def get_resources_processing(phase: str = 'processing') -> Dict[str, Dict[str, Optional[float]]]:
    """CPU and RSS per paradigm in one phase, read from the JSON.

    This used to parse the LaTeX table that another script generates,
    recovering by text numbers that exist in JSON. Two consequences: any format
    change broke the extraction silently, and one paradigm's name went stale --
    it looked for 'processing & polars', which stopped existing at the rename,
    so the third paradigm was never extracted and the `except` hid it.

    The names come from the registry, and not from literals, for the same
    reason.
    """
    payload = load_json(BASE / 'architectural_resource_usage.json')
    if not payload:
        return {}
    per_phase = payload.get('per_phase', {}).get(phase, {})
    resources: Dict[str, Dict[str, Optional[float]]] = {}
    for paradigm in sorted(discover_paradigms()):
        row = per_phase.get(paradigm)
        if row is None:
            continue
        resources[paradigm] = {
            'cpu_proc_mean': row.get('cpu_proc_mean'),
            'rss_mb_mean': row.get('rss_mb_mean'),
        }
    return resources



def build_scorecard() -> str:
    speedups_by_pair = get_speedups()

    # Pairs derived from the registry. The literal keys were pre-rename
    # ('dl_vs_dw' etc.) while the artifacts moved to the paradigm names, so no
    # pair matched and the scorecard came out with an em dash in two of the three
    # rows -- 12 speedups and 9 SESOI decisions lost silently.
    pairs = [(f'{left}_vs_{right}',
              f"{left.replace('_', chr(92) + '_')} vs {right.replace('_', chr(92) + '_')}")
             for left, right in paradigm_pairs()]
    if speedups_by_pair and not any(k in speedups_by_pair for k, _ in pairs):
        raise KeyError(
            f"No pair from the registry {[k for k, _ in pairs]} appears in the "
            f"artifacts {sorted(speedups_by_pair)}: the scorecard would come out empty."
        )

    # Speedup rows per pair
    speedup_rows = {}
    for pair_key, pair_label in pairs:
        sp = speedups_by_pair.get(pair_key, {})
        def row(k: str) -> Optional[str]:
            v = sp.get(k)
            if not v:
                return None
            return f"{v[0]:.2f}$\\times$ [{v[1]:.2f},\\,{v[2]:.2f}]"

        speed_lines = []
        if row('setup'): speed_lines.append(f"Setup: {row('setup')}")
        if row('processing'): speed_lines.append(f"Processing: {row('processing')}")
        if row('baseline'): speed_lines.append(f"Baseline: {row('baseline')}")
        if row('hierarchical'): speed_lines.append(f"Hierarchical: {row('hierarchical')}")
        total = sp.get('total (sem collection)') or sp.get('total_architectural')
        if total:
            total_s = f"{total[0]:.2f}$\\times$ [{total[1]:.2f},\\,{total[2]:.2f}]"
            speed_lines.append(f"Total: {total_s}")
        speedup_rows[pair_key] = '; '.join(speed_lines) if speed_lines else '—'

    # Equivalence rows per pair
    equiv_rows = {}
    for pair_key, pair_label in pairs:
        r2 = summarize_equivalence('r2', pair_key) or '—'
        mase = summarize_equivalence('mase', pair_key) or '—'
        wape = summarize_equivalence('wape', pair_key) or '—'
        equiv_rows[pair_key] = f"R$^2$: {r2}; MASE: {mase}; WAPE: {wape}"

    # Resource information
    # One row per paradigm present, named: the abbreviations DL/DW/PL
    # designated the names from before the rename.
    resources = get_resources_processing()
    resource_lines = []
    for paradigm, values in sorted(resources.items()):
        cpu, rss = values['cpu_proc_mean'], values['rss_mb_mean']
        if cpu is None and rss is None:
            continue
        parts = [paradigm.replace('_', r'\_')]
        if cpu is not None:
            parts.append(f"CPU(proc) {cpu:.1f}\\%")
        if rss is not None:
            parts.append(f"RSS {rss:.1f} MB")
        resource_lines.append(' '.join(parts))
    resource_s = '; '.join(resource_lines) if resource_lines else '—'

    # 3-way table
    parts = []
    parts.append('% Generated automatically')
    parts.append('\\begin{table}[htbp]')
    parts.append('\\centering')
    parts.append('\\caption{Evidence panel (consolidated 3-way summary)}')
    parts.append('\\label{tab:architectural-scorecard}')
    width = 0.78 / max(len(pairs), 1)
    parts.append('\\begin{tabular}{p{0.18\\linewidth}'
                 + f'p{{{width:.2f}\\linewidth}}' * len(pairs) + '}')
    parts.append('\\toprule')
    parts.append(' & ' + ' & '.join(label for _, label in pairs) + ' \\\\')
    parts.append('\\midrule')
    parts.append('\\textbf{Speedup per phase} & '
                 + ' & '.join(speedup_rows[k] for k, _ in pairs) + ' \\\\')
    parts.append('\\textbf{Equivalence (SESOI+CI)} & '
                 + ' & '.join(equiv_rows[k] for k, _ in pairs) + ' \\\\')
    parts.append('\\textbf{Resources (processing)} & \\multicolumn{3}{l}{' + resource_s + '} \\\\')
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
