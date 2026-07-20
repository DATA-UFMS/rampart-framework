#!/usr/bin/env python3
"""
Geração automática do painel consolidado (scorecard) em LaTeX a partir dos
artefatos já produzidos pelo pipeline.

Entradas (melhor esforço, com fallbacks):
 - outputs/statistics/significance_summary.json ou .tex (speedups + IC95 por fase)
 - outputs/statistics/equivalence_estimation.json (equivalência por estimativa SESOI+IC)
 - outputs/statistics/architectural_resource_usage.tex (CPU(proc) média e RSS para processing)

Saída:
 - outputs/statistics/architectural_scorecard.tex

Uso:
 - Execute diretamente: `python src/statistical_validation/make_scorecard.py`
 - Integrado ao pipeline: chamado automaticamente após o benchmark.
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
    """Retorna {pair_key: {phase: (speedup, lo, hi)}}"""
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
                # Detectar chave de speedup dinamicamente
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
    # Sem fallback por parsing de LaTeX. Ele recuperava números da tabela que
    # outro script renderiza e os chaveava sob 'dl_vs_dw', um par que deixou de
    # existir no rename -- então o resultado nunca casava com nada e a ausência
    # do JSON aparecia como scorecard vazio em vez de como ausência.
    return {}


def summarize_equivalence(metric: str, pair_key: str) -> Optional[str]:
    """Lê equivalence_estimation.json e retorna resumo para a métrica dada."""
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
    status = 'Sim' if 'equivalen' in decision.lower() else 'Não'
    return f"{status} ($\\delta={delta:.3f}$, IC=[{ci[0]:.3f},{ci[1]:.3f}])"


def get_resources_processing(phase: str = 'processing') -> Dict[str, Dict[str, Optional[float]]]:
    """CPU e RSS por paradigma numa fase, lidos do JSON.

    Antes isto parseava a tabela LaTeX que outro script gera, recuperando por
    texto números que existem em JSON. Duas consequências: qualquer mudança de
    formato quebrava a extração em silêncio, e o nome de um paradigma ficou
    desatualizado -- procurava-se 'processing & polars', que deixou de existir no
    rename, então o terceiro paradigma nunca era extraído e o `except` escondia.

    Os nomes vêm do registro, e não de literais, pelo mesmo motivo.
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

    # Pares derivados do registro. As chaves literais eram pré-rename
    # ('dl_vs_dw' etc.) enquanto os artefatos passaram a usar os nomes dos
    # paradigmas, então nenhum par casava e o scorecard saía com travessão em
    # duas das três linhas -- 12 speedups e 9 decisões SESOI perdidos em silêncio.
    pairs = [(f'{left}_vs_{right}',
              f"{left.replace('_', chr(92) + '_')} vs {right.replace('_', chr(92) + '_')}")
             for left, right in paradigm_pairs()]
    if speedups_by_pair and not any(k in speedups_by_pair for k, _ in pairs):
        raise KeyError(
            f"Nenhum par do registro {[k for k, _ in pairs]} aparece nos "
            f"artefatos {sorted(speedups_by_pair)}: o scorecard sairia vazio."
        )

    # Linhas de speedup por par
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

    # Linhas de equivalência por par
    equiv_rows = {}
    for pair_key, pair_label in pairs:
        r2 = summarize_equivalence('r2', pair_key) or '—'
        mase = summarize_equivalence('mase', pair_key) or '—'
        wape = summarize_equivalence('wape', pair_key) or '—'
        equiv_rows[pair_key] = f"R$^2$: {r2}; MASE: {mase}; WAPE: {wape}"

    # Informações de recursos
    # Uma linha por paradigma presente, nomeado: as abreviações DL/DW/PL
    # designavam os nomes anteriores ao rename.
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

    # Tabela 3-way
    parts = []
    parts.append('% Gerado automaticamente')
    parts.append('\\begin{table}[htbp]')
    parts.append('\\centering')
    parts.append('\\caption{Painel de evidências (resumo consolidado 3-way)}')
    parts.append('\\label{tab:architectural-scorecard}')
    width = 0.78 / max(len(pairs), 1)
    parts.append('\\begin{tabular}{p{0.18\\linewidth}'
                 + f'p{{{width:.2f}\\linewidth}}' * len(pairs) + '}')
    parts.append('\\toprule')
    parts.append(' & ' + ' & '.join(label for _, label in pairs) + ' \\\\')
    parts.append('\\midrule')
    parts.append('\\textbf{Speedup por fase} & '
                 + ' & '.join(speedup_rows[k] for k, _ in pairs) + ' \\\\')
    parts.append('\\textbf{Equivalência (SESOI+IC)} & '
                 + ' & '.join(equiv_rows[k] for k, _ in pairs) + ' \\\\')
    parts.append('\\textbf{Recursos (processing)} & \\multicolumn{3}{l}{' + resource_s + '} \\\\')
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
