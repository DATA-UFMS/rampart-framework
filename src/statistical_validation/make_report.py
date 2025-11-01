#!/usr/bin/env python3
"""
Geração de relatórios LaTeX a partir de significance_summary.json e resultados hierárquicos/TOST.
"""
import json
from pathlib import Path


def load_json(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def save_text(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def make_significance_table(signif: dict) -> str:
    lines = [
        '% Significance per phase (booktabs required)\n',
        '\\begin{table}[htbp]\n',
        '\\centering\n',
        '\\caption{Benchmark arquitetural (DW vs DL): tempos médios, diferenças, speedups e significância (n conforme execução).}\n',
        '\\label{tab:significance}\n',
        '\\begin{tabular}{lrrrrrr}\n',
        '\\toprule\n',
        'Fase & DL (s) & DW (s) & $\\Delta$ [IC95] & Speedup [IC95] & $p_t$ & $p_W$ \\\\ \n',
        '\\midrule\n'
    ]
    for phase in ['processing','setup','baseline','hierarchical','total_architectural']:
        if phase not in signif:
            continue
        s = signif[phase]
        lines.append(
            f"{phase} & {s['mean_dl_s']:.2f} & {s['mean_dw_s']:.2f} & {s['mean_diff_s']:.2f} [{s['diff_mean_ci95_lo']:.2f}, {s['diff_mean_ci95_hi']:.2f}] & "
            f"{s['speedup_dw_vs_dl']:.2f} [{s['speedup_ci95_lo']:.2f}, {s['speedup_ci95_hi']:.2f}] & {s['t_p']:.2e} & {s['wilcoxon_p']:.5f} \\\\ \n"
        )
    lines += ['\\bottomrule\n','\\end{tabular}\n','\\end{table}\n']
    return ''.join(lines)


def main() -> None:
    outdir = Path('outputs/statistics')
    signif = load_json(outdir / 'significance_summary.json')
    tex = make_significance_table(signif)
    save_text(outdir / 'significance_table.tex', tex)
    print('saved', outdir / 'significance_table.tex')


if __name__ == '__main__':
    main()

