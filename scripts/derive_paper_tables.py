#!/usr/bin/env python3
"""Gera a tabela de latência do paper a partir dos artefatos, para os dois painéis.

Por que existe: transcrever célula a célula é o mecanismo pelo qual uma tabela
publicada deixa de corresponder aos dados sem que nada acuse. Esta tabela é
derivada, e carrega na legenda o commit, o instante e o orçamento de núcleos de
cada execução -- toda latência é condicional a eles.

Roda depois dos dois pipelines, porque atravessa datasets: os artefatos de cada um
vivem sob outputs/<dataset>/, e uma tabela que compara escalas precisa dos dois.

Duas decisões explícitas, que a transcrição escondia:

  * O vencedor por estágio é computado, não marcado à mão.
  * A coluna de p reporta o **maior** valor entre os pares, após Bonferroni sobre
    a família inteira. A afirmação de um estágio é "os paradigmas diferem aqui", e
    ela exige que *todos* os pares difiram -- o maior p é o que a limita. Um p
    único e não qualificado deixa ambíguo qual par ele descreve.

Uso:
    python scripts/derive_paper_tables.py [--datasets worldbank inep_censo]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))

from core.paradigm_registry import discover_paradigms  # noqa: E402

# A ordem das fases na tabela segue o pipeline, não a ordem alfabética.
STAGE_ORDER = ['processing', 'setup', 'baseline', 'hierarchical']
ALPHA = 0.05


def _dataset_root(dataset: str) -> Path:
    return _ROOT / 'outputs' / dataset


def _require_commit(commit: str, dataset: str) -> str:
    """Uma tabela de latência sem o commit que a produziu não é comparável.

    `write_environment_snapshot` grava 'unavailable' quando não consegue
    resolver o commit. A legenda truncava para dez caracteres, então o artefato
    publicado dizia "em unavailabl" -- uma cadeia sem significado, no lugar
    exato onde o leitor procura a procedência.
    """
    if not commit or commit == 'unavailable':
        raise ValueError(
            f"{dataset}: o snapshot não registra o commit "
            f"(git_commit={commit!r}). A tabela de latência é condicional ao "
            f"código que a produziu; sem isso ela não pode ser publicada. "
            f"Rode o pipeline de um clone git com a árvore limpa."
        )
    return commit


def _read(dataset: str) -> Optional[Dict]:
    """Benchmark, significância e procedência de um dataset."""
    root = _dataset_root(dataset)
    benchmark = root / 'benchmarks' / 'architectural_benchmark_results.csv'
    significance = root / 'statistics' / 'significance_summary.csv'
    snapshot = root / 'scientific_config_snapshot.json'

    missing = [p.name for p in (benchmark, significance, snapshot)
               if not p.exists()]
    if missing:
        print(f"  [WARN] {dataset}: ausente {missing}")
        return None

    provenance = json.loads(snapshot.read_text())
    config = provenance.get('scientific_config', {})
    return {
        'benchmark': pd.read_csv(benchmark),
        'significance': pd.read_csv(significance),
        # Sem procedência não se publica a tabela: uma latência sem o commit e o
        # orçamento que a produziram não é comparável a nada.
        'commit': _require_commit(provenance['git_commit'], dataset),
        'timestamp': provenance['timestamp'],
        'engine_threads': config['engine_threads'],
        'blas_threads': config['blas_threads'],
    }


def _stage_rows(data: Dict, paradigms: List[str]) -> List[Dict]:
    benchmark = data['benchmark']
    significance = data['significance']
    # A família de comparações é o conjunto inteiro de testes reportados, e não
    # os pares de um estágio: o limiar depende dela.
    family_size = len(significance)
    threshold = ALPHA / family_size if family_size else float('nan')

    rows = []
    for stage in STAGE_ORDER:
        subset = benchmark[benchmark['phase'] == stage]
        if subset.empty:
            continue
        cells = {}
        for paradigm in paradigms:
            values = subset[subset['architecture'] == paradigm]['duration_s']
            if values.empty:
                continue
            cells[paradigm] = (float(values.mean()), float(values.std(ddof=1)))
        if not cells:
            continue

        stage_tests = significance[significance['phase'] == stage]
        # Piso do signed-rank bilateral: 2/2^n. O n é o das diferenças
        # não-nulas, não o dos pares -- o teste descarta os empates, e o piso
        # calculado sobre os pares subestima o menor p alcançável. Com três
        # empates em dez ele erra por um fator de oito.
        if not stage_tests.empty:
            if 'n_nonzero_diffs' not in stage_tests.columns:
                raise ValueError(
                    f"Estágio '{stage}': o resumo de significância não traz "
                    f"n_nonzero_diffs. É um artefato anterior a esta coluna, e "
                    f"o piso derivado do número de pares subestima o menor p "
                    f"alcançável. Regere o resumo."
                )
            n_pairs = int(stage_tests['n_nonzero_diffs'].min())
        else:
            n_pairs = 0

        p_column = ('wilcoxon_p' if 'wilcoxon_p' in stage_tests.columns
                    else 't_p')
        # O maior p entre os pares limita a afirmação do estágio: ela é "os
        # paradigmas diferem aqui", e isso exige que *todos* os pares difiram.
        #
        # skipna=False de propósito. Um par cujo teste não pôde ser computado
        # -- diferenças todas nulas, n abaixo do mínimo -- sumia do máximo, e o
        # estágio saía mais significativo do que a família sustenta. Sem o
        # teste daquele par a afirmação do estágio não está estabelecida.
        worst_p = (float(stage_tests[p_column].max(skipna=False))
                   if not stage_tests.empty else float('nan'))
        untested = (int(stage_tests[p_column].isna().sum())
                    if not stage_tests.empty else 0)

        # Piso e limiar são independentes: o piso vem das repetições, o limiar
        # vem do tamanho da família, que cresce com o número de paradigmas. Com
        # um quarto paradigma a família passa de 15 para 30 e o limiar cai
        # abaixo do piso -- nenhum estágio pode ser significativo, qualquer que
        # seja o dado. Sair em silêncio nessa condição é reportar ausência de
        # diferença quando o que houve foi ausência de resolução.
        floor = (2.0 / 2 ** n_pairs) if n_pairs else float('nan')
        if n_pairs and floor > threshold:
            raise ValueError(
                f"Estágio '{stage}': o piso do Wilcoxon bilateral com "
                f"{n_pairs} pares é {floor:.5f}, acima do limiar corrigido "
                f"{threshold:.5f} (alpha={ALPHA} sobre família de "
                f"{family_size}). Nenhum estágio pode ser significativo nesta "
                f"configuração. Aumente as repetições para "
                f"{math.ceil(math.log2(2.0 / threshold))} ou mais, ou reduza a "
                f"família."
            )
        rows.append({
            'stage': stage,
            'cells': cells,
            'winner': min(cells, key=lambda p: cells[p][0]),
            'pairs_tested': int(len(stage_tests)),
            'worst_pair_p': worst_p,
            'family_size': int(family_size),
            'n_observations': n_pairs,
            'wilcoxon_floor': floor,
            'threshold': threshold,
            # Uma única formulação: p bruto contra alpha/m. Reportar também o p
            # multiplicado convidaria a compará-lo com 0,05 por hábito, e as duas
            # leituras misturadas é como uma célula passa a dizer duas coisas.
            'pairs_untested': untested,
            'p_bonferroni_equivalent': min(1.0, worst_p * family_size)
                                       if family_size else worst_p,
            # NaN < x é False, então um estágio com par não testado já saía
            # não-significativo -- mas por acidente da comparação, e depois de
            # o máximo ter escondido o par. Explícito agora.
            'significant': bool(np.isfinite(worst_p) and worst_p < threshold),
        })
    return rows


def _fmt(mean: float, sd: float, winner: bool) -> str:
    body = f"{mean:.3g}{{\\tiny$\\pm${sd:.3g}}}"
    return f"\\textbf{{{body}}}" if winner else body


def build(datasets: List[str]) -> Dict:
    paradigms = sorted(discover_paradigms())
    report = {'paradigms': paradigms, 'datasets': {}}
    for dataset in datasets:
        data = _read(dataset)
        if data is None:
            continue
        report['datasets'][dataset] = {
            'commit': data['commit'],
            'timestamp': data['timestamp'],
            'engine_threads': data['engine_threads'],
            'blas_threads': data['blas_threads'],
            'stages': _stage_rows(data, paradigms),
        }
    return report


def to_latex(report: Dict) -> str:
    paradigms = report['paradigms']
    present = report['datasets']
    if not present:
        return ''

    budgets = {(d['engine_threads'], d['blas_threads'])
               for d in present.values()}
    if len(budgets) > 1:
        raise ValueError(
            f"Os painéis foram medidos com orçamentos de núcleos diferentes "
            f"{budgets}: as latências não são comparáveis entre eles."
        )
    engine, blas = budgets.pop()
    thresholds = {row['threshold'] for d in present.values()
                  for row in d['stages']}
    threshold_text = ('/'.join(f'{t:.5f}' for t in sorted(thresholds))
                      if thresholds else 'n/d')
    provenance = '; '.join(
        f"{name.replace('_', chr(92) + '_')} em {d['commit'][:10]} "
        f"({d['timestamp'][:19]})"
        for name, d in sorted(present.items()))

    lines = [
        '% Gerado por scripts/derive_paper_tables.py -- nao editar a mao',
        '\\begin{table}[htb]',
        '\\centering',
        '\\caption{Latência por estágio (média $\\pm$ SD, segundos). '
        '\\textbf{Negrito}: menor média do estágio, computada. '
        '$p$: maior valor entre os pares (Wilcoxon), contra o limiar de '
        f'Bonferroni {threshold_text}. '
        f'{engine} núcleos por engine, {blas} thread de BLAS. '
        f'Procedência: {provenance}.}}',
        '\\label{tab:latency}',
        '\\begin{tabular}{ll' + 'r' * len(paradigms) + 'l}',
        '\\toprule',
        'Painel & Estágio & ' + ' & '.join(
            p.replace('_', r'\_') for p in paradigms) + ' & $p$ \\\\',
        '\\midrule',
    ]
    for dataset, data in sorted(present.items()):
        stages = data['stages']
        for index, row in enumerate(stages):
            # Escape obrigatório: um nome de dataset com underscore quebra a
            # compilação, e a tabela é gerada, não revisada à mão.
            safe = dataset.replace('_', r'\_')
            label = (f"\\multirow{{{len(stages)}}}{{*}}{{{safe}}}"
                     if index == 0 else '')
            cells = [
                _fmt(*row['cells'][p], p == row['winner'])
                if p in row['cells'] else '---'
                for p in paradigms
            ]
            floor = row['wilcoxon_floor']
            at_floor = floor == floor and row['worst_pair_p'] <= floor
            if row['significant']:
                # No piso do teste: reportar "=" ao piso em vez de dígitos que
                # sugerem precisão que n não oferece.
                marker = (f"{floor:.5f} (piso, $n$={row['n_observations']})"
                          if at_floor else f"{row['worst_pair_p']:.4f}")
            else:
                # Explícito em vez de omitido: um p acima do limiar corrigido não
                # sustenta a diferença, e o negrito da célula não deve sugerir
                # que sustenta.
                marker = f"{row['worst_pair_p']:.4f} (n.s.)"
            lines.append(f"{label} & {row['stage']} & "
                         + ' & '.join(cells) + f" & {marker} \\\\")
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines += ['\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+',
                        default=['worldbank', 'inep_censo'])
    parser.add_argument('--out-dir', default=str(_ROOT / 'outputs' / 'tables'))
    args = parser.parse_args(argv)

    report = build(args.datasets)
    if not report['datasets']:
        print("  Nenhum painel com artefatos completos; nada a gerar.")
        return 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'latency_table.json').write_text(json.dumps(report, indent=2))
    (out / 'latency_table.tex').write_text(to_latex(report))
    print(json.dumps({
        'status': 'ok',
        'datasets': sorted(report['datasets']),
        'tex': str(out / 'latency_table.tex'),
        'json': str(out / 'latency_table.json'),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
