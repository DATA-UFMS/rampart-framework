#!/usr/bin/env python3
"""Os geradores de tabela cobrem todos os paradigmas e produzem LaTeX válido.

Quatro tabelas publicadas tinham colunas fixas para dois paradigmas, e o terceiro
não aparecia em lugar nenhum. Uma quinta lia uma chave de speedup com a ordem
invertida em relação à que era escrita, então a coluna saía em travessão em toda
linha -- e travessão parece dado ausente, não defeito.

O scorecard procurava pares pré-rename ('dl_vs_dw') em artefatos pós-rename, e
saía com duas das três linhas vazias; a única linha que funcionava era a de
recursos, que é exatamente a única que tinha teste.

Estes testes verificam a classe do defeito: nenhum gerador nomeia um paradigma,
todos cobrem os que o registro conhece, e a especificação de colunas do LaTeX
bate com o número de células de cada linha.
"""

import ast
import importlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import discover_paradigms, paradigm_pairs

GENERATORS = [
    'benchmarking.derive_latency_percentiles',
    'benchmarking.derive_throughput_percentiles',
    'benchmarking.derive_operational_panel',
    'statistical_validation.make_scorecard',
]
PARADIGMS = sorted(discover_paradigms())


def _source(module_name):
    return (_SRC / (module_name.replace('.', '/') + '.py')).read_text()


class TestNoGeneratorNamesAParadigm:

    @pytest.mark.parametrize('module_name', GENERATORS)
    def test_no_paradigm_literal_in_code(self, module_name):
        """Docstrings podem citar; código não."""
        tree = ast.parse(_source(module_name))
        docstrings = {id(n.value) for n in ast.walk(tree)
                      if isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Constant)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            if id(node) in docstrings:
                continue
            for paradigm in PARADIGMS:
                assert paradigm not in node.value, (
                    f'{module_name}:{node.lineno} nomeia {paradigm!r}; um quarto '
                    f'paradigma ficaria fora da tabela sem que nada acusasse'
                )

    @pytest.mark.parametrize('module_name', GENERATORS)
    def test_no_pre_rename_abbreviation(self, module_name):
        source = _source(module_name)
        tree = ast.parse(source)
        docstrings = {id(n.value) for n in ast.walk(tree)
                      if isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Constant)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)) or id(node) in docstrings:
                continue
            for stale in ('dl_vs_dw', 'dw_vs_pl', 'dl_vs_pl',
                          'DL vs DW', 'DL P50', 'DW P50'):
                assert stale not in node.value, (
                    f'{module_name}:{node.lineno} usa {stale!r}, que nomeia '
                    f'paradigmas que deixaram de existir no rename'
                )

    @pytest.mark.parametrize('module_name', GENERATORS)
    def test_the_registry_is_consulted(self, module_name):
        source = _source(module_name)
        assert 'discover_paradigms' in source or 'paradigm_pairs' in source


class TestSpeedupKeysRoundTrip:
    """Chave escrita e chave lida têm de ser a mesma."""

    def test_written_and_read_keys_match(self):
        module = importlib.import_module(
            'benchmarking.derive_latency_percentiles')
        importlib.reload(module)
        summarise = module.resumir_percentis
        rows = [{'run_id': run, 'phase': phase, 'architecture': paradigm,
                 'duration_s': 1.0 + index, 'records': 100}
                for run in range(6) for phase in ('processing', 'baseline')
                for index, paradigm in enumerate(PARADIGMS)]
        summary = summarise(pd.DataFrame(rows))

        expected = {f'{a}_vs_{b}' for a, b in paradigm_pairs()}
        for phase, entry in summary['per_phase'].items():
            assert set(entry['speedups_p50']) == expected, phase
        assert set(summary['total']['speedups_p50']) == expected

    def test_the_speedups_are_not_all_absent(self):
        """A coluna saía em travessão em toda linha; travessão parecia dado."""
        module = importlib.import_module(
            'benchmarking.derive_latency_percentiles')
        importlib.reload(module)
        summarise = module.resumir_percentis
        rows = [{'run_id': run, 'phase': 'processing', 'architecture': paradigm,
                 'duration_s': 1.0 + index, 'records': 100}
                for run in range(6)
                for index, paradigm in enumerate(PARADIGMS)]
        summary = summarise(pd.DataFrame(rows))
        values = summary['per_phase']['processing']['speedups_p50'].values()
        assert all(v is not None for v in values), (
            f'speedups ausentes com dados presentes: {values}'
        )

    def test_the_table_shows_them(self):
        module = importlib.import_module(
            'benchmarking.derive_latency_percentiles')
        importlib.reload(module)
        summarise = module.resumir_percentis
        rows = [{'run_id': run, 'phase': 'processing', 'architecture': paradigm,
                 'duration_s': 1.0 + index, 'records': 100}
                for run in range(6)
                for index, paradigm in enumerate(PARADIGMS)]
        table = module.para_latex(summarise(pd.DataFrame(rows)))
        speedup_block = table.split('% Speedup')[-1]
        assert '—' not in speedup_block, speedup_block


class TestEveryParadigmAppears:

    def _table(self, module_name):
        module = importlib.import_module(module_name)
        importlib.reload(module)
        rows = [{'run_id': run, 'phase': phase, 'architecture': paradigm,
                 'duration_s': 1.0 + index, 'records': 1000}
                for run in range(6) for phase in ('processing', 'baseline')
                for index, paradigm in enumerate(PARADIGMS)]
        summarise = getattr(module, 'resumir', None) or module.resumir_percentis
        return module.para_latex(summarise(pd.DataFrame(rows)))

    @pytest.mark.parametrize('module_name', [
        'benchmarking.derive_latency_percentiles',
        'benchmarking.derive_throughput_percentiles',
    ])
    def test_all_paradigms_are_rendered(self, module_name):
        table = self._table(module_name)
        for paradigm in PARADIGMS:
            escaped = paradigm.replace('_', r'\_')
            assert escaped in table, (
                f'{module_name} não renderiza {paradigm}; era o caso de '
                f'dataframe_lib nas tabelas publicadas'
            )


class TestLatexIsWellFormed:
    """Especificação de colunas contra células por linha."""

    @pytest.mark.parametrize('module_name', [
        'benchmarking.derive_latency_percentiles',
        'benchmarking.derive_throughput_percentiles',
    ])
    def test_column_counts_agree(self, module_name):
        module = importlib.import_module(module_name)
        importlib.reload(module)
        rows = [{'run_id': run, 'phase': phase, 'architecture': paradigm,
                 'duration_s': 1.0 + index, 'records': 1000}
                for run in range(6) for phase in ('processing', 'baseline')
                for index, paradigm in enumerate(PARADIGMS)]
        summarise = getattr(module, 'resumir', None) or module.resumir_percentis
        table = module.para_latex(summarise(pd.DataFrame(rows)))

        for block in table.split(r'\begin{tabular}')[1:]:
            spec = re.match(r'\{(?:@\{\})?([lrc|]+)(?:@\{\})?\}', block)
            assert spec, block[:80]
            columns = len([c for c in spec.group(1) if c in 'lrc'])
            body = [line for line in block.splitlines()
                    if '&' in line and not line.strip().startswith('%')]
            widths = {line.count('&') + 1 for line in body}
            assert widths == {columns}, (
                f'{module_name}: spec declara {columns} colunas, linhas têm '
                f'{sorted(widths)} -- desalinha ou não compila'
            )

    @pytest.mark.parametrize('module_name', [
        'benchmarking.derive_latency_percentiles',
        'benchmarking.derive_throughput_percentiles',
    ])
    def test_no_unescaped_underscore(self, module_name):
        module = importlib.import_module(module_name)
        importlib.reload(module)
        rows = [{'run_id': run, 'phase': 'processing', 'architecture': paradigm,
                 'duration_s': 1.0 + index, 'records': 1000}
                for run in range(6)
                for index, paradigm in enumerate(PARADIGMS)]
        summarise = getattr(module, 'resumir', None) or module.resumir_percentis
        table = module.para_latex(summarise(pd.DataFrame(rows)))
        for line in table.splitlines():
            if line.strip().startswith('%'):
                continue
            assert not re.search(r'(?<!\\)_', line), line


class TestScorecardFailsLoudOnNoMatch:

    def test_it_raises_rather_than_emitting_dashes(self, monkeypatch):
        """Duas de três linhas saíam vazias e nada acusava."""
        from statistical_validation import make_scorecard as module

        monkeypatch.setattr(module, 'get_speedups',
                            lambda: {'dl_vs_dw': {'processing': (1.0, 0.9, 1.1)}})
        with pytest.raises(KeyError, match='Nenhum par'):
            module.build_scorecard()

    def test_no_latex_parsing_fallback_remains(self):
        """Recuperava números da tabela que outro script renderiza."""
        source = _source('statistical_validation.make_scorecard')
        assert 'parse_significance_tex' not in source
        assert 'significance_summary.tex' not in source
