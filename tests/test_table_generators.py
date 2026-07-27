#!/usr/bin/env python3
"""The table generators cover every paradigm and produce valid LaTeX.

Four published tables had columns fixed for two paradigms, and the third did not
appear anywhere. A fifth read a speedup key with the order reversed relative to
the one that was written, so the column came out as an em dash on every row --
and an em dash looks like missing data, not like a defect.

The scorecard looked for pre-rename pairs ('dl_vs_dw') in post-rename artifacts,
and came out with two of its three rows empty; the only row that worked was the
resources one, which is exactly the only one that had a test.

These tests check the class of defect: no generator names a paradigm, all of
them cover the ones the registry knows about, and the LaTeX column specification
matches the number of cells in each row.
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
        """Docstrings may cite one; code may not."""
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
                    f'{module_name}:{node.lineno} names {paradigm!r}; a fourth '
                    f'paradigm would be left out of the table with nothing '
                    f'reporting it'
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
                    f'{module_name}:{node.lineno} uses {stale!r}, which names '
                    f'paradigms that stopped existing in the rename'
                )

    @pytest.mark.parametrize('module_name', GENERATORS)
    def test_the_registry_is_consulted(self, module_name):
        source = _source(module_name)
        assert 'discover_paradigms' in source or 'paradigm_pairs' in source


class TestSpeedupKeysRoundTrip:
    """The key that is written and the key that is read must be the same."""

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
        """The column came out as an em dash on every row; an em dash looked
        like data.
        """
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
            f'speedups missing while the data is present: {values}'
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
                f'{module_name} does not render {paradigm}; that was the case '
                f'of dataframe_lib in the published tables'
            )


class TestLatexIsWellFormed:
    """Column specification against cells per row."""

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
                f'{module_name}: spec declares {columns} columns, rows have '
                f'{sorted(widths)} -- it misaligns or it does not compile'
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
        """Two of three rows came out empty and nothing reported it."""
        from statistical_validation import make_scorecard as module

        monkeypatch.setattr(module, 'get_speedups',
                            lambda: {'dl_vs_dw': {'processing': (1.0, 0.9, 1.1)}})
        with pytest.raises(KeyError, match='No pair from the registry'):
            module.build_scorecard()

    def test_no_latex_parsing_fallback_remains(self):
        """It recovered numbers from the table that another script renders."""
        source = _source('statistical_validation.make_scorecard')
        assert 'parse_significance_tex' not in source
        assert 'significance_summary.tex' not in source


class TestTheOperationalPanelIsWellFormed:
    """It was built inside main(), so no test reached the table.

    A bare % from _fmt_pct opened a LaTeX comment: everything after it on the
    row vanished, including the remaining columns and the \\\\ that ends the
    line. The table still rendered, with fewer columns than its own
    specification declares.
    """

    @staticmethod
    def _table():
        module = importlib.import_module(
            'benchmarking.derive_operational_panel')
        importlib.reload(module)
        latency = {
            'per_phase': {
                phase: {'architectures': {paradigm: {'p50': 1.0 + index}
                                          for index, paradigm
                                          in enumerate(PARADIGMS)}}
                for phase in ('processing', 'setup')},
            'total': {'architectures': {paradigm: {'p50': 9.0}
                                        for paradigm in PARADIGMS}},
        }
        resources = {phase: {paradigm: {'cpu_proc_mean': 83.4,
                                        'rss_mb_mean': 512.0}
                             for paradigm in PARADIGMS}
                     for phase in ('processing', 'setup')}
        return module.para_latex(latency, resources, PARADIGMS)

    def test_no_unescaped_percent(self):
        for line in self._table().splitlines():
            if line.strip().startswith('%'):
                continue
            assert not re.search(r'(?<!\\)%', line), line

    def test_the_percentages_are_actually_present(self):
        """Otherwise emitting no percent at all would pass the test above."""
        assert r'\%' in self._table()

    def test_no_unescaped_underscore(self):
        for line in self._table().splitlines():
            if line.strip().startswith('%'):
                continue
            assert not re.search(r'(?<!\\)_', line), line

    def test_column_counts_agree(self):
        table = self._table()
        spec = re.search(r'\\begin\{tabular\}\{([lrc|]+)\}', table)
        assert spec
        columns = len([c for c in spec.group(1) if c in 'lrc'])
        body = [line for line in table.splitlines()
                if '&' in line and not line.strip().startswith('%')]
        assert body
        widths = {line.count('&') + 1 for line in body}
        assert widths == {columns}, (
            f'spec declares {columns} columns, rows have {sorted(widths)}'
        )

    def test_every_row_terminates(self):
        """The lost \\\\ was the second casualty of the comment."""
        for line in self._table().splitlines():
            if '&' in line and not line.strip().startswith('%'):
                assert line.rstrip().endswith('\\\\'), line

    def test_every_paradigm_has_a_row_per_phase(self):
        table = self._table()
        for paradigm in PARADIGMS:
            escaped = paradigm.replace('_', r'\_')
            # two phases plus the total block
            assert table.count(escaped) == 3, paradigm
