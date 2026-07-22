#!/usr/bin/env python3
"""What a cross-paradigm table is allowed to compare.

Collection runs once, upstream of the paradigms, and its row carries the
sentinel architecture ``both`` rather than a paradigm name. Four files decided
independently whether to exclude it, and one of them did not: the throughput
table rendered a ``collection`` block with ``both`` standing among the three
paradigms as though it were a fourth, and three empty cells beside it.

The latency deriver excluded the phase and then undid it: when the filter
emptied the frame it fell back to the unfiltered one, so a run that produced
only collection rows would have had its latency table built from them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import (COMPARABLE_PHASES, SHARED_ARCHITECTURE,
                                    comparable_rows, discover_paradigms)

PARADIGMS = sorted(discover_paradigms())


def _frame(include_collection=True):
    rows = []
    if include_collection:
        rows.append({'run_id': -1, 'phase': 'collection',
                     'architecture': SHARED_ARCHITECTURE,
                     'duration_s': 50.0, 'records': 1000})
    rows += [{'run_id': run, 'phase': phase, 'architecture': paradigm,
              'duration_s': 1.0 + index, 'records': 1000}
             for run in range(6) for phase in ('processing', 'baseline')
             for index, paradigm in enumerate(PARADIGMS)]
    return pd.DataFrame(rows)


class TestComparableRows:

    def test_the_collection_phase_is_dropped(self):
        assert 'collection' not in set(
            comparable_rows(_frame())['phase'].unique())

    def test_the_sentinel_architecture_is_dropped(self):
        """On a comparable phase, where the phase filter does not cover it.

        Today the sentinel only appears on collection, so the phase filter
        removes it either way and this check would pass vacuously. It stops
        being vacuous the moment a shared step is measured inside a comparable
        phase -- which is the case the filter exists for.
        """
        frame = _frame()
        frame.loc[len(frame)] = {'run_id': 0, 'phase': 'setup',
                                 'architecture': SHARED_ARCHITECTURE,
                                 'duration_s': 9.0, 'records': 1000}
        assert SHARED_ARCHITECTURE in set(
            frame[frame['phase'].isin(COMPARABLE_PHASES)]['architecture'])
        assert SHARED_ARCHITECTURE not in set(
            comparable_rows(frame)['architecture'].unique())

    def test_the_sentinel_on_collection_is_dropped_too(self):
        assert SHARED_ARCHITECTURE not in set(
            comparable_rows(_frame())['architecture'].unique())

    def test_every_paradigm_survives(self):
        """Dropping rows is how a paradigm disappears from a table."""
        kept = set(comparable_rows(_frame())['architecture'].unique())
        assert kept == set(PARADIGMS)

    def test_an_unknown_architecture_halts(self):
        frame = _frame()
        frame.loc[len(frame)] = {'run_id': 0, 'phase': 'setup',
                                 'architecture': 'spark_sql',
                                 'duration_s': 1.0, 'records': 10}
        with pytest.raises(ValueError, match='registro não conhece'):
            comparable_rows(frame)

    def test_a_frame_with_nothing_comparable_halts(self):
        """The latency deriver used to fall back to the unfiltered frame."""
        frame = _frame(include_collection=True).head(1)
        with pytest.raises(ValueError, match='Nenhuma linha comparável'):
            comparable_rows(frame)

    def test_the_phase_list_matches_what_the_benchmark_runs(self):
        from benchmarking.architectural_benchmark import BenchmarkRunner
        assert set(BenchmarkRunner._DOWNSTREAM_PHASES) == set(
            COMPARABLE_PHASES)


class TestOneDefinitionOfThePolicy:
    """Four files enumerated it; one had already diverged."""

    CONSUMERS = [
        'benchmarking/derive_latency_percentiles.py',
        'benchmarking/derive_throughput_percentiles.py',
        'benchmarking/architectural_benchmark.py',
        'statistical_validation/significance_tests.py',
        'statistical_validation/effect_analysis.py',
    ]

    @pytest.mark.parametrize('relative', CONSUMERS)
    def test_no_file_names_the_excluded_phase_itself(self, relative):
        import ast as ast_module
        source = (_SRC / relative).read_text()
        tree = ast_module.parse(source)
        docstrings = {id(node.value) for node in ast_module.walk(tree)
                      if isinstance(node, ast_module.Expr)
                      and isinstance(node.value, ast_module.Constant)}
        for node in ast_module.walk(tree):
            if not (isinstance(node, ast_module.Constant)
                    and isinstance(node.value, str)):
                continue
            if id(node) in docstrings or node.value != 'collection':
                continue
            # architectural_benchmark runs the phase, so it may name it; what
            # it may not do is redefine which phases are comparable.
            if relative.endswith('architectural_benchmark.py'):
                continue
            raise AssertionError(
                f'{relative}:{node.lineno} decides on its own that '
                f'"collection" is excluded; that is what let one file forget'
            )

    @pytest.mark.parametrize('relative', CONSUMERS)
    def test_each_consumer_reads_the_shared_definition(self, relative):
        source = (_SRC / relative).read_text()
        assert 'COMPARABLE_PHASES' in source or 'comparable_rows' in source


class TestTheDerivedTables:

    def _summary(self, module_name):
        import contextlib
        import importlib
        import io
        module = importlib.import_module(module_name)
        importlib.reload(module)
        summarise = getattr(module, 'resumir', None) or module.resumir_percentis
        with contextlib.redirect_stdout(io.StringIO()):
            return summarise(_frame())

    @pytest.mark.parametrize('module_name', [
        'benchmarking.derive_latency_percentiles',
        'benchmarking.derive_throughput_percentiles',
    ])
    def test_collection_is_absent_from_the_table(self, module_name):
        summary = self._summary(module_name)
        per_phase = summary.get('per_phase', summary.get('por_fase', {}))
        assert 'collection' not in per_phase, module_name

    @pytest.mark.parametrize('module_name', [
        'benchmarking.derive_latency_percentiles',
        'benchmarking.derive_throughput_percentiles',
    ])
    def test_no_pseudo_paradigm_appears(self, module_name):
        summary = self._summary(module_name)
        per_phase = summary.get('per_phase', summary.get('por_fase', {}))
        for phase, entry in per_phase.items():
            listed = set(entry.get('architectures', {}))
            assert SHARED_ARCHITECTURE not in listed, (module_name, phase)
            assert listed <= set(PARADIGMS), (module_name, phase, listed)

    @pytest.mark.parametrize('module_name', [
        'benchmarking.derive_latency_percentiles',
        'benchmarking.derive_throughput_percentiles',
    ])
    def test_the_real_phases_survive(self, module_name):
        summary = self._summary(module_name)
        per_phase = summary.get('per_phase', summary.get('por_fase', {}))
        assert set(per_phase) == {'processing', 'baseline'}, module_name

    def test_the_fixture_would_expose_the_defect(self):
        """Without a collection row the tests above prove nothing."""
        frame = _frame()
        assert (frame['phase'] == 'collection').any()
        assert (frame['architecture'] == SHARED_ARCHITECTURE).any()
