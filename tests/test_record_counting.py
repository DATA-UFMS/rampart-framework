#!/usr/bin/env python3
"""Record counts work for every paradigm, and failures say why.

The count located the master artifact through a filename template, which assumed
every paradigm writes a parquet. The SQL engine keeps its data in the database and
writes none, so the template pointed at a file that never existed. A bare
`except Exception: return None` then hid the reason.

The consequence is visible in the published data: `records` is absent for the ML
stages of every paradigm and for one paradigm entirely, so `throughput_rps` exists
only for collection and one processing row -- which is why the throughput percentile
table was never produced, despite the script that generates it working correctly.
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import discover_paradigms

BENCHMARK = _SRC / 'benchmarking' / 'architectural_benchmark.py'


class TestEveryParadigmDeclaresItsMaster:

    def test_all_paradigms_declare_one(self):
        for name, meta in sorted(discover_paradigms().items()):
            assert 'master_artifact' in meta, (
                f'{name} does not declare where its master data lives'
            )

    @pytest.mark.parametrize('name', sorted(discover_paradigms()))
    def test_declaration_is_well_formed(self, name):
        artifact = discover_paradigms()[name]['master_artifact']
        assert artifact['kind'] in ('parquet', 'duckdb_table')
        if artifact['kind'] == 'parquet':
            assert artifact['path'].endswith('.parquet')
        else:
            assert artifact['table'] and artifact['database']

    def test_the_sql_engine_is_not_expected_to_write_a_parquet(self):
        """It keeps its data in the database; that is the paradigm."""
        artifact = discover_paradigms()['sql_engine']['master_artifact']
        assert artifact['kind'] == 'duckdb_table'

    def test_the_frame_paradigms_write_parquet(self):
        for name in ('task_graph', 'dataframe_lib'):
            assert discover_paradigms()[name]['master_artifact']['kind'] == \
                'parquet'

    def test_declared_paths_are_distinct(self):
        located = []
        for meta in discover_paradigms().values():
            artifact = meta['master_artifact']
            located.append(artifact.get('path') or
                           (artifact['database'], artifact['table']))
        assert len(set(map(str, located))) == len(located)


class TestFailuresAreNotSwallowed:

    @pytest.mark.parametrize('function', ['_count_fold_records', '_fold_years'])
    def test_no_bare_exception_handler(self, function):
        """Checked over handlers, not text: the docstring names the pattern."""
        import ast

        tree = ast.parse(BENCHMARK.read_text())
        target = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == function)
        for handler in [n for n in ast.walk(target)
                        if isinstance(n, ast.ExceptHandler)]:
            caught = getattr(handler.type, 'id', None) if handler.type else None
            assert caught not in (None, 'Exception'), (
                f'{function} swallows {caught or "every"} exception and returns '
                f'None for any cause, which is how the throughput dimension '
                f'went unmeasured without anyone noticing'
            )

    def test_missing_folds_are_reported(self, tmp_path, monkeypatch, capsys):
        import benchmarking.architectural_benchmark as ab

        monkeypatch.setattr(ab, 'get_absolute_output_path',
                            lambda rel: str(tmp_path / rel))
        runner = ab.BenchmarkRunner.__new__(ab.BenchmarkRunner)
        assert runner._count_fold_records('task_graph') is None
        assert 'folds missing at' in capsys.readouterr().out

    def test_missing_master_is_reported(self, tmp_path, monkeypatch, capsys):
        import benchmarking.architectural_benchmark as ab

        folds = (tmp_path / 'ml_pipeline/architectures/task_graph/prep')
        folds.mkdir(parents=True)
        (folds / 'temporal_folds_task_graph.json').write_text(
            json.dumps({'folds': [{'train_start': 2000, 'train_end': 2005,
                                   'val_start': 2008, 'val_end': 2009,
                                   'test_start': 2012, 'test_end': 2013}]}))
        monkeypatch.setattr(ab, 'get_absolute_output_path',
                            lambda rel: str(tmp_path / rel))
        runner = ab.BenchmarkRunner.__new__(ab.BenchmarkRunner)
        assert runner._count_fold_records('task_graph') is None
        assert 'master missing at' in capsys.readouterr().out

    def test_an_undeclared_paradigm_raises(self, monkeypatch):
        import benchmarking.architectural_benchmark as ab

        monkeypatch.setattr(ab, 'discover_paradigms',
                            lambda **kw: {'toy': {'name': 'toy'}})
        runner = ab.BenchmarkRunner.__new__(ab.BenchmarkRunner)
        with pytest.raises(KeyError, match='master_artifact'):
            runner._fold_years('toy')

    def test_an_unknown_artifact_kind_raises(self, monkeypatch):
        import benchmarking.architectural_benchmark as ab

        monkeypatch.setattr(
            ab, 'discover_paradigms',
            lambda **kw: {'toy': {'master_artifact': {'kind': 'csv'}}})
        runner = ab.BenchmarkRunner.__new__(ab.BenchmarkRunner)
        with pytest.raises(ValueError,
                           match='master_artifact of unknown kind'):
            runner._fold_years('toy')


class TestCountingIsCorrect:

    def test_counts_every_window_of_every_fold(self, tmp_path, monkeypatch):
        import benchmarking.architectural_benchmark as ab
        import pandas as pd

        prep = tmp_path / 'ml_pipeline/architectures/task_graph/prep'
        prep.mkdir(parents=True)
        # One row per year, 2000-2013 inclusive.
        pd.DataFrame({'year': list(range(2000, 2014))}).to_parquet(
            prep / 'master_data_task_graph.parquet')
        (prep / 'temporal_folds_task_graph.json').write_text(json.dumps({'folds': [
            {'train_start': 2000, 'train_end': 2005,    # 6
             'val_start': 2008, 'val_end': 2009,        # 2
             'test_start': 2012, 'test_end': 2013},     # 2
        ]}))
        monkeypatch.setattr(ab, 'get_absolute_output_path',
                            lambda rel: str(tmp_path / rel))
        runner = ab.BenchmarkRunner.__new__(ab.BenchmarkRunner)
        assert runner._count_fold_records('task_graph') == 10

    def test_folds_accumulate(self, tmp_path, monkeypatch):
        import benchmarking.architectural_benchmark as ab
        import pandas as pd

        prep = tmp_path / 'ml_pipeline/architectures/task_graph/prep'
        prep.mkdir(parents=True)
        pd.DataFrame({'year': list(range(2000, 2014))}).to_parquet(
            prep / 'master_data_task_graph.parquet')
        fold = {'train_start': 2000, 'train_end': 2001,
                'val_start': 2004, 'val_end': 2004,
                'test_start': 2007, 'test_end': 2007}
        (prep / 'temporal_folds_task_graph.json').write_text(
            json.dumps({'folds': [fold, fold]}))
        monkeypatch.setattr(ab, 'get_absolute_output_path',
                            lambda rel: str(tmp_path / rel))
        runner = ab.BenchmarkRunner.__new__(ab.BenchmarkRunner)
        assert runner._count_fold_records('task_graph') == 8


class TestColoursAreDerived:

    def test_no_paradigm_colour_is_written_out(self):
        source = BENCHMARK.read_text()
        for hardcoded in ('"task_graph": "#', '"sql_engine": "#',
                          '"dataframe_lib": "#'):
            assert hardcoded not in source, (
                'a fourth paradigm would need a colour chosen by hand here'
            )

    def test_colours_come_from_the_property_cycle(self):
        source = BENCHMARK.read_text()
        assert "prop_cycle" in source

    def test_every_paradigm_gets_a_distinct_colour(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        names = sorted(discover_paradigms())
        colours = {n: cycle[i % len(cycle)] for i, n in enumerate(names)}
        assert len(set(colours.values())) == len(names)
