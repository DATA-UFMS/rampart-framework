#!/usr/bin/env python3
"""Artifact paths are segregated by dataset and resolved in one place.

Two datasets wrote to the same paths. Running INEP after World Bank overwrote the
first set under identical names, and an interrupted run left artifacts from two
datasets side by side with nothing recording it. The published results were split
into per-dataset directories by hand, not by the code.

Resolution was also inconsistent: sixteen modules resolved against the project
root while six used literals relative to the working directory, so where an
artifact landed depended on where the process was started.
"""

import ast
import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.config import (DEFAULT_DATASET, get_absolute_output_path,
                         get_dataset_name, get_outputs_root)

_MODULES = sorted(p for p in _SRC.rglob('*.py') if p.name != '__init__.py')

RESOLVER = 'get_absolute_output_path'

# The resolver itself matches on the 'outputs/' prefix, so it is the one module
# that legitimately mentions the literal.
RESOLVER_MODULE = _SRC / 'core' / 'config.py'


@pytest.fixture
def dataset(monkeypatch):
    def _set(name):
        monkeypatch.setenv('DATASET_NAME', name)
        return name
    return _set


class TestSegregation:

    def test_root_carries_the_dataset(self, dataset):
        dataset('inep_censo')
        assert get_outputs_root().endswith(os.path.join('outputs', 'inep_censo'))

    def test_two_datasets_never_share_a_path(self, dataset):
        relative = 'outputs/statistics/equivalence_estimation.json'
        dataset('worldbank')
        first = get_absolute_output_path(relative)
        dataset('inep_censo')
        second = get_absolute_output_path(relative)
        assert first != second, 'a second dataset would overwrite the first'

    def test_unset_environment_falls_back_to_the_declared_default(self,
                                                                 monkeypatch):
        monkeypatch.delenv('DATASET_NAME', raising=False)
        assert get_dataset_name() == DEFAULT_DATASET

    @pytest.mark.parametrize('relative', [
        'outputs/statistics',
        'outputs/benchmarks/architectural_benchmark_results.csv',
        'statistics',
        'outputs/ml_pipeline/architectures/sql_engine/prep',
        # The bare form does not start with 'outputs/', so it escaped the
        # prefix stripping and produced outputs/<dataset>/outputs.
        'outputs',
        'outputs/',
    ])
    def test_every_form_lands_under_the_dataset_root(self, relative, dataset):
        name = dataset('inep_censo')
        resolved = get_absolute_output_path(relative)
        assert get_outputs_root() in resolved
        assert f'outputs{os.sep}{name}' in resolved
        # 'outputs' must not appear twice in the tail.
        tail = resolved.split(f'outputs{os.sep}{name}{os.sep}')[-1]
        assert not tail.startswith('outputs')

    def test_the_bare_form_is_the_dataset_root(self, dataset):
        """It was outputs/<dataset>/outputs, one level below the consumers."""
        dataset('worldbank')
        assert get_absolute_output_path('outputs') == get_outputs_root()
        assert get_absolute_output_path('') == get_outputs_root()

    def test_the_snapshot_lands_where_its_readers_look(self, dataset):
        """The table generator warned and exited 0 without generating anything."""
        dataset('worldbank')
        written = Path(get_absolute_output_path('outputs')) / \
            'scientific_config_snapshot.json'
        read_by_tables = Path(get_outputs_root()) / \
            'scientific_config_snapshot.json'
        assert written == read_by_tables

    def test_absolute_paths_are_returned(self, dataset):
        dataset('worldbank')
        assert os.path.isabs(get_absolute_output_path('outputs/statistics'))


class TestSingleResolver:
    """A literal path is resolved against the working directory, not the root."""

    @pytest.mark.parametrize('path', _MODULES,
                             ids=lambda p: str(p.relative_to(_SRC)))
    def test_no_literal_output_path_outside_the_resolver(self, path):
        if path == RESOLVER_MODULE:
            pytest.skip('defines the resolver; matches on the prefix itself')
        source = path.read_text()
        tree = ast.parse(source)

        # Strings inside a call to the resolver are fine; so are docstrings.
        allowed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                    getattr(node.func, 'id', None) == RESOLVER
                    or getattr(node.func, 'attr', None) == RESOLVER):
                for arg in ast.walk(node):
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        allowed.add(id(arg))
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                allowed.add(id(node.value))

        offenders = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            if id(node) in allowed:
                continue
            if re.search(r'(^|[^\w])outputs[/\\]', node.value):
                offenders.append((node.lineno, node.value[:70]))

        assert not offenders, (
            f"{path.relative_to(_ROOT)} builds an output path outside "
            f"{RESOLVER}: {offenders}. A literal is resolved against the "
            f"working directory and skips the per-dataset segregation."
        )


class TestProducersAndConsumersAgree:
    """The benchmark CSV is written by one module and read by four."""

    ARTIFACT = 'architectural_benchmark_results.csv'
    READERS = [
        'statistical_validation.significance_tests',
        'statistical_validation.effect_analysis',
        'benchmarking.derive_latency_percentiles',
        'benchmarking.derive_throughput_percentiles',
    ]

    def _resolved_paths(self, module_name):
        module = importlib.import_module(module_name)
        importlib.reload(module)
        found = []
        for attr in dir(module):
            value = getattr(module, attr)
            text = str(value) if not isinstance(value, str) else value
            if attr == '__doc__' or not text:
                continue
            if self.ARTIFACT in text and os.sep in text:
                found.append(text)
        return found

    @pytest.mark.parametrize('module_name', READERS)
    def test_reader_resolves_under_the_dataset_root(self, module_name,
                                                    monkeypatch):
        monkeypatch.setenv('DATASET_NAME', 'inep_censo')
        paths = self._resolved_paths(module_name)
        if not paths:
            pytest.skip(f'{module_name} resolves the CSV at call time')
        for path in paths:
            assert os.path.join('outputs', 'inep_censo') in path, (
                f'{module_name} reads {path}, outside the dataset root'
            )

    def test_readers_agree_with_each_other(self, monkeypatch):
        monkeypatch.setenv('DATASET_NAME', 'worldbank')
        resolved = set()
        for module_name in self.READERS:
            resolved.update(self._resolved_paths(module_name))
        assert len(resolved) <= 1, (
            f'readers disagree on where the benchmark CSV lives: {resolved}'
        )


class TestSnapshotRecordsTheDataset:
    """Checked by writing one, not by looking for a string in a module.

    The snapshot moved out of the orchestrator so the benchmark could write the
    same record; a test tied to the orchestrator's source would pass or fail for
    reasons unrelated to what the snapshot contains.
    """

    def test_the_snapshot_names_the_dataset(self, tmp_path, dataset):
        from core.config import write_environment_snapshot
        name = dataset('inep_censo')
        payload = json.loads(
            Path(write_environment_snapshot(str(tmp_path))).read_text())
        assert payload['dataset'] == name

    def test_the_snapshot_carries_the_core_budget(self, tmp_path, dataset):
        """A latency without the budget that produced it is not comparable."""
        from core.config import write_environment_snapshot
        dataset('worldbank')
        payload = json.loads(
            Path(write_environment_snapshot(str(tmp_path))).read_text())
        for key in ('engine_threads', 'blas_threads'):
            assert key in payload['scientific_config']

    def test_the_snapshot_records_provenance(self, tmp_path, dataset):
        from core.config import write_environment_snapshot
        dataset('worldbank')
        payload = json.loads(
            Path(write_environment_snapshot(str(tmp_path))).read_text())
        for key in ('git_commit', 'processor', 'platform', 'python',
                    'installed_packages', 'requirements_lock_sha256'):
            assert key in payload, key

    def test_callers_may_add_their_own_fields(self, tmp_path, dataset):
        from core.config import write_environment_snapshot
        dataset('worldbank')
        payload = json.loads(Path(write_environment_snapshot(
            str(tmp_path), extra={'measured_phases': ['baseline']})).read_text())
        assert payload['measured_phases'] == ['baseline']

    def test_both_writers_use_the_shared_implementation(self):
        """Two copies would diverge in the file that says how the run was made."""
        for module in (_ROOT / 'pipeline.py',
                       _SRC / 'benchmarking' / 'architectural_benchmark.py'):
            assert 'write_environment_snapshot(' in module.read_text(), module.name
