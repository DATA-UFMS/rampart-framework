#!/usr/bin/env python3
"""Benchmark instrumentation contracts: record counting and throughput."""

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(scope='module')
def phase_result_cls():
    from benchmarking.architectural_benchmark import PhaseResult
    return PhaseResult


class TestRecordCountContract:
    """An unavailable count must be None, never 0."""

    @staticmethod
    def _counter():
        from benchmarking.architectural_benchmark import BenchmarkRunner
        # Bypasses __init__, which discovers paradigms and creates directories;
        # the method under test uses no instance state.
        return BenchmarkRunner._count_rows_parquet.__get__(object(), object)

    def test_missing_artifact_returns_none(self, tmp_path):
        count = self._counter()
        assert count(str(tmp_path / 'absent.parquet')) is None

    def test_unreadable_artifact_returns_none(self, tmp_path):
        corrupted = tmp_path / 'corrupted.parquet'
        corrupted.write_bytes(b'not a parquet file')
        count = self._counter()
        assert count(str(corrupted)) is None

    def test_valid_artifact_returns_row_count(self, tmp_path):
        pd = pytest.importorskip('pandas')
        pytest.importorskip('pyarrow')
        path = tmp_path / 'valid.parquet'
        pd.DataFrame({'year': [2000, 2001, 2002]}).to_parquet(path)
        count = self._counter()
        assert count(str(path)) == 3


class TestThroughputDerivation:
    """Throughput is reported only when the record count is known."""

    def test_unknown_records_yields_no_throughput(self, phase_result_cls):
        r = phase_result_cls(run_id=1, phase='setup', architecture='sql_engine',
                             step='s', duration_ns=1_000_000_000, records=None)
        assert r.throughput_rps is None

    def test_zero_records_yields_no_throughput(self, phase_result_cls):
        r = phase_result_cls(run_id=1, phase='setup', architecture='sql_engine',
                             step='s', duration_ns=1_000_000_000, records=0)
        assert r.throughput_rps is None

    def test_known_records_yield_throughput(self, phase_result_cls):
        r = phase_result_cls(run_id=1, phase='setup', architecture='task_graph',
                             step='s', duration_ns=2_000_000_000, records=1000)
        assert r.throughput_rps == pytest.approx(500.0)


class TestSetupMainContract:
    """Each setup must hand its status dictionary back to the benchmark."""

    def test_every_setup_main_returns_on_success_path(self):
        import ast
        from core.paradigm_registry import discover_paradigms

        root = Path(__file__).resolve().parents[1]
        for name, meta in sorted(discover_paradigms().items()):
            tree = ast.parse((root / meta['setup_script']).read_text())
            main = next(
                (n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == 'main'),
                None,
            )
            assert main is not None, f"{name}: setup has no main()"

            # A return inside an except handler does not satisfy the contract:
            # the benchmark counts records only on the success path.
            tries = [n for n in ast.walk(main) if isinstance(n, ast.Try)]
            if tries:
                returns = [
                    n for t in tries for stmt in t.body
                    for n in ast.walk(stmt) if isinstance(n, ast.Return)
                ]
            else:
                returns = [n for n in ast.walk(main) if isinstance(n, ast.Return)]

            assert returns, (
                f"{name}: main() returns nothing on success, so the benchmark "
                f"treats the run as unmeasured"
            )
