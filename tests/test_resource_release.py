#!/usr/bin/env python3
"""Nothing a paradigm opens survives its repetition.

The benchmark re-executes each phase `warmup + n` times in one process. A
resource that outlives a repetition is measured by the next one, and only for
the paradigm that leaked it.

The sql_engine setup opened a DuckDB connection and never closed it -- twelve
by the end of a run, each with its own buffer pool -- while the baseline and
hierarchical phases did close theirs. So its later repetitions measured under
conditions the other two never met, and the asymmetry lands directly in the
latency the paper reports.

Release is a contract on the base class rather than three separate habits.
Polars holds nothing and the collections Dask persists are locals that fall
with scope, so their implementations are empty on purpose: what matters is
that the release is symmetric, not that each paradigm invents its own.
"""

import ast
import contextlib
import io
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.base_architecture import BaseArchitectureML
from core.paradigm_registry import discover_paradigms

PARADIGMS = sorted(discover_paradigms())


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class TestTheContractExists:

    def test_the_base_class_declares_it(self):
        assert hasattr(BaseArchitectureML, 'release_resources')

    def test_the_default_is_documented_as_empty(self):
        import inspect
        source = inspect.getsource(BaseArchitectureML.release_resources)
        assert 'default is empty, and deliberately so' in source

    def test_it_is_safe_to_call_twice(self):
        """A finally clause can run after an earlier release."""
        from architectures_ml.sql_engine.setup import SqlEngineArchitectureML
        architecture = _quiet(SqlEngineArchitectureML)
        architecture.release_resources()
        architecture.release_resources()

    def test_it_is_safe_before_anything_is_opened(self):
        from architectures_ml.sql_engine.setup import SqlEngineArchitectureML
        architecture = _quiet(SqlEngineArchitectureML)
        assert architecture.conn_manager is None
        architecture.release_resources()


class TestEveryMainReleases:
    """Symmetry is the point: one paradigm releasing is worse than none."""

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_the_main_has_a_finally_that_releases(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'setup.py').read_text()
        tree = ast.parse(source)
        main = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        handlers = [node for node in ast.walk(main)
                    if isinstance(node, ast.Try) and node.finalbody]
        assert handlers, f'{paradigm}: main has no finally clause'
        released = any('release_resources' in ast.unparse(statement)
                       for handler in handlers
                       for statement in handler.finalbody)
        assert released, f'{paradigm}: the finally does not release'

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_the_instance_is_declared_before_the_try(self, paradigm):
        """Otherwise the finally raises NameError when construction fails."""
        source = (_SRC / 'architectures_ml' / paradigm / 'setup.py').read_text()
        tree = ast.parse(source)
        main = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        body = ast.unparse(main)
        assert body.index('setup = None') < body.index('release_resources')


class TestTheConnectionIsActuallyClosed:

    @staticmethod
    def _counting_manager(monkeypatch):
        from collection.sql_engine import connection_manager as module
        calls = []
        original = module.DuckDBConnectionManager.close_connection

        def counted(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(module.DuckDBConnectionManager,
                            'close_connection', counted)
        return calls

    def test_release_closes_an_open_connection(self, tmp_path, monkeypatch):
        from architectures_ml.sql_engine.setup import SqlEngineArchitectureML
        from collection.sql_engine.connection_manager import (
            DuckDBConnectionManager)

        calls = self._counting_manager(monkeypatch)
        architecture = _quiet(SqlEngineArchitectureML)
        architecture.conn_manager = DuckDBConnectionManager(
            str(tmp_path / 'probe.duckdb'))
        architecture.conn_manager.get_connection()

        architecture.release_resources()
        assert calls, 'the connection was not closed'
        assert architecture.conn_manager is None

    def test_the_failure_path_releases_too(self, tmp_path, monkeypatch):
        """A run that died mid-way left the connection open just the same."""
        import architectures_ml.sql_engine.setup as module
        from collection.sql_engine.connection_manager import (
            DuckDBConnectionManager)

        released = []

        class Exploding(module.SqlEngineArchitectureML):
            def run_setup(self):
                self.conn_manager = DuckDBConnectionManager(
                    str(tmp_path / 'probe.duckdb'))
                self.conn_manager.get_connection()
                raise RuntimeError('mid-run failure')

            def release_resources(self):
                released.append(1)
                super().release_resources()

        monkeypatch.setattr(module, 'SqlEngineArchitectureML', Exploding)
        result = _quiet(module.main)
        assert result['status'] == 'failed'
        assert released, 'the finally did not run on the failure path'

    def test_a_second_run_starts_without_a_connection(self, tmp_path,
                                                      monkeypatch):
        """The leak's actual shape: twelve repetitions, twelve buffer pools."""
        import architectures_ml.sql_engine.setup as module
        from collection.sql_engine.connection_manager import (
            DuckDBConnectionManager)

        seen = []

        class Recording(module.SqlEngineArchitectureML):
            def run_setup(self):
                seen.append(self.conn_manager)
                self.conn_manager = DuckDBConnectionManager(
                    str(tmp_path / 'probe.duckdb'))
                self.conn_manager.get_connection()
                return {'status': 'success'}

        monkeypatch.setattr(module, 'SqlEngineArchitectureML', Recording)
        _quiet(module.main)
        _quiet(module.main)
        assert seen == [None, None], (
            'a repetition started with a connection the previous one left open'
        )
