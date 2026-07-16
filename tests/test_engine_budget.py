#!/usr/bin/env python3
"""Every paradigm gets the same core budget, and it is recorded.

Two different things were conflated before. The libraries beneath scikit-learn are
the *shared* component -- all three paradigms materialise to pandas before the fit
-- so their thread pool is nobody's paradigm property and pinning it to one removes
a confound at no cost (measured: RidgeCV 0.64s at one thread against 0.84s at
twelve, since trees and small linear algebra are not BLAS-bound).

The engines' own parallelism is the opposite case. A SQL engine vectorises across
threads, a DataFrame library schedules work-stealing over Arrow, a task-graph
scheduler runs workers: that *is* the paradigm. Pinning it to one would measure a
configuration nobody deploys and dissolve the premise of the comparison. Nothing
pinned it at all, so each engine sized itself from the host -- twelve cores here --
and no artifact recorded how many any engine had.

The criterion is an equal budget, declared and validated: each paradigm gets the
same cores and is free to exploit them as its design dictates. How well it does is
a finding about the paradigm.
"""

import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
for path in (str(_SRC), str(_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.scientific_config import SCIENTIFIC_CONFIG
from pipeline import _validate_core_budget, deterministic_environment

BUDGET = SCIENTIFIC_CONFIG['engine_threads']
BLAS = SCIENTIFIC_CONFIG['blas_threads']


class TestBudgetIsDeclared:

    def test_engine_budget_is_an_explicit_integer(self):
        """Derived from the host, it would not be reproducible elsewhere."""
        assert isinstance(BUDGET, int) and BUDGET >= 1

    def test_engines_get_more_than_one_core(self):
        """A scheduler with a single worker is not a scheduler."""
        assert BUDGET > 1, (
            'pinning engine parallelism to one measures a configuration nobody '
            'deploys and removes the property under study'
        )

    def test_shared_numerical_component_is_single_threaded(self):
        assert BLAS == 1

    def test_polars_budget_is_exported(self):
        """Polars sizes its pool at import, so only the environment reaches it."""
        assert deterministic_environment()['POLARS_MAX_THREADS'] == str(BUDGET)

    def test_blas_and_engine_budgets_are_distinct_settings(self):
        env = deterministic_environment()
        assert env['OMP_NUM_THREADS'] != env['POLARS_MAX_THREADS'], (
            'the shared component and the paradigm component are being pinned '
            'to the same value, which conflates them'
        )


class TestBudgetIsValidated:

    def test_a_fitting_budget_passes(self):
        _validate_core_budget()

    def test_oversubscription_is_refused(self, monkeypatch):
        """Silently oversubscribing makes latency reflect contention."""
        monkeypatch.setitem(SCIENTIFIC_CONFIG, 'engine_threads',
                            multiprocessing.cpu_count() + 4)
        with pytest.raises(RuntimeError, match='não cabe'):
            _validate_core_budget()

    def test_the_error_names_both_budgets(self, monkeypatch):
        monkeypatch.setitem(SCIENTIFIC_CONFIG, 'engine_threads', 10_000)
        with pytest.raises(RuntimeError) as exc:
            _validate_core_budget()
        assert 'engine_threads' in str(exc.value)
        assert 'blas_threads' in str(exc.value)

    def test_pipeline_validates_before_running(self):
        source = (_ROOT / 'pipeline.py').read_text()
        main = source[source.index('def main('):]
        validation = main.index('_validate_core_budget()')
        first_stage = main.index('run([py')
        assert validation < first_stage, (
            'the budget must be checked before any stage executes'
        )


class TestEveryEngineHonoursIt:
    """Checked by spawning, since two of the three read the value at load."""

    @pytest.fixture(scope='class')
    def observed(self):
        # Each engine is asked through its own entry point. Setting the Dask
        # config here and reading it back would prove nothing about whether the
        # paradigm sets it.
        probe = '''
import os, sys, tempfile
sys.path.insert(0, "src")
import duckdb, polars as pl, dask
from core.scientific_config import SCIENTIFIC_CONFIG
from collection.sql_engine.connection_manager import DuckDBConnectionManager
from architectures_ml.task_graph.setup import TaskGraphArchitectureML

conn = DuckDBConnectionManager(
    os.path.join(tempfile.mkdtemp(), "t.duckdb")).get_connection()
print("BUDGET duckdb", conn.execute("SELECT current_setting('threads')").fetchone()[0])
print("BUDGET polars", pl.thread_pool_size())

dask.config.set({"num_workers": 999})            # sentinel the setup must replace
instance = TaskGraphArchitectureML.__new__(TaskGraphArchitectureML)
instance.config = SCIENTIFIC_CONFIG
instance.setup_environment()
print("BUDGET dask", dask.config.get("num_workers"))
'''
        script = _ROOT / '.engine_probe_tmp.py'
        script.write_text(probe)
        try:
            env = os.environ.copy()
            env.update(deterministic_environment())
            out = subprocess.run([sys.executable, str(script)], cwd=str(_ROOT),
                                 capture_output=True, text=True, env=env)
            if out.returncode:
                pytest.skip(f'probe failed: {out.stderr[-300:]}')
            # setup_environment() prints diagnostics; only the marked lines
            # are the measurement.
            return {parts[1]: int(parts[2])
                    for parts in (line.split()
                                  for line in out.stdout.splitlines())
                    if len(parts) == 3 and parts[0] == 'BUDGET'}
        finally:
            script.unlink(missing_ok=True)

    @pytest.mark.parametrize('engine', ['duckdb', 'polars', 'dask'])
    def test_engine_uses_the_budget(self, observed, engine):
        assert observed[engine] == BUDGET, (
            f'{engine} runs with {observed[engine]} threads against a declared '
            f'budget of {BUDGET}, so its latency is not comparable to the others'
        )

    def test_all_three_agree(self, observed):
        assert len(set(observed.values())) == 1, (
            f'unequal budgets make the comparison measure the budget: {observed}'
        )


class TestBudgetReachesTheSnapshot:

    def test_both_settings_are_in_the_serialised_config(self):
        import json
        payload = json.dumps({'scientific_config': SCIENTIFIC_CONFIG},
                             default=str)
        assert 'engine_threads' in payload
        assert 'blas_threads' in payload
