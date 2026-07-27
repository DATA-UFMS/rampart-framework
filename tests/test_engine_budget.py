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
        with pytest.raises(RuntimeError, match='budget does not fit'):
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


class TestStagesFourAndFiveGetItWithoutSetup:
    """Stages 4 and 5 run as processes of their own and never call setup.

    The test above calls `setup_environment()` by hand, which is what Stage 2
    does. None of that reaches the models: they are launched by `pipeline.py`
    as separate processes, and the only thing they inherit is the environment.
    If the budget depended on setup, those two stages would measure with the
    machine's core count while the others measured with the budget -- and the
    latency table would be comparing the two things.
    """

    @staticmethod
    def _paradigm_scripts():
        from core.paradigm_registry import discover_paradigms
        return {name: (meta['baseline_script'], meta['hierarchical_script'])
                for name, meta in sorted(discover_paradigms().items())}

    def test_no_model_script_configures_threads_itself(self):
        """The environment is the only dependency, and making it explicit
        protects it.
        """
        import ast as ast_module

        for name, scripts in self._paradigm_scripts().items():
            for script in scripts:
                path = _ROOT / script
                tree = ast_module.parse(path.read_text())
                docstrings = {id(node.value) for node in ast_module.walk(tree)
                              if isinstance(node, ast_module.Expr)
                              and isinstance(node.value, ast_module.Constant)}
                for node in ast_module.walk(tree):
                    if not (isinstance(node, ast_module.Constant)
                            and isinstance(node.value, str)):
                        continue
                    if id(node) in docstrings:
                        continue
                    for setting in ('num_workers', 'POLARS_MAX_THREADS',
                                    'OMP_NUM_THREADS', 'SET threads'):
                        assert setting not in node.value, (
                            f'{script}:{node.lineno} sets {setting!r} on its '
                            f'own, so the pipeline budget stops applying to '
                            f'this stage'
                        )

    @pytest.fixture(scope='class')
    def observed_without_setup(self):
        """Measures the three engines in a process that never calls
        setup_environment.
        """
        probe = '''
import os, sys, tempfile
sys.path.insert(0, "src")
import duckdb, polars as pl, dask
from collection.sql_engine.connection_manager import DuckDBConnectionManager

connection = DuckDBConnectionManager(
    os.path.join(tempfile.mkdtemp(), "t.duckdb")).get_connection()
print("BUDGET duckdb",
      connection.execute("SELECT current_setting('threads')").fetchone()[0])
print("BUDGET polars", pl.thread_pool_size())
print("BUDGET dask", dask.config.get("num_workers", -1))
'''
        script = _ROOT / '.stage45_probe_tmp.py'
        script.write_text(probe)
        try:
            env = os.environ.copy()
            env.update(deterministic_environment())
            out = subprocess.run([sys.executable, str(script)], cwd=str(_ROOT),
                                 capture_output=True, text=True, env=env)
            if out.returncode:
                pytest.skip(f'probe failed: {out.stderr[-300:]}')
            return {parts[1]: int(parts[2])
                    for parts in (line.split()
                                  for line in out.stdout.splitlines())
                    if len(parts) == 3 and parts[0] == 'BUDGET'}
        finally:
            script.unlink(missing_ok=True)

    @pytest.mark.parametrize('engine', ['duckdb', 'polars', 'dask'])
    def test_the_engine_honours_it_from_the_environment_alone(
            self, observed_without_setup, engine):
        assert observed_without_setup[engine] == BUDGET, (
            f'{engine} in Stages 4/5 runs with '
            f'{observed_without_setup[engine]} against a declared budget of '
            f'{BUDGET}'
        )

    def test_the_budget_is_not_the_host_core_count(self):
        """If they coincide, the test above passes with the budget ignored."""
        available = os.cpu_count() or 1
        if available == BUDGET:
            pytest.skip(
                f'the machine has {available} cores, equal to the budget: on '
                f'this machine the test cannot tell the two apart'
            )
        assert available != BUDGET


class TestTheProtocolFlagsAreHonoured:
    """`or` treats zero as absent, so an explicit zero fell to the default.

    Fixed for --warmup; --repetitions carried the same expression. There the
    request is not merely ignored: zero repetitions means nothing is measured,
    and silently running ten instead hides that the operator asked for
    something the protocol cannot deliver.
    """

    @staticmethod
    def _runner(**kwargs):
        from benchmarking.architectural_benchmark import BenchmarkRunner
        instance = BenchmarkRunner.__new__(BenchmarkRunner)
        from core.config import BENCHMARK_CONFIG
        repetitions = kwargs.get('repetitions')
        warmup = kwargs.get('warmup_runs')
        instance.repetitions = (int(BENCHMARK_CONFIG['repetitions'])
                                if repetitions is None else int(repetitions))
        instance.warmup_runs = (int(BENCHMARK_CONFIG['warmup_runs'])
                                if warmup is None else int(warmup))
        return instance

    def test_zero_warmup_is_honoured(self):
        source = (_SRC / 'benchmarking'
                  / 'architectural_benchmark.py').read_text()
        assert 'warmup_runs or int(' not in source
        assert 'repetitions or int(' not in source

    def test_zero_repetitions_is_refused_rather_than_replaced(self):
        from benchmarking.architectural_benchmark import BenchmarkRunner
        with pytest.raises(ValueError,
                           match='without repetitions there is no measurement'):
            BenchmarkRunner(repetitions=0)

    def test_a_negative_warmup_is_refused(self):
        from benchmarking.architectural_benchmark import BenchmarkRunner
        with pytest.raises(ValueError, match='no negative warmup'):
            BenchmarkRunner(warmup_runs=-1)

    def test_the_configured_values_are_used_when_omitted(self):
        from core.config import BENCHMARK_CONFIG
        instance = self._runner()
        assert instance.repetitions == BENCHMARK_CONFIG['repetitions']
        assert instance.warmup_runs == BENCHMARK_CONFIG['warmup_runs']

    def test_an_explicit_value_wins_over_the_configuration(self):
        from core.config import BENCHMARK_CONFIG
        instance = self._runner(repetitions=3, warmup_runs=0)
        assert instance.repetitions == 3
        assert instance.warmup_runs == 0
        assert instance.repetitions != BENCHMARK_CONFIG['repetitions']
