#!/usr/bin/env python3
"""Numerical libraries get a fixed thread budget.

Nothing pinned the thread pools. OpenBLAS sizes its pool from the available cores
-- twelve on the development machine -- so a stage's latency depended on how many
cores it happened to get.

That is worse than irreproducible across machines. The paradigms do not contend
for cores equally: the task-graph scheduler runs workers alongside the fit, so
part of a measured difference between paradigms would be thread contention rather
than the paradigm, which is precisely the attribution the benchmark claims.

These variables are read when the numerical library loads and cannot be lowered
afterwards from inside the process, so they are exported to each subprocess rather
than set in the orchestrator.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

sys.path.insert(0, str(_ROOT))

from core.scientific_config import SCIENTIFIC_CONFIG
from pipeline import deterministic_environment

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS')


class TestThreadBudget:

    def test_every_backend_is_covered(self):
        env = deterministic_environment()
        for variable in THREAD_VARIABLES:
            assert variable in env, (
                f'{variable} unset leaves that backend sizing its own pool'
            )

    def test_all_backends_get_the_same_budget(self):
        env = deterministic_environment()
        values = {env[v] for v in THREAD_VARIABLES}
        assert len(values) == 1, f'backends disagree: {values}'

    def test_budget_comes_from_the_configuration(self):
        env = deterministic_environment()
        assert env['OMP_NUM_THREADS'] == str(SCIENTIFIC_CONFIG['blas_threads'])

    def test_budget_is_single_threaded(self):
        """Above one, thread contention re-enters the measured difference."""
        assert SCIENTIFIC_CONFIG['blas_threads'] == 1

    def test_hash_seed_is_pinned_to_the_configured_seed(self):
        env = deterministic_environment()
        assert env['PYTHONHASHSEED'] == str(SCIENTIFIC_CONFIG['random_seed'])

    def test_values_are_strings(self):
        """An int in the environment mapping raises when the subprocess spawns."""
        assert all(isinstance(v, str)
                   for v in deterministic_environment().values())


class TestEngineBudgetReachesEveryStage:
    """As etapas 4 e 5 rodam como processos separados.

    O dask.config.set da etapa de processamento não as alcança, então elas
    mediam com o número de núcleos do host enquanto as outras mediam com o
    orçamento -- e a tabela de latência comparava as duas coisas.
    """

    def test_dask_receives_the_engine_budget(self):
        env = deterministic_environment()
        assert env['DASK_NUM_WORKERS'] == str(
            SCIENTIFIC_CONFIG['engine_threads'])

    def test_dask_honours_it_in_a_child(self, tmp_path):
        import os
        import subprocess

        script = tmp_path / 'probe.py'
        script.write_text('import dask\n'
                          'print(dask.config.get("num_workers", "unset"))\n')
        env = os.environ.copy()
        env.update(deterministic_environment())
        out = subprocess.run([sys.executable, str(script)], check=True,
                             capture_output=True, text=True, env=env)
        assert out.stdout.strip() == str(SCIENTIFIC_CONFIG['engine_threads'])

    def test_it_matches_the_other_engines(self):
        env = deterministic_environment()
        assert env['DASK_NUM_WORKERS'] == env['POLARS_MAX_THREADS']


class TestEnvironmentReachesTheSubprocess:

    def test_a_child_sees_the_thread_budget(self, tmp_path):
        """Verified by spawning, since the value is read at library load."""
        script = tmp_path / 'probe.py'
        script.write_text(
            'import os\n'
            'print(os.environ.get("OMP_NUM_THREADS", "unset"))\n'
            'print(os.environ.get("PYTHONHASHSEED", "unset"))\n'
        )
        import os
        env = os.environ.copy()
        env.update(deterministic_environment())
        out = subprocess.run([sys.executable, str(script)], check=True,
                             capture_output=True, text=True, env=env)
        threads, hashseed = out.stdout.split()
        assert threads == str(SCIENTIFIC_CONFIG['blas_threads'])
        assert hashseed == str(SCIENTIFIC_CONFIG['random_seed'])

    def test_openblas_honours_the_budget(self, tmp_path):
        """The point of the exercise: the pool must actually be that size."""
        pytest.importorskip('threadpoolctl')
        script = tmp_path / 'probe.py'
        script.write_text(
            'import numpy\n'
            'from threadpoolctl import threadpool_info\n'
            'print(max(p["num_threads"] for p in threadpool_info()))\n'
        )
        import os
        env = os.environ.copy()
        env.update(deterministic_environment())
        out = subprocess.run([sys.executable, str(script)], check=True,
                             capture_output=True, text=True, env=env)
        assert int(out.stdout.strip()) == SCIENTIFIC_CONFIG['blas_threads'], (
            'the pool ignored the budget, so latency still depends on cores'
        )


class TestPipelineAppliesIt:

    def test_run_updates_the_child_environment(self):
        source = (_ROOT / 'pipeline.py').read_text()
        block = source[source.index('def run('):]
        block = block[:block.index('\ndef ')]
        assert 'deterministic_environment()' in block

    def test_no_shell_interpretation(self):
        """A repository path containing a space would break the string form."""
        source = (_ROOT / 'pipeline.py').read_text()
        assert 'shell=True' not in source

    def test_repetitions_are_not_restated(self):
        """The benchmark reads BENCHMARK_CONFIG; a flag here is a second source.

        Checked over string literals rather than the file text, so a comment
        explaining the absence does not count as a restatement.
        """
        import ast
        tree = ast.parse((_ROOT / 'pipeline.py').read_text())
        literals = {node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)}
        assert '--repetitions' not in literals
        assert '--warmup' not in literals
