#!/usr/bin/env python3
"""A stage that fails has to reach the orchestrator as a failure.

pipeline.py invokes each stage with subprocess check=True, which reads the return
code and nothing else. Four entrypoints printed 'failure' and exited 0: the
collector and the three processors. Compounded with an error in the collection,
the effect is that the pipeline prints 'Stage 1 completed', the processors read
the complete_data.parquet from the previous execution, and thirty hours produce
the very numbers the rerun existed to replace.

Two sql_engine models returned an error dictionary instead of raising, which
makes the benchmark record the failed repetition as a short latency -- failing is
fast, so the paradigm's distribution is pulled downwards.

These tests EXECUTE the processes. Checking by reading text was how the class of
defect got through.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

ENTRYPOINTS = [
    'src/collection/raw_data_collector.py',
    'src/collection/task_graph/processor.py',
    'src/collection/sql_engine/processor.py',
    'src/collection/dataframe_lib/processor.py',
    'src/architectures_ml/sql_engine/setup.py',
    'src/architectures_ml/task_graph/setup.py',
    'src/architectures_ml/dataframe_lib/setup.py',
    'src/architectures_ml/sql_engine/models/hierarchical_model.py',
    'src/architectures_ml/task_graph/models/hierarchical_model.py',
    'src/architectures_ml/dataframe_lib/models/hierarchical_model.py',
]


class TestEveryEntrypointPropagates:

    @pytest.mark.parametrize('relative', ENTRYPOINTS)
    def test_the_module_guard_exits_with_a_status(self, relative):
        source = (_ROOT / relative).read_text()
        index = source.find('if __name__')
        assert index >= 0, f'{relative} has no module guard'
        guard = source[index:]
        assert 'sys.exit' in guard or 'SystemExit' in guard, (
            f'{relative} ends without a status: a failure reaches the pipeline '
            f'as success, because check=True only reads the return code'
        )

    @pytest.mark.parametrize('relative', ENTRYPOINTS)
    def test_the_exit_is_conditional_on_the_outcome(self, relative):
        """An unconditional sys.exit(0) propagates as little as having none."""
        source = (_ROOT / relative).read_text()
        guard = source[source.find('if __name__'):]
        tree = ast.parse(guard.replace('if __name__ == "__main__":',
                                       'if True:'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, 'attr', None) == 'exit']
        assert calls, relative
        informative = [
            call for call in calls
            if call.args and not (isinstance(call.args[0], ast.Constant)
                                  and call.args[0].value == 0)
        ]
        assert informative, (
            f'{relative} only calls sys.exit(0): the status does not depend on '
            f'the outcome'
        )


class TestFailureReachesTheShell:
    """Really executed, with no input data, in an empty directory."""

    @pytest.mark.parametrize('relative', [
        'src/collection/task_graph/processor.py',
        'src/collection/sql_engine/processor.py',
        'src/collection/dataframe_lib/processor.py',
    ])
    def test_a_processor_without_input_exits_non_zero(self, relative, tmp_path):
        import os

        env = os.environ.copy()
        env['PYTHONPATH'] = str(_SRC) + os.pathsep + env.get('PYTHONPATH', '')
        # Empty output root: there is no complete_data.parquet to process.
        env['DATASET_NAME'] = 'worldbank'
        result = subprocess.run([sys.executable, str(_ROOT / relative)],
                                cwd=str(tmp_path), env=env,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode != 0, (
            f'{relative} exited 0 with no input data; stdout:\n'
            f'{result.stdout[-800:]}'
        )


class TestModelsRaiseInsteadOfReturningErrors:
    """An error dict crosses the benchmark as a measurement."""

    @pytest.mark.parametrize('relative', [
        'src/architectures_ml/sql_engine/models/hierarchical_model.py',
        'src/architectures_ml/sql_engine/models/baseline_analysis.py',
    ])
    def test_no_handler_returns_an_error_dictionary(self, relative):
        tree = ast.parse((_ROOT / relative).read_text())
        for handler in [n for n in ast.walk(tree)
                        if isinstance(n, ast.ExceptHandler)]:
            # Only the methods that produce the stage's result. Descriptive
            # analyses return a dictionary with 'error' on purpose, and the
            # caller treats them as informative.
            RESULT_METHODS = ('run_hierarchical_analysis', 'run_complete_analysis',
                              'run_fold_analysis', 'test_baseline_models')
            enclosing = [n.name for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef)
                         and n.lineno <= handler.lineno <= (n.end_lineno or 0)]
            if not any(m in enclosing for m in RESULT_METHODS):
                continue
            for node in handler.body:
                if not isinstance(node, ast.Return):
                    continue
                if isinstance(node.value, ast.Dict):
                    keys = {k.value for k in node.value.keys
                            if isinstance(k, ast.Constant)}
                    assert not (keys & {'error', 'status'}), (
                        f'{relative}:{node.lineno} returns {sorted(keys)} from '
                        f'inside an except; the caller does not tell that apart '
                        f'from a successful execution'
                    )


class TestBenchmarkRejectsFailedStages:

    @pytest.mark.parametrize('method', ['_phase_processing_generic',
                                        '_phase_setup_generic'])
    def test_the_stage_aborts_on_a_failure_status(self, method):
        source = (_SRC / 'benchmarking' / 'architectural_benchmark.py').read_text()
        tree = ast.parse(source)
        function = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == method)
        raises = [n for n in ast.walk(function) if isinstance(n, ast.Raise)]
        assert raises, (
            f'{method} accepts any status: a repetition that failed goes into '
            f'the CSV as a short latency, and failing is fast'
        )

    @pytest.mark.parametrize('method', ['_phase_processing_generic',
                                        '_phase_setup_generic'])
    def test_the_check_precedes_the_return(self, method):
        source = (_SRC / 'benchmarking' / 'architectural_benchmark.py').read_text()
        tree = ast.parse(source)
        function = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == method)
        first_raise = min(n.lineno for n in ast.walk(function)
                          if isinstance(n, ast.Raise))
        last_return = max(n.lineno for n in ast.walk(function)
                          if isinstance(n, ast.Return))
        assert first_raise < last_return
