#!/usr/bin/env python3
"""Uma etapa que falha tem de chegar ao orquestrador como falha.

pipeline.py invoca cada etapa com subprocess check=True, que lê exclusivamente o
código de retorno. Quatro entrypoints imprimiam 'falha' e saíam 0: o coletor e os
três processadores. O efeito composto com um erro na coleta é que o pipeline
imprime 'Etapa 1 concluida', os processadores leem o complete_data.parquet da
execução anterior, e trinta horas produzem os números que o rerun existia para
substituir.

Dois modelos do sql_engine devolviam dicionário de erro em vez de levantar, o que
faz o benchmark registrar a repetição falha como uma latência curta -- falhar é
rápido, então a distribuição do paradigma é puxada para baixo.

Estes testes EXECUTAM os processos. Verificar por leitura de texto foi como a
classe de defeito passou.
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
        assert index >= 0, f'{relative} sem guarda de módulo'
        guard = source[index:]
        assert 'sys.exit' in guard or 'SystemExit' in guard, (
            f'{relative} termina sem status: uma falha chega ao pipeline como '
            f'sucesso, porque check=True só lê o código de retorno'
        )

    @pytest.mark.parametrize('relative', ENTRYPOINTS)
    def test_the_exit_is_conditional_on_the_outcome(self, relative):
        """sys.exit(0) incondicional propaga tão pouco quanto não ter nenhum."""
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
            f'{relative} só chama sys.exit(0): o status não depende do desfecho'
        )


class TestFailureReachesTheShell:
    """Executado de verdade, sem dado de entrada, num diretório vazio."""

    @pytest.mark.parametrize('relative', [
        'src/collection/task_graph/processor.py',
        'src/collection/sql_engine/processor.py',
        'src/collection/dataframe_lib/processor.py',
    ])
    def test_a_processor_without_input_exits_non_zero(self, relative, tmp_path):
        import os

        env = os.environ.copy()
        env['PYTHONPATH'] = str(_SRC) + os.pathsep + env.get('PYTHONPATH', '')
        # Raiz de saída vazia: não há complete_data.parquet para processar.
        env['DATASET_NAME'] = 'worldbank'
        result = subprocess.run([sys.executable, str(_ROOT / relative)],
                                cwd=str(tmp_path), env=env,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode != 0, (
            f'{relative} saiu 0 sem dado de entrada; stdout:\n'
            f'{result.stdout[-800:]}'
        )


class TestModelsRaiseInsteadOfReturningErrors:
    """Um dict de erro atravessa o benchmark como medição."""

    @pytest.mark.parametrize('relative', [
        'src/architectures_ml/sql_engine/models/hierarchical_model.py',
        'src/architectures_ml/sql_engine/models/baseline_analysis.py',
    ])
    def test_no_handler_returns_an_error_dictionary(self, relative):
        tree = ast.parse((_ROOT / relative).read_text())
        for handler in [n for n in ast.walk(tree)
                        if isinstance(n, ast.ExceptHandler)]:
            # Só os métodos que produzem o resultado do estágio. Análises
            # descritivas devolvem dicionário com 'error' de propósito, e o
            # chamador as trata como informativas.
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
                        f'{relative}:{node.lineno} devolve {sorted(keys)} de '
                        f'dentro de um except; o chamador não distingue isso de '
                        f'uma execução bem-sucedida'
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
            f'{method} aceita qualquer status: uma repetição que falhou entra '
            f'no CSV como latência curta, e falhar é rápido'
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
