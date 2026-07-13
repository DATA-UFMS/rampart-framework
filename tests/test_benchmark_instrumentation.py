#!/usr/bin/env python3
"""
Testes da instrumentação do benchmark.

Cobrem o contrato de contagem de registros, cuja falha silenciosa mascarou por
completo a ausência de throughput nos resultados: um artefato com nome errado
era contado como zero linhas, indistinguível de um artefato legitimamente vazio.
"""

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
    """`_count_rows_parquet` precisa distinguir 'não medido' de 'zero linhas'."""

    @staticmethod
    def _counter():
        from benchmarking.architectural_benchmark import BenchmarkRunner
        # Evita __init__ (que descobre paradigmas e cria diretórios): o método
        # sob teste não depende de estado da instância.
        return BenchmarkRunner._count_rows_parquet.__get__(object(), object)

    def test_missing_artifact_returns_none(self, tmp_path):
        count = self._counter()
        assert count(str(tmp_path / 'nao_existe.parquet')) is None

    def test_unreadable_artifact_returns_none(self, tmp_path):
        corrupted = tmp_path / 'corrompido.parquet'
        corrupted.write_bytes(b'nao sou um parquet')
        count = self._counter()
        assert count(str(corrupted)) is None

    def test_valid_artifact_returns_row_count(self, tmp_path):
        pd = pytest.importorskip('pandas')
        pytest.importorskip('pyarrow')
        path = tmp_path / 'valido.parquet'
        pd.DataFrame({'year': [2000, 2001, 2002]}).to_parquet(path)
        count = self._counter()
        assert count(str(path)) == 3


class TestThroughputDerivation:
    """Throughput não deve ser inventado quando a contagem é desconhecida."""

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
    """Todo setup precisa devolver o dicionário de status ao benchmark."""

    def test_every_setup_main_returns_status_on_success_path(self):
        import ast
        from core.paradigm_registry import discover_paradigms

        root = Path(__file__).resolve().parents[1]
        for name, meta in sorted(discover_paradigms().items()):
            source = (root / meta['setup_script']).read_text()
            tree = ast.parse(source)
            main = next(
                (n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == 'main'),
                None,
            )
            assert main is not None, f"{name}: setup sem função main()"

            # Um `return` no bloco `except` não basta: o benchmark só conta
            # registros quando o caminho de SUCESSO devolve o dicionário. Por
            # isso procuramos returns no corpo do try, ignorando os handlers.
            success_returns = []
            for node in ast.walk(main):
                if not isinstance(node, ast.Try):
                    continue
                for stmt in node.body:
                    success_returns += [
                        n for n in ast.walk(stmt) if isinstance(n, ast.Return)
                    ]
            # Cobre também um main() sem try/except que retorne direto.
            if not any(isinstance(n, ast.Try) for n in ast.walk(main)):
                success_returns = [
                    n for n in ast.walk(main) if isinstance(n, ast.Return)
                ]

            assert success_returns, (
                f"{name}: main() não devolve o dicionário de status no caminho "
                f"de sucesso; o benchmark trata isso como 'não medido' e o "
                f"throughput fica ausente"
            )
