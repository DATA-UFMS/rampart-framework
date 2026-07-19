#!/usr/bin/env python3
"""A coleta é EXECUTADA, não inspecionada por texto.

Motivo de existir: a imputação foi reescrita e ficou com uma referência pendurada
a duas variáveis que a reescrita removeu do escopo. Qualquer painel com uma célula
ausente levantava NameError na primeira coluna. Os 34 testes de
test_imputation_scope.py passaram verdes porque validam a função inteiramente por
COLLECTOR.read_text() e busca de substring -- confirmam que a deleção textual
aconteceu, não que o que sobrou roda.

Nenhum dos 715 testes executava apply_conservative_imputation. Estes executam.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

TARGET = 'lower_secondary_completion_rate'


def _collector(tmp_path):
    """Instância sem __init__: ele imprime, lê rede e monta caminhos reais."""
    import collection.raw_data_collector as module

    cls = next(getattr(module, name) for name in dir(module)
               if isinstance(getattr(module, name), type)
               and hasattr(getattr(module, name), 'apply_conservative_imputation'))
    instance = cls.__new__(cls)
    instance.output_dir = str(tmp_path)
    instance.indicator_to_category = {}
    instance.target_source_column = TARGET
    return instance


def _panel(n_years=6, gap_at=2002):
    """Painel com lacunas: sem elas a imputação não é exercitada."""
    rows = []
    for entity in ('AAA', 'BBB'):
        for year in range(2000, 2000 + n_years):
            rows.append({'country_code': entity, 'year': year})
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(5)
    frame[TARGET] = rng.uniform(5.0, 25.0, len(frame))
    frame['gini_index'] = rng.uniform(30.0, 55.0, len(frame))
    frame['internet_users_percent'] = rng.uniform(10.0, 90.0, len(frame))
    # Lacunas em duas features, uma delas no meio da série.
    frame.loc[frame['year'] == gap_at, 'gini_index'] = np.nan
    frame.loc[(frame['country_code'] == 'AAA') & (frame['year'] == 2001),
              'internet_users_percent'] = np.nan
    return frame


class TestImputationExecutes:
    """Cada um destes teria falhado com o NameError."""

    def test_it_returns_a_frame(self, tmp_path):
        result = _collector(tmp_path).apply_conservative_imputation(_panel())
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_a_panel_with_gaps_does_not_raise(self, tmp_path):
        """A condição exata que quebrava: coluna com célula ausente."""
        panel = _panel()
        assert panel['gini_index'].isna().any(), 'fixture sem lacuna'
        _collector(tmp_path).apply_conservative_imputation(panel)

    def test_a_panel_without_gaps_also_runs(self, tmp_path):
        """O caminho que pula a coluna, para não passar só pelo ramo feliz."""
        panel = _panel()
        panel['gini_index'] = 40.0
        panel['internet_users_percent'] = 50.0
        _collector(tmp_path).apply_conservative_imputation(panel)

    def test_the_log_is_written_and_readable(self, tmp_path):
        _collector(tmp_path).apply_conservative_imputation(_panel())
        log = json.loads(
            (tmp_path / 'scientific_imputation_log.json').read_text())
        assert log['imputation_log'], 'log vazio'

    def test_the_log_records_a_single_mechanism(self, tmp_path):
        """Os tiers cross-seccionais saíram; o log não pode sugerir escolha."""
        _collector(tmp_path).apply_conservative_imputation(_panel())
        log = json.loads(
            (tmp_path / 'scientific_imputation_log.json').read_text())
        for column, entry in log['imputation_log'].items():
            assert entry['method_used'] == 'entity_forward_fill', column
            assert entry['geographic_count'] == 0, column
            assert entry['global_count'] == 0, column

    def test_coverage_is_written(self, tmp_path):
        _collector(tmp_path).apply_conservative_imputation(_panel())
        coverage = json.loads((tmp_path / 'target_coverage.json').read_text())
        assert coverage['target_source_column'] == TARGET
        assert coverage['rows_after'] <= coverage['rows_before']

    def test_the_target_is_not_imputed(self, tmp_path):
        """Executado, não verificado por substring."""
        panel = _panel()
        panel.loc[panel['year'] == 2003, TARGET] = np.nan
        before = panel[TARGET].notna().sum()
        result = _collector(tmp_path).apply_conservative_imputation(panel)
        assert len(result) == before, (
            'linhas sem alvo observado deveriam sair, não ser preenchidas'
        )
        assert result[TARGET].notna().all()

    def test_forward_fill_uses_only_the_entity_past(self, tmp_path):
        """Uma entidade não pode receber valor de outra."""
        panel = _panel()
        panel.loc[panel['country_code'] == 'AAA', 'gini_index'] = np.nan
        panel.loc[panel['country_code'] == 'BBB', 'gini_index'] = 99.0
        result = _collector(tmp_path).apply_conservative_imputation(panel)
        filled = result[result['country_code'] == 'AAA']['gini_index'].dropna()
        assert (filled != 99.0).all(), (
            'valor de outra entidade vazou para AAA'
        )


class TestNoDanglingReference:
    """A classe de defeito, não só a instância."""

    def test_no_name_is_used_before_assignment_in_the_imputation(self):
        """Todo nome lido dentro da função tem de estar ligado em algum lugar.

        Coleta os alvos como Name em contexto Store, o que cobre de uma vez
        desempacotamento de tupla, alvos de for, compreensões e `with ... as`;
        mais os nomes de `except ... as`, os argumentos (inclusive de lambdas
        aninhadas) e o escopo de módulo.
        """
        import ast
        import builtins

        source = (_SRC / 'collection' / 'raw_data_collector.py').read_text()
        tree = ast.parse(source)
        function = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)
                        and n.name == 'apply_conservative_imputation')

        bound = {n.id for n in ast.walk(function)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        bound |= {h.name for h in ast.walk(function)
                  if isinstance(h, ast.ExceptHandler) and h.name}
        for node in ast.walk(function):
            if isinstance(node, (ast.FunctionDef, ast.Lambda)):
                args = node.args
                bound |= {a.arg for a in
                          args.posonlyargs + args.args + args.kwonlyargs}
                for extra in (args.vararg, args.kwarg):
                    if extra:
                        bound.add(extra.arg)

        module_scope = {n.id for node in ast.walk(tree)
                        if isinstance(node, ast.Assign)
                        for n in ast.walk(node) if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Store)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_scope |= {(a.asname or a.name.split('.')[0])
                                 for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                module_scope |= {(a.asname or a.name) for a in node.names}
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                module_scope.add(node.name)

        used = {n.id for n in ast.walk(function)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        dangling = sorted(used - bound - module_scope - set(dir(builtins)))
        assert not dangling, (
            f'nomes lidos e nunca ligados em apply_conservative_imputation: '
            f'{dangling} -- é a classe de defeito que matou a coleta'
        )
