#!/usr/bin/env python3
"""The Ridge model is one implementation, and the extraction is provable.

It was three copies of 121 lines. The extraction is not a bet: compared by AST
with print calls and paradigm-name literals normalised, the three bodies are
identical, and none of them reads a single attribute of self -- they were pure
functions written as methods.

Structural equality is not sufficient evidence for code that produces the
predictions, so the equality is also empirical. Before the extraction the three
paradigms returned bitwise-identical predictions on the same input across all
three shrinkage values; this file asserts that property still holds, and that the
three paradigms agree with each other, which is the premise the Delta=0 claim
rests on.

Deliberately not extracted, with the measurement behind the decision:

  run_fold_analysis, run_hierarchical_analysis, _prepare_data,
  test_baseline_models, analyze_target_distribution, analyze_predictability,
  save_results, run_complete_analysis   -- three distinct ASTs each
  random_forest_hierarchical            -- two variants (sql_engine differs)

Those differ in how each engine materialises a fold, which is what the paradigms
are. Collapsing them means choosing one engine's idiom for all three, which is a
rewrite of the code that produces the numbers, not an extraction.
"""

import ast
import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PARADIGMS = ('task_graph', 'sql_engine', 'dataframe_lib')
SHARED = _SRC / 'core' / 'models' / 'hierarchical.py'


def _panel(n=90, seed=17):
    rng = np.random.default_rng(seed)
    entities = pd.Series(['BRA', 'ARG', 'CHL', 'MEX', 'COL', 'PER'] * (n // 6))
    X = pd.DataFrame({'gini': rng.normal(size=n),
                      'internet': rng.normal(size=n),
                      'gdp': rng.normal(size=n)})
    y = pd.Series(X['gini'] * 0.5 - X['internet'] * 0.2 + rng.normal(0, 0.3, n))
    return X, y, entities


def _model_class(paradigm):
    module = importlib.import_module(
        f'architectures_ml.{paradigm}.models.hierarchical_model')
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, 'simple_hierarchical_model'):
            return obj
    pytest.skip(f'no hierarchical class in {paradigm}')


def _run(paradigm, shrinkage):
    cls = _model_class(paradigm)
    instance = cls.__new__(cls)
    X, y, entities = _panel()
    return cls.simple_hierarchical_model(instance, X, y, X, y, entities,
                                        entities,
                                        residual_shrinkage=shrinkage)


def _digest(result):
    return hashlib.sha256(
        np.asarray(result['predictions'], dtype=float).tobytes()).hexdigest()


class TestOneImplementation:

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_the_paradigm_delegates(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        tree = ast.parse(source)
        method = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef)
                      and n.name == 'simple_hierarchical_model')
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                 and 'shared' in getattr(n.func, 'id', '')]
        assert calls, f'{paradigm} still carries its own copy'

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_no_paradigm_reimplements_the_body(self, paradigm):
        """A delegation is a handful of lines; a copy is over a hundred."""
        source = (_SRC / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        tree = ast.parse(source)
        method = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef)
                      and n.name == 'simple_hierarchical_model')
        assert method.end_lineno - method.lineno < 20, (
            f'{paradigm}: {method.end_lineno - method.lineno} lines, which is a '
            f'body rather than a delegation'
        )

    def test_the_shared_module_reads_no_instance_state(self):
        """It was extractable precisely because it is a pure function."""
        tree = ast.parse(SHARED.read_text())
        for function in [n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef)]:
            selves = [n for n in ast.walk(function) if isinstance(n, ast.Name)
                      and n.id == 'self']
            assert not selves, f'{function.name} refers to self'

    def test_the_architecture_is_a_parameter(self):
        """The only thing that varied between the copies."""
        import inspect
        from core.models.hierarchical import simple_hierarchical_model
        parameters = inspect.signature(simple_hierarchical_model).parameters
        assert 'architecture' in parameters
        assert parameters['architecture'].kind == \
            inspect.Parameter.KEYWORD_ONLY


class TestBehaviourIsPreserved:
    """The reference hashes were captured before the extraction."""

    # One digest captured from the three paradigms before the copies were
    # removed. Anchoring one value is enough: agreement between paradigms is
    # asserted separately, so the two together rule out drifting in step.
    ANCHOR_SHRINKAGE = 0.6
    ANCHOR_DIGEST_PREFIX = 'b37dce5e9e2b0be9e69c'

    @pytest.mark.parametrize('shrinkage', [0.6, 0.8, 1.0])
    def test_the_three_paradigms_agree_bitwise(self, shrinkage):
        """The premise of the equivalence claim, at the level of one method."""
        digests = {paradigm: _digest(_run(paradigm, shrinkage))
                   for paradigm in PARADIGMS}
        assert len(set(digests.values())) == 1, digests

    def test_the_captured_digest_still_holds(self):
        """One anchored value, so agreement cannot drift together."""
        digest = _digest(_run('task_graph', self.ANCHOR_SHRINKAGE))
        assert digest.startswith(self.ANCHOR_DIGEST_PREFIX), (
            f'predictions changed: {digest[:20]} != {self.ANCHOR_DIGEST_PREFIX}'
        )

    @pytest.mark.parametrize('shrinkage', [0.6, 0.8, 1.0])
    def test_metrics_agree_across_paradigms(self, shrinkage):
        results = [_run(paradigm, shrinkage) for paradigm in PARADIGMS]
        for metric in ('r2', 'rmse', 'mae'):
            values = {round(r[metric], 15) for r in results}
            assert len(values) == 1, (metric, values)

    @pytest.mark.parametrize('shrinkage', [0.6, 0.8, 1.0])
    def test_entity_effects_agree_across_paradigms(self, shrinkage):
        effects = [tuple(sorted(_run(p, shrinkage)['country_effects'].items()))
                   for p in PARADIGMS]
        assert len(set(effects)) == 1

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_the_result_is_labelled_with_its_paradigm(self, paradigm):
        assert _run(paradigm, 0.8)['architecture'] == paradigm

    def test_shrinkage_still_changes_the_result(self):
        """Otherwise the agreement above would be vacuous."""
        digests = {_digest(_run('task_graph', s)) for s in (0.6, 0.8, 1.0)}
        assert len(digests) == 3, 'the shrinkage parameter stopped mattering'


class TestWhatWasNotExtracted:
    """The decision is recorded, with the measurement that produced it."""

    @pytest.mark.parametrize('method', [
        'run_fold_analysis', 'run_hierarchical_analysis', '_prepare_data',
    ])
    def test_engine_specific_methods_remain_per_paradigm(self, method):
        for paradigm in PARADIGMS:
            source = (_SRC / 'architectures_ml' / paradigm / 'models'
                      / 'hierarchical_model.py').read_text()
            tree = ast.parse(source)
            assert any(isinstance(n, ast.FunctionDef) and n.name == method
                       for n in ast.walk(tree)), (
                f'{paradigm} lost {method}, which materialises a fold the way '
                f'that engine does'
            )

    def test_the_shared_module_holds_only_the_verified_functions(self):
        tree = ast.parse(SHARED.read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)}
        assert names == {'simple_hierarchical_model',
                         'write_prediction_artifact',
                         'write_baseline_predictions',
                         'write_imputation_report'}, names


class TestInnerCrossValidationIsDeliberate:
    """A partição da seleção de alpha, explícita em vez de acidental.

    cv=<int> faz o RidgeCV usar KFold sem shuffle. Como os resíduos são
    concatenados por entidade, os blocos contíguos eram blocos de entidade: a
    seleção de alpha fazia leave-some-entities-out sem que ninguém a tivesse
    escolhido, e mudaria em silêncio se a ordem das linhas mudasse.

    Não é leakage em nenhuma das duas formas -- todos os resíduos vêm da janela
    de treino. O que a mudança compra é determinismo, e é isso que este teste
    fixa: comportamento idêntico sob permutação das linhas. Um teste de saída
    não distinguiria as duas versões, porque a partição resultante é a mesma.
    """

    @staticmethod
    def _alpha(X, y, entities):
        from core.models.hierarchical import simple_hierarchical_model
        result = simple_hierarchical_model(X, y, X, y, entities, entities,
                                           architecture='probe')
        return result['regularization_details']['ridgecv_alpha']

    def test_alpha_does_not_depend_on_row_order(self):
        X, y, entities = _panel()
        original = self._alpha(X, y, entities)
        order = np.random.default_rng(11).permutation(len(X))
        permuted = self._alpha(X.iloc[order].reset_index(drop=True),
                               y.iloc[order].reset_index(drop=True),
                               entities.iloc[order].reset_index(drop=True))
        assert original == permuted, (
            f'alpha mudou com a ordem das linhas: {original} != {permuted}, '
            f'o que significa que a partição da CV interna é implícita'
        )

    def test_the_splitter_is_declared_not_an_integer(self):
        """cv=<int> delega a escolha da partição ao sklearn."""
        source = SHARED.read_text()
        assert 'GroupKFold' in source
        tree = ast.parse(source)
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and \
                    getattr(call.func, 'id', None) == 'RidgeCV':
                for keyword in call.keywords:
                    if keyword.arg == 'cv':
                        assert not isinstance(keyword.value, ast.Constant), (
                            'cv passado como literal reintroduz a partição '
                            'implícita'
                        )

    def test_the_groups_follow_the_entities(self):
        source = SHARED.read_text()
        assert 'groups=residual_groups' in source
        assert 'residual_groups.extend(' in source

    def test_alpha_still_varies_with_the_data(self):
        """Sem isto, a estabilidade acima poderia ser um valor constante."""
        X, y, entities = _panel()
        other = _panel(seed=99)
        assert self._alpha(X, y, entities) != self._alpha(*other)


class TestDiagnosticsNameTheRunningParadigm:
    """The shared model printed one paradigm's name in all three runs.

    "Simple Hierarchical Dask" appeared in the sql_engine and dataframe_lib
    logs too, because the line was a literal left over from when the code lived
    in the Dask pipeline. A log that names the wrong engine is worse than no
    log: it is the artifact someone reads to decide which run they are looking
    at.
    """

    def test_no_paradigm_is_named_in_a_literal(self):
        import ast as ast_module
        source = (_ROOT / 'src' / 'core' / 'models' / 'hierarchical.py')
        tree = ast_module.parse(source.read_text())
        docstrings = {id(n.value) for n in ast_module.walk(tree)
                      if isinstance(n, ast_module.Expr)
                      and isinstance(n.value, ast_module.Constant)}
        for node in ast_module.walk(tree):
            if not (isinstance(node, ast_module.Constant)
                    and isinstance(node.value, str)) or id(node) in docstrings:
                continue
            for stale in ('Dask', 'DuckDB', 'Polars', 'Data Lake',
                          'Data Warehouse'):
                assert stale not in node.value, (
                    f'line {node.lineno} names {stale!r} in shared code that '
                    f'all three paradigms run'
                )

    def test_the_diagnostic_uses_the_architecture_argument(self):
        source = (_ROOT / 'src' / 'core' / 'models' / 'hierarchical.py') \
            .read_text()
        assert 'Simple hierarchical ({architecture})' in source


class TestNoDeadImports:
    """An unused import of a model class reads as a second estimator."""

    MODELS = ['sql_engine', 'task_graph', 'dataframe_lib']

    @pytest.mark.parametrize('paradigm', MODELS)
    def test_the_hierarchical_model_imports_nothing_it_does_not_use(
            self, paradigm):
        import ast as ast_module
        path = (_ROOT / 'src' / 'architectures_ml' / paradigm / 'models'
                / 'hierarchical_model.py')
        tree = ast_module.parse(path.read_text())
        imported = {}
        for node in ast_module.walk(tree):
            if isinstance(node, (ast_module.Import, ast_module.ImportFrom)):
                for alias in node.names:
                    name = (alias.asname or alias.name).split('.')[0]
                    imported.setdefault(name, node.lineno)
        used = {n.id for n in ast_module.walk(tree)
                if isinstance(n, ast_module.Name)}
        used |= {n.value.id for n in ast_module.walk(tree)
                 if isinstance(n, ast_module.Attribute)
                 and isinstance(n.value, ast_module.Name)}
        dead = sorted((line, name) for name, line in imported.items()
                      if name not in used)
        assert not dead, (
            f'{paradigm}: imported and never used: '
            f'{[f"{n} (line {l})" for l, n in dead]}'
        )
