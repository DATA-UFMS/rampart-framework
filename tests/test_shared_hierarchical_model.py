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
        assert names == {'simple_hierarchical_model', 'write_prediction_artifact',
                         'write_baseline_predictions'}, names
