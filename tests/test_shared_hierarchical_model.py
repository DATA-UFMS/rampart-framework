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

    #: The anchored prediction vector, captured on the reference machine.
    #: Twelve significant digits: the machine-to-machine variation this
    #: test exists to tolerate lives far below that, and a change in
    #: behaviour lives far above it.
    ANCHOR_VALUES = (
        0.284427774414, 0.151939393543, -0.390101308994,
        -0.258143549023, -0.646382345967, -0.100596202308,
        -0.149884588174, -0.179016876685, -0.469049317026,
        0.115530791198, -0.975724198504, -0.109168477788,
        -0.590592675518, 0.443159808044, -0.146140810064,
        -0.0498414110694, -0.0134746793179, -0.238552634851,
        0.131810301402, 0.115720081852, 0.271259984334, 0.38915942002,
        -0.734687463408, -0.303359112532, -0.041166631428,
        -0.100063161145, -0.698882150468, 0.918769894948,
        -0.925338345758, -0.677436994365, -0.143896570995,
        0.0957432315001, 0.322534585077, -0.00483703436976,
        -0.365919710629, -0.149703683552, -0.0380064617983,
        -0.0957350156063, -0.452914485268, 0.37146420961,
        -0.267454854849, -0.799026333327, -0.00393636088951,
        0.205182022427, -0.118080754089, -0.0410847365973,
        -0.864541589889, -0.337058963345, 0.140310722882,
        -0.205192312336, -0.360323306044, 0.0848595767971,
        -0.480838682632, -0.640902182022, -0.098421918893,
        0.143657660055, -0.0173393972502, -0.231868502597,
        -0.854882834494, -0.0452307811504, 0.118268918212,
        -0.321399163935, -0.256632563461, 0.18859806971,
        -0.282493965815, -0.329572405046, -0.107898069867,
        0.0185120353224, -0.52076672396, 0.387601761989,
        -0.637343334446, 0.363129657813, -0.388019911539,
        0.819618924666, -0.705077487587, 0.456937962413, -0.65513734735,
        -0.152656910233, 0.183866801035, -0.275960119977, 0.38581967544,
        0.159616205888, -0.617144513598, 0.0994844003182, 0.39013933411,
        -0.153637941994, 0.00783125566054, -0.072363096613,
        -0.177836501104, -0.172120277387,
    )

    @pytest.mark.parametrize('shrinkage', [0.6, 0.8, 1.0])
    def test_the_three_paradigms_agree_bitwise(self, shrinkage):
        """The premise of the equivalence claim, at the level of one method."""
        digests = {paradigm: _digest(_run(paradigm, shrinkage))
                   for paradigm in PARADIGMS}
        assert len(set(digests.values())) == 1, digests

    #: How far the anchored predictions may move before the difference stops
    #: being arithmetic and starts being a change in behaviour. Two orders of
    #: magnitude below the SESOI on R2 (0.01), so a drift this size cannot
    #: reach any published decision.
    ANCHOR_TOLERANCE = 1e-4

    def test_the_captured_values_still_hold(self):
        """One anchored vector, so agreement between paradigms cannot drift together.

        Compared within a tolerance rather than by digest, and the reason is a
        measurement rather than caution. The digest form failed on CI and passed
        on re-run of the same commit, same interpreter: GitHub allocates runners
        with different instruction sets, and a different reduction order in BLAS
        moves the last bits. The absolute values are a property of the machine.

        What is *not* machine-dependent, and is the claim the paper makes, is
        that the three paradigms agree with each other -- that test passed on
        every runner, including the ones where this one failed. Keeping a
        bitwise anchor here would have gone on reporting hardware allocation as
        a regression in the model.
        """
        import numpy as np

        values = np.asarray(
            _run('task_graph', self.ANCHOR_SHRINKAGE)['predictions'],
            dtype=float)
        anchored = np.asarray(self.ANCHOR_VALUES, dtype=float)
        assert values.shape == anchored.shape, (
            f'the anchored vector changed shape: {values.shape} != '
            f'{anchored.shape}')
        worst = float(np.max(np.abs(values - anchored)))
        assert worst <= self.ANCHOR_TOLERANCE, (
            f'predictions moved by {worst:.2e}, beyond the {self.ANCHOR_TOLERANCE:.0e} '
            f'attributable to floating-point reduction order; this is a change '
            f'in behaviour, not in hardware')

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
                         'write_imputation_report',
                         'write_feature_audit'}, names


class TestInnerCrossValidationIsDeliberate:
    """The partition of the alpha selection, explicit rather than accidental.

    cv=<int> makes RidgeCV use KFold without shuffle. Since the residuals are
    concatenated by entity, the contiguous blocks were entity blocks: the alpha
    selection was doing leave-some-entities-out without anyone having chosen
    it, and it would change silently if the row order changed.

    It is not leakage in either of the two forms -- every residual comes from
    the training window. What the change buys is determinism, and that is what
    this test pins down: identical behaviour under a permutation of the rows.
    An output test would not distinguish the two versions, because the
    resulting partition is the same.
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
            f'alpha changed with the row order: {original} != {permuted}, '
            f'which means the partition of the inner CV is implicit'
        )

    def test_the_splitter_is_declared_not_an_integer(self):
        """cv=<int> delegates the choice of the partition to sklearn."""
        source = SHARED.read_text()
        assert 'GroupKFold' in source
        tree = ast.parse(source)
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and \
                    getattr(call.func, 'id', None) == 'RidgeCV':
                for keyword in call.keywords:
                    if keyword.arg == 'cv':
                        assert not isinstance(keyword.value, ast.Constant), (
                            'cv passed as a literal reintroduces the implicit '
                            'partition'
                        )

    def test_the_groups_follow_the_entities(self):
        source = SHARED.read_text()
        assert 'groups=residual_groups' in source
        assert 'residual_groups.extend(' in source

    def test_alpha_still_varies_with_the_data(self):
        """Without this, the stability above could be a constant value."""
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
