#!/usr/bin/env python3
"""Which models a run fits, and what it costs a run that asked for none.

The registry exists so the factorial can add a capacity ladder and two
in-context families without the published run changing. That guarantee is the
whole point, so most of what is below is about the empty case: nothing built,
nothing imported, nothing different.

The rest is about the failure that would be worst to have silently -- a
misspelled model name quietly fitting fewer models than the arm's label claims,
leaving a hole in the ladder and no evidence of why.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.models import registry  # noqa
from core.models.ladder import LADDER, RUNGS  # noqa


def panel(rows=120, entities=6, seed=0):
    rng = np.random.default_rng(seed)
    entity = pd.Series([f'E{i % entities}' for i in range(rows)])
    frame = pd.DataFrame(rng.normal(size=(rows, 4)),
                         columns=[f'f{i}' for i in range(4)])
    target = pd.Series(frame['f0'] * 2 - frame['f1'] + rng.normal(scale=.3,
                                                                  size=rows))
    years = pd.Series([2000 + (i // entities) for i in range(rows)])
    return frame, target, entity, years


def split_panel(rows=150, entities=5, seed=4):
    """Distinct training and evaluation frames, with noise in the target.

    Both matter. Handing the training frame back as the evaluation frame lets a
    decision tree predict every evaluation row exactly, so every absorption
    probe divides by a residual of zero and is skipped -- which is the primitive
    behaving correctly and the test asking the wrong thing.
    """
    rng = np.random.default_rng(seed)
    def make(n, offset):
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=['a', 'b', 'c'])
        y = pd.Series(X['a'] * 2 - X['b'] + rng.normal(scale=1.0, size=n))
        entity = pd.Series([f'E{i % entities}' for i in range(n)])
        years = pd.Series([offset + i // entities for i in range(n)])
        return X, y, entity, years
    return make(rows, 2000), make(max(20, rows // 4), 2100)


class TestSilenceByDefault:

    def test_no_variable_means_no_extra_models(self, monkeypatch):
        monkeypatch.delenv(registry.ENV_VAR, raising=False)
        assert registry.requested() == []

    def test_empty_variable_is_the_same_as_absent(self, monkeypatch):
        monkeypatch.setenv(registry.ENV_VAR, '   ')
        assert registry.requested() == []

    def test_fitting_none_builds_nothing(self, monkeypatch):
        monkeypatch.delenv(registry.ENV_VAR, raising=False)
        X, y, entity, years = panel()
        assert registry.fit_requested(X, y, X, y, entity, entity,
                                      architecture='dataframe_lib',
                                      years_train=years) == {}

    def test_importing_the_registry_does_not_import_torch(self):
        """The published artifact reproduces without a deep learning stack.

        Asserted on a subprocess rather than on sys.modules, because by the time
        this test runs another test may already have imported torch for its own
        reasons and the check would pass for the wrong reason.
        """
        import subprocess
        result = subprocess.run(
            [sys.executable, '-c',
             'import sys; sys.path.insert(0, %r);'
             'import core.models.registry;'
             'print(any(m in sys.modules for m in ("torch", "tabpfn", "tabicl")))'
             % str(_SRC)],
            capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'False', (
            'importing the registry pulled in an optional dependency')


class TestWhatWasAskedFor:

    def test_a_group_expands_to_its_members(self, monkeypatch):
        monkeypatch.setenv(registry.ENV_VAR, 'ladder')
        assert registry.requested() == list(RUNGS)

    def test_an_unknown_name_stops_the_run(self, monkeypatch):
        monkeypatch.setenv(registry.ENV_VAR, 'ladder_kn')
        with pytest.raises(ValueError, match='unknown model'):
            registry.requested()

    def test_the_order_does_not_depend_on_how_it_was_typed(self, monkeypatch):
        monkeypatch.setenv(registry.ENV_VAR, 'ladder_knn,ladder_ridge')
        one = registry.requested()
        monkeypatch.setenv(registry.ENV_VAR, 'ladder_ridge,ladder_knn')
        assert registry.requested() == one == ['ladder_ridge', 'ladder_knn']

    def test_a_name_repeated_is_fitted_once(self, monkeypatch):
        monkeypatch.setenv(registry.ENV_VAR, 'ladder_ridge,ladder,ladder_ridge')
        assert registry.requested().count('ladder_ridge') == 1

    def test_the_variable_is_read_at_the_point_of_use(self, monkeypatch):
        """Paradigms run as subprocesses; a cached answer would outlive them."""
        monkeypatch.delenv(registry.ENV_VAR, raising=False)
        assert registry.requested() == []
        monkeypatch.setenv(registry.ENV_VAR, 'ladder_knn')
        assert registry.requested() == ['ladder_knn']


class TestTheLadderFits:

    def test_every_rung_returns_the_shared_contract(self, monkeypatch):
        monkeypatch.setenv(registry.ENV_VAR, 'ladder')
        X, y, entity, years = panel()
        results = registry.fit_requested(X, y, X, y, entity, entity,
                                         architecture='sql_engine',
                                         years_train=years)
        assert set(results) == set(RUNGS)
        for name, result in results.items():
            assert result['model_name'] == name
            assert result['architecture'] == 'sql_engine'
            assert len(result['predictions']) == len(y)
            assert np.isfinite(result['r2'])
            for key in ('mse', 'rmse', 'mae', 'entities', 'features_count'):
                assert key in result, f'{name} is missing {key}'

    def test_the_rungs_are_ordered_by_the_severity_they_are_ordered_by(self):
        severities = [rung.roth_severity for rung in LADDER]
        assert severities == sorted(severities), (
            'the ladder is declared in capacity order; a rung out of place '
            'would be read as a trend reversal rather than as a typo')

    def test_the_entity_effect_is_fitted_on_training_rows_only(self):
        """P5, for the one column the ladder adds itself.

        An entity that appears only in the evaluation window must receive the
        global training mean. Its own mean could only be read from the window
        being predicted.
        """
        from core.models.ladder import ENTITY_EFFECT_COLUMN, entity_effect_frames
        X, y, entity, _years = panel(rows=40, entities=4)
        unseen = pd.Series(['NEW'] * len(entity))
        _train, test_augmented, _means, global_mean = entity_effect_frames(
            X, X, y, entity, unseen)
        assert (test_augmented[ENTITY_EFFECT_COLUMN] == global_mean).all()
        assert global_mean == pytest.approx(float(y.mean()))


class TestTheSummaryFollowsWhatWasFitted:
    """The aggregate print used to iterate a list written out in each paradigm.

    Three copies of the same two names. A rung added to the run was a rung
    missing from all three summaries, and the summary gave no sign of it -- it
    was simply short, which reads exactly like complete.
    """

    def test_it_reports_every_model_the_folds_carry(self):
        folds = [{'models': {'simple_hierarchical': {}, 'ladder_knn': {}}},
                 {'models': {'simple_hierarchical': {}, 'icl_tabpfn': {}}}]
        assert registry.models_reported(folds) == [
            'simple_hierarchical', 'ladder_knn', 'icl_tabpfn']

    def test_a_fold_without_models_does_not_break_it(self):
        assert registry.models_reported([{}, {'models': {}}]) == []

    def test_no_paradigm_writes_the_list_out_again(self):
        """Read from the syntax tree: a list of model names inside a paradigm
        is the arrangement this replaced."""
        import ast

        offending = []
        for path in sorted((_SRC / 'architectures_ml').rglob('hierarchical_model.py')):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, (ast.List, ast.Tuple)):
                    continue
                names = {element.value for element in node.elts
                         if isinstance(element, ast.Constant)
                         and isinstance(element.value, str)}
                if {'simple_hierarchical', 'random_forest_hierarchical'} <= names:
                    offending.append(f'{path.name}:{node.lineno}')
        assert not offending, (
            f'a paradigm lists the models again instead of deriving them: '
            f'{offending}')


class TestAbsorptionIsMeasuredOnlyOnACleanArm:
    """The axis the ladder is read along is a property of the model.

    Read off an injected frame it would answer a different question -- how much
    of one handed answer a model keeps when it has already been handed several --
    and nothing reads that. The decision lives in the registry so the two fitters
    cannot come to disagree about when it applies.
    """

    def test_a_clean_run_records_it(self, monkeypatch):
        monkeypatch.setenv(registry.ENV_VAR, 'ladder_ridge,ladder_decision_tree')
        monkeypatch.delenv('RAMPART_INJECTION', raising=False)
        (X, y, e, yr), (Xe, ye, ee, _) = split_panel()
        results = registry.fit_requested(X, y, Xe, ye, e, ee,
                                         architecture='dataframe_lib',
                                         years_train=yr)
        for name, result in results.items():
            assert result['absorption'] is not None, name
            assert result['absorption']['probes_used'] > 0, name

    def test_an_injected_run_does_not(self, monkeypatch):
        monkeypatch.setenv(registry.ENV_VAR, 'ladder_ridge')
        monkeypatch.setenv('RAMPART_INJECTION',
                           '{"class": "C3", "dose": 0.1}')
        (X, y, e, yr), (Xe, ye, ee, _) = split_panel()
        results = registry.fit_requested(X, y, Xe, ye, e, ee,
                                         architecture='dataframe_lib',
                                         years_train=yr)
        assert results['ladder_ridge']['absorption'] is None

    def test_the_reading_separates_the_two_extreme_rungs(self, monkeypatch):
        """End to end through the registry, not just through the primitive."""
        monkeypatch.setenv(registry.ENV_VAR, 'ladder_ridge,ladder_decision_tree')
        monkeypatch.delenv('RAMPART_INJECTION', raising=False)
        (X, y, e, yr), (Xe, ye, ee, _) = split_panel(seed=5)
        results = registry.fit_requested(X, y, Xe, ye, e, ee,
                                         architecture='dataframe_lib',
                                         years_train=yr)
        ridge = results['ladder_ridge']['absorption']['absorption']
        tree = results['ladder_decision_tree']['absorption']['absorption']
        assert ridge < tree, f'ridge {ridge}, tree {tree}'
