#!/usr/bin/env python3
"""Behaviour under degenerate folds, pinned rather than assumed.

The panels are real and uneven: a fold can hold one entity, one year, a column
that never varies, or a target that does not move. None of that crashes the
pipeline today, and most of it is handled deliberately -- the inner
cross-validation falls back to generalized CV below two groups, the
reconstruction check returns no verdict on a constant target, imputation
refuses a column with nothing observed in training.

But "does not crash" was resting on nothing. A twenty-nine hour run reaching a
degenerate fold at hour twenty and dying there costs the run; reaching it and
silently producing a number costs more.

One boundary is now load-bearing and was not tested: the selection ceiling and
the proxy audit read the same parameter with different operators -- `<=` to
admit, `>` to flag. A feature sitting exactly on it must be admitted by one
and cleared by the other, or the two disagree about the same number.
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import audit_panel

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT / 'tests') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'tests'))

from core.scientific_config import SCIENTIFIC_CONFIG
from core.validation import (audit_feature_set, canonical_fold,
                             impute_from_training_window,
                             linear_reconstruction_r2)


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings('ignore')
        return function(*args, **kwargs)


def _architecture():
    from test_candidate_pool import _probe
    return _quiet(_probe('sql_engine'), 'sql_engine', '/tmp')


class TestTheThresholdIsReadConsistently:
    """Selection admits with `<=`; the audit flags with `>`. Same number."""

    THRESHOLD = SCIENTIFIC_CONFIG['proxy_correlation_threshold']

    def test_a_feature_exactly_on_the_ceiling_is_admitted(self):
        selected = _quiet(
            _architecture().select_features_by_correlation,
            {'edge': self.THRESHOLD, 'other': 0.30})
        assert 'edge' in selected

    def test_the_two_operators_partition_the_line(self):
        """No gap and no overlap: `<=` to admit, `>` to flag.

        Exactly on the boundary is not constructible in floating point -- a
        correlation built to be 0.80 comes out 0.8000000000000007 -- so the
        invariant is checked where it lives, in the two comparisons. Changing
        one to `<` without changing the other to `>=` opens a band where a
        feature is refused by selection and cleared by the audit, or admitted
        by both.
        """
        import ast as ast_module
        source = (_SRC / 'core' / 'validation.py').read_text()
        audit = next(node for node in ast_module.walk(ast_module.parse(source))
                     if isinstance(node, ast_module.FunctionDef)
                     and node.name == 'audit_feature_set')
        flags = [node for node in ast_module.walk(audit)
                 if isinstance(node, ast_module.Compare)
                 and any(isinstance(op, (ast_module.Gt, ast_module.GtE))
                         for op in node.ops)
                 and 'proxy_threshold' in ast_module.unparse(node)]
        assert flags, 'the audit no longer compares against the threshold'
        for node in flags:
            assert isinstance(node.ops[0], ast_module.Gt), (
                f'the audit flags with {ast_module.unparse(node)}; selection '
                f'admits with <=, so this must be > or a band opens'
            )

        selection = (_SRC / 'core' / 'base_architecture.py').read_text()
        assert 'abs(float(correlation)) <= ceiling' in selection

    def test_just_below_the_ceiling_is_admitted_and_cleared(self):
        margin = self.THRESHOLD - 1e-6
        selected = _quiet(_architecture().select_features_by_correlation,
                          {'edge': margin, 'other': 0.30})
        assert 'edge' in selected

        rng = np.random.default_rng(5)
        size = 400
        target = rng.normal(size=size)
        noise = rng.normal(size=size)
        noise -= (np.cov(target, noise, bias=True)[0, 1] / target.var()) * target
        edge = margin * (target / target.std()) + np.sqrt(
            1 - margin ** 2) * (noise / noise.std())
        panel = pd.DataFrame({'target': target, 'edge': edge})
        assert abs(panel['edge'].corr(panel['target'])) < self.THRESHOLD
        report = audit_panel(panel, ['edge'], 'target')
        assert report['features_audited'] == ['edge']

    def test_just_above_the_ceiling_is_refused_and_flagged(self):
        """Otherwise the two tests above could both be trivially satisfied."""
        from core.validation import AntiLeakageViolation
        selected = _quiet(_architecture().select_features_by_correlation,
                          {'edge': 0.30, 'over': self.THRESHOLD + 0.05})
        assert 'over' not in selected

        rng = np.random.default_rng(6)
        target = rng.normal(size=200)
        panel = pd.DataFrame({'target': target,
                              'over': 0.99 * target + 0.01 * rng.normal(size=200)})
        with pytest.raises(AntiLeakageViolation):
            audit_panel(panel, ['over'], 'target')


class TestConstantColumns:

    def test_a_constant_feature_is_not_selected(self):
        """No variation, no association to measure."""
        selected = _quiet(_architecture().select_features_by_correlation,
                          {'a': 0.30, 'constant': float('nan')})
        assert selected == ['a']

    def test_its_disappearance_is_recorded(self):
        """Counting candidates against selected would show an unexplained gap."""
        _, bounds = _quiet(_architecture().select_features_with_bounds,
                           {'a': 0.30, 'constant': float('nan')})
        assert bounds['undefined_correlation'] == ['constant']

    def test_nothing_is_recorded_when_all_are_defined(self):
        _, bounds = _quiet(_architecture().select_features_with_bounds,
                           {'a': 0.30, 'b': 0.40})
        assert bounds['undefined_correlation'] == []

    def test_all_undefined_halts_and_says_why(self):
        with pytest.raises(ValueError, match='with an undefined correlation'):
            _quiet(_architecture().select_features_by_correlation,
                   {'a': float('nan'), 'b': float('nan')})

    def test_a_constant_feature_explains_nothing(self):
        panel = pd.DataFrame({'k': [5.0] * 30,
                              't': np.random.default_rng(1).normal(size=30)})
        assert linear_reconstruction_r2(panel, ['k'], 't') == 0.0

    def test_a_constant_target_yields_no_verdict(self):
        """Not zero: with no variance to explain, R2 is undefined."""
        panel = pd.DataFrame({'f': np.arange(30.0), 't': [7.0] * 30})
        assert linear_reconstruction_r2(panel, ['f'], 't') is None


class TestSingleEntityAndSingleYear:

    def test_one_entity_is_a_valid_fold(self):
        frame = pd.DataFrame({'e': ['A'] * 6, 'y': range(2000, 2006),
                              'f': np.arange(6.0), 't': np.arange(6.0)})
        X, y, entities = canonical_fold(frame[['f']], frame['t'], frame['e'],
                                        frame['y'], paradigm='probe')
        assert len(X) == len(y) == len(entities) == 6

    def test_one_year_across_entities_is_a_valid_fold(self):
        frame = pd.DataFrame({'e': list('ABCDEF'), 'y': [2000] * 6,
                              'f': np.arange(6.0), 't': np.arange(6.0)})
        X, _, _ = canonical_fold(frame[['f']], frame['t'], frame['e'],
                                 frame['y'], paradigm='probe')
        assert len(X) == 6

    def test_one_entity_and_one_year_is_a_single_row(self):
        frame = pd.DataFrame({'e': ['A'], 'y': [2000], 'f': [1.0],
                              't': [2.0]})
        X, _, _ = canonical_fold(frame[['f']], frame['t'], frame['e'],
                                 frame['y'], paradigm='probe')
        assert len(X) == 1


class TestTheHierarchicalModelSurvives:
    """Each of these took a different branch and none of them crashed."""

    @staticmethod
    def _fit(rows, entities, constant_feature=False, constant_target=False,
             train=None):
        from core.models.hierarchical import simple_hierarchical_model
        rng = np.random.default_rng(2)
        X = pd.DataFrame({
            'a': np.ones(rows) if constant_feature else rng.normal(size=rows),
            'b': rng.normal(size=rows)})
        y = pd.Series([3.0] * rows if constant_target
                      else rng.normal(size=rows))
        group = pd.Series(np.resize([f'C{index}' for index in range(entities)],
                                    rows))
        cut = train if train is not None else rows * 3 // 4
        return _quiet(simple_hierarchical_model, X[:cut], y[:cut], X[cut:],
                      y[cut:], group[:cut], group[cut:],
                      architecture='sql_engine')

    def test_a_single_entity_falls_back_to_generalized_cv(self):
        """Below two groups GroupKFold cannot split; the fallback is deliberate."""
        result = self._fit(40, 1)
        assert len(result['predictions']) == 10
        assert np.all(np.isfinite(result['predictions']))

    def test_two_entities_use_the_grouped_split(self):
        result = self._fit(40, 2)
        assert np.all(np.isfinite(result['predictions']))

    def test_a_constant_feature_does_not_break_the_fit(self):
        result = self._fit(40, 4, constant_feature=True)
        assert np.all(np.isfinite(result['predictions']))

    def test_a_constant_target_is_predicted_exactly(self):
        result = self._fit(40, 4, constant_target=True)
        assert np.allclose(result['predictions'], 3.0)

    def test_a_tiny_training_window_still_fits(self):
        result = self._fit(6, 2, train=4)
        assert len(result['predictions']) == 2


class TestImputationUnderDegenerateWindows:

    def test_a_single_training_row_gives_that_value(self):
        (_, applied), report = impute_from_training_window(
            pd.DataFrame({'a': [1.0]}), pd.DataFrame({'a': [np.nan]}))
        assert report['values']['a'] == 1.0
        assert applied['a'].iloc[0] == 1.0

    def test_a_column_with_nothing_observed_halts(self):
        with pytest.raises(ValueError,
                           match='no observation in the training window'):
            impute_from_training_window(pd.DataFrame({'a': [np.nan, np.nan]}),
                                        pd.DataFrame({'a': [np.nan]}))

    def test_a_constant_training_column_is_usable(self):
        """Zero variance is not missing data; the median is defined."""
        (_, applied), report = impute_from_training_window(
            pd.DataFrame({'a': [4.0, 4.0, 4.0]}),
            pd.DataFrame({'a': [np.nan, 4.0]}))
        assert report['values']['a'] == 4.0
        assert applied['a'].tolist() == [4.0, 4.0]
