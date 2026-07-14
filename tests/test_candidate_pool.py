#!/usr/bin/env python3
"""Candidate pool for feature selection.

The pool must be identical across paradigms: a paradigm that starts from a
different search space is not selecting features under the same conditions as
the others, which is what the cross-paradigm comparison assumes.

The regression these tests exist for: the target's own source column, which
correlates -1.0 with the target, reached one paradigm's pool because that
paradigm enumerated its exclusions in a literal list instead of deriving them.
Nothing in the P3 gate saw it -- the audit runs over the selected features, not
over the candidates -- so it was discarded only by the selection's correlation
ceiling.
"""

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.base_architecture import BaseArchitectureML
from core.paradigm_registry import discover_paradigms


# Mirrors the real schema: legitimate features, the target of every paradigm,
# the target's lags, the column the target is derived from, and metadata.
SCHEMA = [
    'gini_index',
    'internet_users_percent',
    'lower_secondary_completion_rate',
    'dropout_rate_sql_engine',
    'dropout_rate_task_graph',
    'dropout_rate_dataframe_lib',
    'dropout_rate_lag_2',
    'dropout_rate_lag_3',
    'year',
    'country_code',
    'country_name',
    'data_completeness_score',
]

EXPECTED_POOL = ['gini_index', 'internet_users_percent']


class _Config:
    """Dataset config whose excluded_columns does NOT list the target source.

    This is the state that produced the bug: INEP's excluded_columns named
    columns that did not exist, so it never covered the target's source column.
    A pool that stays correct here is protected by the policy rather than by the
    configuration happening to be complete.
    """

    year_column = 'year'
    entity_column = 'country_code'
    entity_name_column = 'country_name'
    stratification_column = None
    target_source_column = 'lower_secondary_completion_rate'
    feature_columns = ['gini_index', 'internet_users_percent']
    excluded_columns = ['municipality_code', 'state_code']


def _probe(name, schema=SCHEMA):
    class Probe(BaseArchitectureML):
        def setup_environment(self): pass
        def load_data(self): pass
        def validate_data(self, data): pass
        def create_target_implementation(self, data): return data
        def _compute_target_statistics(self, data): pass
        def _validate_temporal_folds(self, data, folds): pass
        def save_folds(self, data, folds): pass
        def compute_feature_correlations(self, data, features): return {}
        def apply_collinearity_filter(self, data, features): return features
        def prepare_features(self, data, features): return data
        def discover_numeric_columns(self, data): return list(schema)

    return Probe


class TestPoolPolicy:

    def test_pool_holds_only_legitimate_features(self, tmp_path):
        arch = _probe('sql_engine')('sql_engine', str(tmp_path))
        assert arch.get_numeric_features(None) == EXPECTED_POOL

    def test_source_column_is_never_a_candidate(self, tmp_path):
        """The regression: correlation -1.0 with the target.

        Uses a config that does not list the source column, so passing depends
        on the policy and not on the configuration being complete.
        """
        arch = _probe('sql_engine')('sql_engine', str(tmp_path),
                                    dataset_config=_Config())
        pool = arch.get_numeric_features(None)
        assert arch.source_column not in pool
        # data_completeness_score survives: excluding dataset metadata is the
        # configuration's job, and this config is deliberately incomplete. Every
        # target-derived column is gone regardless.
        assert pool == ['data_completeness_score', 'gini_index',
                        'internet_users_percent']

    def test_other_paradigms_targets_are_never_candidates(self, tmp_path):
        """A paradigm must not train on another paradigm's copy of the target."""
        arch = _probe('sql_engine')('sql_engine', str(tmp_path))
        pool = arch.get_numeric_features(None)
        assert 'dropout_rate_task_graph' not in pool
        assert 'dropout_rate_dataframe_lib' not in pool

    def test_target_lags_are_never_candidates(self, tmp_path):
        """Lags are appended by the models, after selection."""
        arch = _probe('sql_engine')('sql_engine', str(tmp_path))
        pool = arch.get_numeric_features(None)
        assert not [c for c in pool if '_lag_' in c]

    def test_pool_is_sorted(self, tmp_path):
        """Selection order must not depend on schema order."""
        shuffled = list(reversed(SCHEMA))
        arch = _probe('x', shuffled)('sql_engine', str(tmp_path))
        pool = arch.get_numeric_features(None)
        assert pool == sorted(pool)

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_pool_does_not_depend_on_the_paradigm(self, paradigm, tmp_path):
        arch = _probe(paradigm)(paradigm, str(tmp_path))
        assert arch.get_numeric_features(None) == EXPECTED_POOL

    def test_declared_feature_colliding_with_the_prefix_aborts(self, tmp_path):
        """Silently dropping a legitimate feature would change results."""
        arch = _probe('sql_engine')('sql_engine', str(tmp_path))
        real = arch.dataset_config

        class Config:
            feature_columns = ['gini_index', 'dropout_rate_of_teachers']

            def __getattr__(self, item):
                return getattr(real, item)

        arch.dataset_config = Config()
        with pytest.raises(ValueError, match='reserved for target-derived'):
            arch.get_numeric_features(None)


class TestPolicyIsNotOverridden:
    """The policy lives in the base class, once."""

    def test_no_paradigm_defines_its_own_pool_policy(self):
        for name, meta in sorted(discover_paradigms().items()):
            cls = BaseArchitectureML._registry[name]
            for klass in cls.__mro__:
                if klass is BaseArchitectureML:
                    break
                assert 'get_numeric_features' not in klass.__dict__, (
                    f"{name}: {klass.__name__} overrides get_numeric_features, "
                    f"so its candidate pool can diverge from the other "
                    f"paradigms. Override discover_numeric_columns instead."
                )
                assert 'candidate_exclusions' not in klass.__dict__, (
                    f"{name}: {klass.__name__} overrides candidate_exclusions, "
                    f"so its exclusion policy can diverge from the other "
                    f"paradigms."
                )

    def test_target_is_derived_from_the_stem(self):
        """The prefix rule only covers the targets if they share the stem."""
        for name in sorted(discover_paradigms()):
            assert name in f'{BaseArchitectureML.TARGET_STEM}_{name}'
