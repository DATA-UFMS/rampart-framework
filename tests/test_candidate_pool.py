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
from core.validation import AntiLeakageViolation


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
        def apply_collinearity_filter(self, data, features, threshold=0.8):
            return features
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


class TestPoolGate:
    """P3 applied to the pool, independent of the policy that builds it.

    A candidate the correlation ceiling discards is never audited, so the pool
    itself has to be checked. These probes override the policy to simulate the
    regression the gate is there to survive.

    The panel is real and the correlations are empty, so without the gate
    run_feature_selection completes normally. A failure is then attributable to
    the gate rather than to the probe tripping over something else.
    """

    @staticmethod
    def _panel():
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(3)
        years = list(range(2000, 2016))
        return pd.DataFrame({
            'year': years,
            'country_code': ['BRA'] * len(years),
            'gini_index': rng.normal(size=len(years)),
            'lower_secondary_completion_rate': rng.normal(size=len(years)),
            'dropout_rate_sql_engine': rng.normal(size=len(years)),
        })

    @staticmethod
    def _leaking_probe(leak):
        class Leaking(_probe('sql_engine')):
            def get_numeric_features(self, data):
                return sorted(['gini_index', leak(self)])

        return Leaking

    def test_completes_without_the_leak(self, tmp_path):
        """Baseline: the probe reaches the end when the pool is clean."""
        class Clean(_probe('sql_engine')):
            def get_numeric_features(self, data):
                return ['gini_index']

        stats = Clean('sql_engine', str(tmp_path)).run_feature_selection(
            self._panel())
        assert stats['total_features_analyzed'] == 1

    def test_target_in_the_pool_halts(self, tmp_path):
        Leaking = self._leaking_probe(lambda s: s.target_column)
        with pytest.raises(AntiLeakageViolation, match='P3 data separation'):
            Leaking('sql_engine', str(tmp_path)).run_feature_selection(
                self._panel())

    def test_source_column_in_the_pool_halts(self, tmp_path):
        Leaking = self._leaking_probe(lambda s: s.source_column)
        with pytest.raises(AntiLeakageViolation, match='P3 data separation'):
            Leaking('sql_engine', str(tmp_path)).run_feature_selection(
                self._panel())

    def test_the_halt_names_the_offending_column(self, tmp_path):
        Leaking = self._leaking_probe(lambda s: s.source_column)
        with pytest.raises(AntiLeakageViolation) as exc:
            Leaking('sql_engine', str(tmp_path)).run_feature_selection(
                self._panel())
        assert 'lower_secondary_completion_rate' in str(exc.value)


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


class TestSelectionByCorrelation:
    """The rule that decides which candidates survive, untested until now.

    Widening it to accept everything left the whole suite green. It carries a
    relaxation branch that fires below five survivors and drops the upper
    bound entirely -- the bound that keeps a near-perfect proxy out. That
    branch is the one the real runs take, since the pool is small.
    """

    @staticmethod
    def _select(correlations, **kwargs):
        import contextlib
        import io
        architecture = _probe('sql_engine')('sql_engine', '/tmp')
        with contextlib.redirect_stdout(io.StringIO()):
            return architecture.select_features_by_correlation(correlations,
                                                               **kwargs)

    #: Six candidates, so the relaxation branch stays out of the way.
    BAND = {'a': 0.20, 'b': 0.35, 'c': 0.50, 'd': 0.65, 'e': 0.75, 'f': 0.79}

    def test_a_feature_below_the_floor_is_dropped(self):
        selected = self._select({**self.BAND, 'weak': 0.05})
        assert 'weak' not in selected

    def test_a_feature_above_the_ceiling_is_dropped(self):
        """The ceiling is the last thing between a proxy and the model."""
        selected = self._select({**self.BAND, 'proxy': 0.99})
        assert 'proxy' not in selected

    def test_features_inside_the_band_survive(self):
        assert sorted(self._select(self.BAND)) == sorted(self.BAND)

    def test_the_result_is_sorted(self):
        """Selection order must not depend on dict insertion order."""
        reversed_band = dict(reversed(list(self.BAND.items())))
        selected = self._select(reversed_band)
        assert selected == sorted(selected)

    def test_the_floor_is_inclusive_and_the_ceiling_is_inclusive(self):
        selected = self._select({**self.BAND, 'at_floor': 0.15,
                                 'at_ceiling': 0.80})
        assert 'at_floor' in selected and 'at_ceiling' in selected

    def test_just_outside_the_band_is_excluded(self):
        selected = self._select({**self.BAND, 'below': 0.15 - 1e-9,
                                 'above': 0.80 + 1e-9})
        assert 'below' not in selected and 'above' not in selected

    def test_negative_correlations_are_not_selected(self):
        """The rule compares the signed value, not its magnitude.

        Worth pinning: it is why the proxy audit downstream needs an absolute
        value, and why a feature negative in the training window never reaches
        the model at all.
        """
        selected = self._select({**self.BAND, 'inverse': -0.90})
        assert 'inverse' not in selected

    def test_the_relaxation_fires_below_five(self):
        selected = self._select({'a': 0.20, 'b': 0.12})
        assert 'b' in selected, (
            'the relaxed floor is 0.15 * 0.67 = 0.1005, so 0.12 qualifies'
        )

    def test_the_relaxation_drops_the_ceiling(self):
        """Documented here because it is a real hole, not an oversight.

        With fewer than five survivors the upper bound disappears and a feature
        correlating 0.99 with the target is selected. What keeps it out of the
        model is the proxy audit that runs afterwards over the full panel.
        """
        selected = self._select({'a': 0.20, 'proxy': 0.99})
        assert 'proxy' in selected

    def test_the_proxy_audit_covers_what_the_relaxation_admits(self):
        """So the hole above is closed downstream rather than left open."""
        import contextlib
        import io
        import numpy as np
        import pandas as pd
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import AntiLeakageViolation, audit_feature_set

        rng = np.random.default_rng(4)
        target = rng.normal(size=200)
        panel = pd.DataFrame({'target': target,
                              'proxy': 0.99 * target
                                       + 0.01 * rng.normal(size=200)})
        assert abs(panel['proxy'].corr(panel['target'])) > \
            SCIENTIFIC_CONFIG['proxy_correlation_threshold']
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(AntiLeakageViolation, match='proxy detection'):
                audit_feature_set(panel, ['proxy'], 'target',
                                  SCIENTIFIC_CONFIG)

    def test_an_empty_input_selects_nothing(self):
        assert self._select({}) == []
