#!/usr/bin/env python3
"""Imputation respects P5, and the target is never fabricated.

The collection stage imputed with the mean of stratum peers in the same year and
with the mean of the whole panel across all years, then added calibrated noise to
the filled cells. Measured on the World Bank panel, the target's source column was
33.5% imputed -- 193 of 257 filled values coming from other countries -- and
features reached 81.5%.

Two distinct defects. Full-panel statistics written into training cells are a P5
violation (Kaufman et al., 2012), committed at the stage that precedes the folds,
where the P1-P5 gates cannot reach. And a fabricated target is worse than leakage:
accuracy measured against an imputed y is agreement with an imputation model, and a
mean is systematically easier to predict than real data, so R2 inflates without
predictive content.

The split follows from one asymmetry: forward fill within an entity fits no
statistic, so it is P5-safe by construction and stays in collection. Anything that
fits a statistic moved to the fold-scoped layer, once, for every paradigm.
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

from core.validation import impute_from_training_window

COLLECTOR = _SRC / 'collection' / 'raw_data_collector.py'
MODELS = sorted((_SRC / 'architectures_ml').glob('*/models/hierarchical_model.py'))


class TestCollectionNoLongerUsesPanelStatistics:

    def test_no_cross_sectional_fill_in_the_imputation_loop(self):
        source = COLLECTOR.read_text()
        block = source[source.index('def apply_conservative_imputation'):]
        block = block[:block.index('\n    def ', 1)]
        for forbidden in ('geographic_mask', 'stratum_values', 'global_mask',
                          'global_value'):
            assert forbidden not in block, (
                f'{forbidden} fills a cell with information from other entities '
                f'or from the whole panel'
            )

    def test_no_synthetic_noise(self):
        source = COLLECTOR.read_text()
        block = source[source.index('def apply_conservative_imputation'):]
        block = block[:block.index('\n    def ', 1)]
        assert 'noise' not in block, (
            'calibrated noise fabricates variance in cells later evaluated as '
            'observations'
        )

    def test_forward_fill_is_kept(self):
        """It fits no statistic, so it needs no fold awareness."""
        source = COLLECTOR.read_text()
        block = source[source.index('def apply_conservative_imputation'):]
        assert 'shift(1)' in block and 'groupby(' in block


class TestTargetIsNeverImputed:

    def test_the_loop_skips_the_target_source(self):
        source = COLLECTOR.read_text()
        assert 'if column == target_source:' in source
        assert 'continue' in source[source.index('if column == target_source:'):
                                    source.index('if column == target_source:') + 200]

    def test_rows_without_a_target_are_removed(self):
        source = COLLECTOR.read_text()
        assert 'df_imputed[target_source].notna()' in source, (
            'rows lacking an observed target must be dropped, not filled'
        )

    def test_coverage_is_recorded(self):
        """The extent of imputation belongs in an artifact, not a stray log."""
        source = COLLECTOR.read_text()
        assert 'target_coverage.json' in source
        assert 'observed_fraction' in source
        assert 'rows_removed_missing_target' in source


class TestFoldScopedImputation:

    @staticmethod
    def _frames():
        train = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [10.0, 20.0, np.nan]})
        test = pd.DataFrame({'a': [np.nan, 5.0], 'b': [np.nan, 1.0]})
        return train, test

    def test_statistics_come_from_the_training_frame(self):
        train, test = self._frames()
        (_, filled_test), report = impute_from_training_window(train, test)
        assert report['values']['a'] == pytest.approx(2.0)   # median of 1, 3
        assert filled_test['a'].iloc[0] == pytest.approx(2.0)

    def test_test_values_never_influence_the_statistic(self):
        """The regression this exists for: a statistic seeing the test window."""
        train, test = self._frames()
        _, report_a = impute_from_training_window(train, test)
        far_test = test.copy()
        far_test['a'] = [1e6, 1e6]
        _, report_b = impute_from_training_window(train, far_test)
        assert report_a['values'] == report_b['values'], (
            'changing the test frame changed the fitted statistic'
        )

    def test_no_missing_value_survives_in_a_covered_column(self):
        train, test = self._frames()
        (filled_train, filled_test), _ = impute_from_training_window(train, test)
        assert not filled_train['a'].isna().any()
        assert not filled_test['a'].isna().any()

    def test_column_unobserved_in_training_is_left_alone_and_reported(self):
        """Inventing a value for what training never saw is the old practice."""
        train = pd.DataFrame({'a': [1.0, 2.0], 'empty': [np.nan, np.nan]})
        test = pd.DataFrame({'a': [np.nan], 'empty': [np.nan]})
        (_, filled_test), report = impute_from_training_window(train, test)
        assert report['columns_without_training_observation'] == ['empty']
        assert filled_test['empty'].isna().all()

    def test_median_is_the_default(self):
        train = pd.DataFrame({'a': [1.0, 2.0, 100.0, np.nan]})
        _, report = impute_from_training_window(train)
        assert report['strategy'] == 'median'
        assert report['values']['a'] == pytest.approx(2.0), 'an outlier moved it'

    def test_mean_is_available_and_differs(self):
        train = pd.DataFrame({'a': [1.0, 2.0, 100.0, np.nan]})
        _, report = impute_from_training_window(train, strategy='mean')
        assert report['values']['a'] == pytest.approx(34.3333, abs=1e-3)

    def test_unknown_strategy_is_refused(self):
        with pytest.raises(ValueError, match='unsupported strategy'):
            impute_from_training_window(pd.DataFrame({'a': [1.0]}),
                                       strategy='knn')

    def test_input_frames_are_not_mutated(self):
        train, test = self._frames()
        impute_from_training_window(train, test)
        assert train['a'].isna().any() and test['a'].isna().any()


class TestEveryParadigmUsesTheSharedImplementation:

    def test_all_three_models_were_found(self):
        assert len(MODELS) == 3

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_model_imputes_from_the_training_window(self, path):
        """Checked as a call, not as text: the import line contains the name."""
        import ast

        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, 'id', None) == 'impute_from_training_window']
        assert calls, (
            f'{path.parts[-3]} imports the shared imputation but never calls it, '
            f'so the scaler will fail on the missing values collection now leaves '
            f'behind'
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_imputation_is_fitted_on_the_training_frame(self, path):
        """The first positional argument is what the statistic is fitted on."""
        import ast

        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    getattr(node.func, 'id', None) == 'impute_from_training_window':
                assert node.args, 'called without frames'
                first = node.args[0]
                assert getattr(first, 'id', '').startswith('X_train'), (
                    f'{path.parts[-3]} fits the imputation on '
                    f'{getattr(first, "id", "?")} rather than the training frame'
                )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_imputation_precedes_the_scaler(self, path):
        source = path.read_text()
        assert source.index('impute_from_training_window') < \
            source.index('StandardScaler()'), (
            'the scaler does not accept missing values'
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_no_paradigm_rolls_its_own_fill(self, path):
        source = path.read_text()
        body = source[source.index('def run_fold_analysis'):]
        for forbidden in ('.fillna(', '.interpolate(', 'SimpleImputer'):
            assert forbidden not in body, (
                f'{path.parts[-3]} fills values outside the shared '
                f'implementation, so the paradigms can preprocess differently'
            )
