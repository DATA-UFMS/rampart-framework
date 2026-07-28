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
        # The comments record that the noise was removed, so the word survives
        # in prose; what must not survive is the code that produces it.
        code = '\n'.join(line.split('#')[0] for line in block.splitlines())
        assert 'noise' not in code, (
            'calibrated noise fabricates variance in cells later evaluated as '
            'observations'
        )
        assert 'np.random' not in code, (
            'drawing random values here is how the calibrated noise was added'
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


class TestDiagnosticsMeasureWhatIsApplied:
    """A published diagnostic of an unapplied method is worse than none.

    Both diagnostics replicated the old cascade. The leave-one-out estimate
    reported the error of temporal-then-geographic filling, and the sensitivity
    report compared three methods as though the results depended on the choice
    among them. Their outputs are published artifacts.
    """

    def test_leave_one_out_no_longer_imputes_geographically(self):
        source = COLLECTOR.read_text()
        block = source[source.index('def perform_leave_one_out_validation'):]
        block = block[:block.index('\n    def ', 1)]
        assert '_apply_geographic_imputation' not in block, (
            'the estimate would report the error of a method not applied'
        )

    def test_leave_one_out_still_measures_forward_fill(self):
        source = COLLECTOR.read_text()
        block = source[source.index('def perform_leave_one_out_validation'):]
        block = block[:block.index('\n    def ', 1)]
        assert 'shift(1)' in block, 'nothing is being measured any more'
        assert 'mae' in block.lower()

    def test_the_comparison_is_not_called_a_sensitivity_analysis(self):
        source = COLLECTOR.read_text()
        assert 'def perform_sensitivity_analysis' not in source, (
            'the results do not depend on a choice among the three, since only '
            'one is applied'
        )
        assert 'def compare_candidate_imputation_methods' in source

    def test_the_comparison_records_which_method_is_applied(self):
        source = COLLECTOR.read_text()
        assert "'applied_method': 'temporal_only'" in source
        assert "'not_applied'" in source

    def test_geographic_helper_declares_it_is_not_applied(self):
        """It survives only to quantify the rejected alternative."""
        source = COLLECTOR.read_text()
        block = source[source.index('def _apply_geographic_imputation'):]
        block = block[:block.index('"""', block.index('"""') + 3)]
        assert 'NOT APPLIED TO THE DATA' in block

    def test_no_unverifiable_claims_remain(self):
        """"Reduces RMSE by 23% (data not shown)" is not a citation."""
        source = COLLECTOR.read_text()
        for claim in ('dados não mostrados', 'análise não mostrada',
                      'Reduz RMSE em 23%'):
            assert claim not in source, f'unverifiable claim: {claim}'

    def test_function_docstring_no_longer_describes_the_removed_tiers(self):
        source = COLLECTOR.read_text()
        block = source[source.index('def apply_conservative_imputation'):]
        block = block[:block.index('"""', block.index('"""') + 3)]
        for stale in ('GEOGRÁFICA ESTRATIFICADA', 'GLOBAL CONSERVADORA',
                      'RUÍDO ESTOCÁSTICO'):
            assert stale not in block, (
                f'the docstring still describes {stale}, which the code no '
                f'longer does'
            )


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

    def test_column_unobserved_in_training_raises(self):
        """It cannot occur under an expansive window + P4 selection; if it
        does, it stops.

        The alternatives are worse: a constant fabricates a value the training
        window never observed and makes the feature constant in training and
        variable in test; dropping it changes the feature set between folds and
        between paradigms; leaving it missing defers the failure to RidgeCV,
        because StandardScaler propagates NaN silently rather than rejecting it.
        """
        train = pd.DataFrame({'a': [1.0, 2.0], 'empty': [np.nan, np.nan]})
        test = pd.DataFrame({'a': [np.nan], 'empty': [np.nan]})
        with pytest.raises(ValueError,
                           match='no observation in the training window'):
            impute_from_training_window(train, test)

    def test_the_error_names_the_offending_columns(self):
        train = pd.DataFrame({'a': [1.0], 'x': [np.nan], 'y': [np.nan]})
        with pytest.raises(ValueError) as exc:
            impute_from_training_window(train)
        assert "'x'" in str(exc.value) and "'y'" in str(exc.value)

    def test_the_scaler_would_not_have_caught_it(self):
        """Justifies raising here: the scaler propagates NaN without complaining."""
        from sklearn.preprocessing import StandardScaler

        frame = pd.DataFrame({'a': [1.0, 2.0], 'empty': [np.nan, np.nan]})
        scaled = StandardScaler().fit_transform(frame)
        assert np.isnan(scaled[:, 1]).all(), (
            'if the scaler rejected NaN, raising here would be redundant'
        )

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
        """The scaler does not accept missing values, so the order is load-bearing.

        Checked in the paradigm because that is where the two calls sit next to
        each other. The scaler itself moved to core -- it was written out three
        times identically -- so the marker is the shared call rather than the
        sklearn class.
        """
        source = path.read_text()
        assert source.index('impute_from_training_window') < \
            source.index('scale_from_training_window'), (
            'the scaler does not accept missing values'
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_no_paradigm_rolls_its_own_fill(self, path):
        """The whole file, and the idioms of the three engines.

        The previous version sliced from run_fold_analysis, and _prepare_data
        comes before it in all three files -- so the fill that turned the
        shared helper into a no-op fell outside the examined stretch. The
        forbidden tuple also did not include fill_null, which is how polars
        fills.
        """
        source = path.read_text()
        for forbidden in ('.fillna(', '.fill_null(', '.interpolate(',
                          'SimpleImputer', 'KNNImputer', '.bfill(', '.ffill('):
            assert forbidden not in source, (
                f'{path.parts[-3]} fills missing values outside the shared '
                f'implementation ({forbidden}), so the paradigms can '
                f'preprocess differently -- and the helper becomes a no-op'
            )

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_materialisation_leaves_gaps_for_the_shared_layer(self, paradigm):
        """Behavioural: _prepare_data returns the missing value it received.

        A textual test does not distinguish "does not fill" from "fills in an
        idiom the list does not cover". This one hands over a fold with a gap
        and requires that it reach the shared layer intact.
        """
        import importlib
        import warnings

        warnings.filterwarnings('ignore')
        rng = np.random.default_rng(3)
        n = 16
        # Sorted by (entity, year), which is the canonical form of a fold: in
        # production sql_engine receives it from the view's ORDER BY, and the
        # other two produce it themselves. Handing over rows out of order would
        # make canonical_fold fail before the test reached the gap.
        frame = pd.DataFrame({
            'entity_id': np.repeat(['AAA', 'BBB'], n // 2),
            'year': np.tile(np.arange(2000, 2000 + n // 2), 2),
            'gini': rng.normal(40, 5, n),
            'internet': rng.normal(50, 8, n),
        })
        frame['dropout_rate'] = rng.normal(10, 2, n)
        frame.loc[2, 'gini'] = np.nan          # the gap under test

        module = importlib.import_module(
            f'architectures_ml.{paradigm}.models.hierarchical_model')
        cls = next(getattr(module, name) for name in dir(module)
                   if isinstance(getattr(module, name), type)
                   and hasattr(getattr(module, name), '_prepare_data'))
        instance = cls.__new__(cls)
        instance.target_col = 'dropout_rate'
        instance.available_features = ['gini', 'internet']

        if paradigm == 'sql_engine':
            X, _, _ = instance._prepare_data(frame, ['gini', 'internet'])
        elif paradigm == 'task_graph':
            import dask.dataframe as dd
            X, _, _ = instance._prepare_data(dd.from_pandas(frame, npartitions=1))
        else:
            import polars as pl
            X, _, _ = instance._prepare_data(pl.from_pandas(frame).lazy())

        assert pd.DataFrame(X)['gini'].isna().any(), (
            f'{paradigm}._prepare_data filled the gap, which turns '
            f'impute_from_training_window into a no-op and brings back the '
            f'three implementations that centralisation removed'
        )


class TestTheFoldLevelImputationIsRecorded:
    """The reports were produced on every fold and thrown away.

    How much of each training and evaluation window is fabricated appeared in
    no artifact. Only the collection-stage imputation did, and that is the part
    bounded by the carry limit; the fold-scoped fill is the unbounded one --
    every cell the carry did not reach receives the training-window median.
    """

    @staticmethod
    def _reports(folds=3):
        import numpy as np
        import pandas as pd
        from core.validation import impute_from_training_window

        collected = []
        for fold_id in range(folds):
            train = pd.DataFrame({'a': [1.0, 2.0, np.nan],
                                  'b': [1.0, np.nan, 3.0]})
            test = pd.DataFrame({'a': [np.nan, np.nan], 'b': [1.0, 2.0]})
            _, report = impute_from_training_window(train, test)
            collected.append((fold_id, report))
        return collected

    def test_the_report_counts_cells_per_split(self):
        _, report = self._reports(folds=1)[0]
        counts = report['filled_cells']
        assert counts['train']['by_column'] == {'a': 1, 'b': 1}
        assert counts['apply_0']['by_column'] == {'a': 2}
        assert counts['train']['rows'] == 3

    def test_a_split_with_no_gaps_reports_zero(self):
        """Otherwise the count could be reporting the column list."""
        import numpy as np
        import pandas as pd
        from core.validation import impute_from_training_window

        train = pd.DataFrame({'a': [1.0, np.nan]})
        clean = pd.DataFrame({'a': [3.0, 4.0]})
        _, report = impute_from_training_window(train, clean)
        assert report['filled_cells']['apply_0']['total'] == 0

    def test_it_is_written_next_to_the_fold_artifacts(self, tmp_path,
                                                      monkeypatch):
        import json
        import core.config as config
        from core.models.hierarchical import write_imputation_report

        monkeypatch.setattr(config, 'get_absolute_output_path',
                            lambda relative: str(tmp_path / relative))
        path = write_imputation_report(self._reports(),
                                       architecture='sql_engine')
        payload = json.loads(Path(path).read_text())
        assert payload['architecture'] == 'sql_engine'
        assert set(payload['folds']) == {'0', '1', '2'}

    def test_the_totals_are_summed_across_folds(self, tmp_path, monkeypatch):
        import json
        import core.config as config
        from core.models.hierarchical import write_imputation_report

        monkeypatch.setattr(config, 'get_absolute_output_path',
                            lambda relative: str(tmp_path / relative))
        path = write_imputation_report(self._reports(folds=3),
                                       architecture='task_graph')
        totals = json.loads(Path(path).read_text())['across_folds']
        assert totals['train']['rows'] == 9
        assert totals['train']['total'] == 6
        assert totals['train']['fraction'] == pytest.approx(6 / 9)

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_every_paradigm_accumulates_and_writes(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        assert 'self._imputation_reports = []' in source
        assert 'self._imputation_reports.append((fold_id, imputation_report))' \
            in source
        assert 'shared_write_imputation_report(' in source

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_the_report_is_not_discarded(self, paradigm):
        """It was captured into a name nothing read."""
        import ast as ast_module
        source = (_SRC / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        tree = ast_module.parse(source)
        read = {node.attr for node in ast_module.walk(tree)
                if isinstance(node, ast_module.Attribute)}
        assert '_imputation_reports' in read
