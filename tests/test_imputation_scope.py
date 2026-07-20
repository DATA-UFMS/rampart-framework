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
        assert 'NÃO É APLICADA' in block

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
        """Não pode ocorrer sob janela expansiva + seleção P4; se ocorrer, para.

        As alternativas são piores: constante fabrica um valor que o treino nunca
        observou e torna a feature constante no treino e variável no teste;
        descartar muda o conjunto de features entre folds e entre paradigmas;
        deixar ausente adia a falha para o RidgeCV, porque o StandardScaler
        propaga NaN em silêncio em vez de rejeitar.
        """
        train = pd.DataFrame({'a': [1.0, 2.0], 'empty': [np.nan, np.nan]})
        test = pd.DataFrame({'a': [np.nan], 'empty': [np.nan]})
        with pytest.raises(ValueError, match='nenhuma observação'):
            impute_from_training_window(train, test)

    def test_the_error_names_the_offending_columns(self):
        train = pd.DataFrame({'a': [1.0], 'x': [np.nan], 'y': [np.nan]})
        with pytest.raises(ValueError) as exc:
            impute_from_training_window(train)
        assert "'x'" in str(exc.value) and "'y'" in str(exc.value)

    def test_the_scaler_would_not_have_caught_it(self):
        """Justifica levantar aqui: o scaler propaga NaN sem reclamar."""
        from sklearn.preprocessing import StandardScaler

        frame = pd.DataFrame({'a': [1.0, 2.0], 'empty': [np.nan, np.nan]})
        scaled = StandardScaler().fit_transform(frame)
        assert np.isnan(scaled[:, 1]).all(), (
            'se o scaler rejeitasse NaN, levantar aqui seria redundante'
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
        source = path.read_text()
        assert source.index('impute_from_training_window') < \
            source.index('StandardScaler()'), (
            'the scaler does not accept missing values'
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_no_paradigm_rolls_its_own_fill(self, path):
        """O arquivo inteiro, e os idiomas dos três engines.

        A versão anterior fatiava a partir de run_fold_analysis, e _prepare_data
        vem antes dele nos três arquivos -- então o preenchimento que tornava o
        helper compartilhado um no-op ficava fora do trecho examinado. A tupla
        proibida também não incluía fill_null, que é como o polars preenche.
        """
        source = path.read_text()
        for forbidden in ('.fillna(', '.fill_null(', '.interpolate(',
                          'SimpleImputer', 'KNNImputer', '.bfill(', '.ffill('):
            assert forbidden not in source, (
                f'{path.parts[-3]} preenche ausentes fora da implementação '
                f'compartilhada ({forbidden}), então os paradigmas podem '
                f'preprocessar de forma diferente -- e o helper vira no-op'
            )

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_materialisation_leaves_gaps_for_the_shared_layer(self, paradigm):
        """Comportamental: _prepare_data devolve o ausente que recebeu.

        Um teste textual não distingue "não preenche" de "preenche em um idioma
        que a lista não cobre". Este entrega um fold com lacuna e exige que ela
        chegue intacta à camada compartilhada.
        """
        import importlib
        import warnings

        warnings.filterwarnings('ignore')
        rng = np.random.default_rng(3)
        n = 16
        frame = pd.DataFrame({
            'country_code': ['AAA', 'BBB'] * (n // 2),
            'year': np.repeat(np.arange(2000, 2000 + n // 2), 2),
            'gini': rng.normal(40, 5, n),
            'internet': rng.normal(50, 8, n),
        })
        frame['dropout_rate'] = rng.normal(10, 2, n)
        frame.loc[2, 'gini'] = np.nan          # a lacuna sob teste

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
            f'{paradigm}._prepare_data preencheu a lacuna, o que torna '
            f'impute_from_training_window um no-op e devolve as três '
            f'implementações que a centralização removeu'
        )
