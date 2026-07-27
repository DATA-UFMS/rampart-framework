#!/usr/bin/env python3
"""Unit tests for the framework's core logic.

They validate fundamental algorithms without depending on pre-generated
pipeline outputs or on external data.

What they no longer do is reimplement those algorithms here. Four sections
carried local copies -- symmetric log, fold generator, collinearity filter,
temporal integrity -- and tested the copies. Twenty-eight tests passed with
`src/` deleted, and the copies had already diverged: the local filter did not
sort the features, while the production one does, so `test_order_matters`
asserted a property that production does not have.

Now each section drives the production implementation. Where there are three
(one per paradigm), the three are exercised over the same input and compared
against each other: a divergence here is a divergence in Δ=0.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
import sys
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _quiet(function, *args, **kwargs):
    """The production implementations print progress on every call."""
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


@pytest.fixture(scope='module')
def architectures():
    """A real instance of each paradigm. The constructors do no I/O."""
    from architectures_ml.dataframe_lib.setup import DataFrameLibArchitectureML
    from architectures_ml.sql_engine.setup import SqlEngineArchitectureML
    from architectures_ml.task_graph.setup import TaskGraphArchitectureML

    return {
        'sql_engine': _quiet(SqlEngineArchitectureML),
        'dataframe_lib': _quiet(DataFrameLibArchitectureML),
        'task_graph': _quiet(TaskGraphArchitectureML),
    }


def _native(paradigm, frame):
    """Convert a pandas frame to the type that paradigm consumes."""
    if paradigm == 'dataframe_lib':
        import polars as pl
        return pl.from_pandas(frame)
    if paradigm == 'task_graph':
        import dask.dataframe as dd
        return dd.from_pandas(frame, npartitions=1)
    return frame


def _to_pandas(result):
    if hasattr(result, 'compute'):
        return result.compute()
    if hasattr(result, 'to_pandas'):
        return result.to_pandas()
    return result


# 1. Symmetric log transform, in the three production implementations

def _symmetric_log(x):
    """Mathematical reference: T(x) = sign(x) * ln(|x| + 1)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log(np.abs(x) + 1)


# Covers both signs, zero (its own branch in the three implementations), values
# below and above 1 -- where ln changes sign -- and extreme magnitudes.
PROBE_VALUES = [-1e6, -1000.0, -3.5, -1.0, -0.5, 0.0, 0.5, 1.0, 7.25,
                1000.0, 1e6]


def _transform_via_production(architecture, values, features=('gini_index',),
                              tmp_path=None):
    """Run the paradigm's prepare_features and return the transformed columns."""
    paradigm = architecture.architecture_name
    frame = pd.DataFrame({
        'country_code': ['BRA'] * len(values),
        'year': list(range(2000, 2000 + len(values))),
        architecture.target_column: np.arange(len(values), dtype=float),
    })
    for offset, feature in enumerate(features):
        frame[feature] = np.asarray(values, dtype=float) + offset

    if paradigm == 'sql_engine':
        from collection.sql_engine.connection_manager import (
            DuckDBConnectionManager)
        architecture.conn_manager = DuckDBConnectionManager(
            str(tmp_path / f'{paradigm}.duckdb'))
        connection = architecture.conn_manager.get_connection()
        connection.register('source_frame', frame)
        connection.execute(
            'CREATE OR REPLACE TABLE analytics_wide AS SELECT * FROM source_frame')
        _quiet(architecture.prepare_features, None, list(features))
        return connection.execute(
            'SELECT * FROM vw_selected_features ORDER BY year').df()

    result = _quiet(architecture.prepare_features,
                    _native(paradigm, frame), list(features))
    return _to_pandas(result).sort_values('year').reset_index(drop=True)


class TestSymmetricLogTransform:
    """T(x) = sign(x) * ln(|x| + 1), as each paradigm actually applies it.

    This used to test a local numpy copy. A paradigm that swapped LN for LOG10
    in its SQL expression, or lost the zero branch, passed.
    """

    @pytest.fixture(scope='class')
    def transformed(self, architectures, tmp_path_factory):
        directory = tmp_path_factory.mktemp('transform')
        return {name: _transform_via_production(arch, PROBE_VALUES,
                                                tmp_path=directory)
                for name, arch in architectures.items()}

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'dataframe_lib',
                                          'task_graph'])
    def test_matches_the_reference(self, transformed, paradigm):
        frame = transformed[paradigm]
        assert 'gini_index_log_transform' in frame.columns
        np.testing.assert_allclose(
            frame['gini_index_log_transform'].to_numpy(dtype=float),
            _symmetric_log(frame['gini_index'].to_numpy(dtype=float)),
            rtol=0, atol=1e-12)

    def test_the_three_paradigms_agree(self, transformed):
        """A divergence here is a divergence in Δ=0."""
        columns = {name: frame['gini_index_log_transform'].to_numpy(dtype=float)
                   for name, frame in transformed.items()}
        reference = columns['sql_engine']
        for name, values in columns.items():
            np.testing.assert_allclose(values, reference, rtol=0, atol=0,
                                       err_msg=f'{name} differs from sql_engine')

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'dataframe_lib',
                                          'task_graph'])
    def test_zero_maps_to_zero(self, transformed, paradigm):
        frame = transformed[paradigm]
        at_zero = frame.loc[frame['gini_index'] == 0.0,
                            'gini_index_log_transform']
        assert len(at_zero) == 1
        assert float(at_zero.iloc[0]) == 0.0

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'dataframe_lib',
                                          'task_graph'])
    def test_negatives_mirror_positives(self, transformed, paradigm):
        frame = transformed[paradigm].set_index('gini_index')
        column = frame['gini_index_log_transform']
        for value in (0.5, 1.0, 1000.0, 1e6):
            assert float(column.loc[value]) == pytest.approx(
                -float(column.loc[-value]), abs=1e-12)

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'dataframe_lib',
                                          'task_graph'])
    def test_monotonic_increasing(self, transformed, paradigm):
        frame = transformed[paradigm].sort_values('gini_index')
        column = frame['gini_index_log_transform'].to_numpy(dtype=float)
        assert np.all(np.diff(column) > 0)

    def test_only_the_first_five_features_are_transformed(
            self, architectures, tmp_path_factory):
        """The top-5 limit is declared in all three; no test exercised it."""
        directory = tmp_path_factory.mktemp('top5')
        features = [f'feature_{index}' for index in range(7)]
        for name, architecture in architectures.items():
            frame = _transform_via_production(
                architecture, PROBE_VALUES, features=features,
                tmp_path=directory)
            transformed = sorted(column for column in frame.columns
                                 if column.endswith('_log_transform'))
            assert transformed == sorted(
                f'{feature}_log_transform' for feature in features[:5]), name


# 2. Walk-forward folds, through the production generator

def _generate_folds(start_year, end_year, min_train, val_len, test_len, gap,
                    step=1):
    """Call BaseArchitectureML._generate_walkforward_folds_auto.

    The generator reads everything from `config` and `dataset_config`; this
    function only assembles the two from the parameters and delegates. There
    used to be a copy of the algorithm here, which lacked production's
    truncation by `folds_max`.
    """
    from core.base_architecture import BaseArchitectureML

    class Config:
        temporal_range = (start_year, end_year)
        walk_forward_config = {'min_train': min_train, 'val_len': val_len,
                               'test_len': test_len, 'step': step}
        year_column = 'year'
        entity_column = 'country_code'
        entity_name_column = 'country_name'
        stratification_column = None
        target_source_column = 'source_rate'
        feature_columns = []
        excluded_columns = []

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
        def discover_numeric_columns(self, data): return []

    architecture = _quiet(Probe, 'sql_engine', '/tmp', dataset_config=Config())
    architecture.config = {**architecture.config, 'temporal_gap_years': gap}
    return _quiet(architecture._generate_walkforward_folds_auto)


class TestWalkForwardFolds:
    """Walk-forward temporal fold generation, in the production implementation."""

    @pytest.fixture
    def default_folds(self):
        return _generate_folds(
            start_year=2000, end_year=2023,
            min_train=8, val_len=2, test_len=2, gap=2,
        )

    def test_nine_folds(self, default_folds):
        assert len(default_folds) == 9

    def test_no_temporal_overlap(self, default_folds):
        for fold in default_folds:
            assert fold['train_end'] < fold['val_start']
            assert fold['val_end'] < fold['test_start']

    def test_minimum_gap_respected(self, default_folds):
        for fold in default_folds:
            assert fold['val_start'] - fold['train_end'] - 1 >= 2
            assert fold['test_start'] - fold['val_end'] - 1 >= 2

    def test_expanding_train(self, default_folds):
        """Expanding window: the start is fixed and the end advances."""
        starts = {fold['train_start'] for fold in default_folds}
        assert starts == {2000}
        ends = [fold['train_end'] for fold in default_folds]
        assert ends == sorted(ends)
        assert len(set(ends)) == len(ends)

    def test_first_fold_starts_at_2000(self, default_folds):
        assert default_folds[0]['train_start'] == 2000

    def test_last_fold_ends_at_2023(self, default_folds):
        assert default_folds[-1]['test_end'] == 2023

    def test_val_and_test_length(self, default_folds):
        for fold in default_folds:
            assert fold['val_end'] - fold['val_start'] + 1 == 2
            assert fold['test_end'] - fold['test_start'] + 1 == 2

    def test_impossible_params_halt(self):
        """Production raises; the local copy returned an empty list.

        Divergence found when reconnecting this test. Raising is the right
        behaviour: an empty list travels through the whole pipeline and comes
        out as zero folds evaluated, with nothing flagging the unviable
        configuration.
        """
        with pytest.raises(ValueError, match='No fold could be generated'):
            _generate_folds(2000, 2005, min_train=8, val_len=2, test_len=2,
                            gap=2)

    def test_custom_step(self):
        folds = _generate_folds(2000, 2023, 8, 2, 2, 2, step=2)
        starts = [fold['test_start'] for fold in folds]
        assert all(later - earlier == 2
                   for earlier, later in zip(starts, starts[1:]))

    def test_the_first_fold_train_end_is_the_one_selection_uses(self):
        """P4 reads `_first_fold_train_end`; the folds come from another method.

        Nothing guaranteed the two derived the same window, and feature
        selection happens before any fold is materialised.
        """
        from core.base_architecture import BaseArchitectureML

        class Config:
            temporal_range = (2000, 2023)
            walk_forward_config = {'min_train': 8, 'val_len': 2, 'test_len': 2}
            year_column = 'year'
            entity_column = 'country_code'
            entity_name_column = 'country_name'
            stratification_column = None
            target_source_column = 'source_rate'
            feature_columns = []
            excluded_columns = []

        class Probe(BaseArchitectureML):
            def setup_environment(self): pass
            def load_data(self): pass
            def validate_data(self, data): pass
            def create_target_implementation(self, data): return data
            def _compute_target_statistics(self, data): pass
            def _validate_temporal_folds(self, data, folds): pass
            def save_folds(self, data, folds): pass
            def compute_feature_correlations(self, data, features): return {}
            def apply_collinearity_filter(self, data, features,
                                          threshold=0.8): return features
            def prepare_features(self, data, features): return data
            def discover_numeric_columns(self, data): return []

        architecture = _quiet(Probe, 'sql_engine', '/tmp',
                              dataset_config=Config())
        folds = _quiet(architecture._generate_walkforward_folds_auto)
        assert architecture._first_fold_train_end() == folds[0]['train_end']


# 3. Greedy pairwise collinearity filter, in the three implementations

def _filter_all(architectures, frame, features, threshold=0.8):
    """{paradigm: features kept}, each one by that paradigm's own filter."""
    return {name: list(_quiet(architecture.apply_collinearity_filter,
                              _native(name, frame), list(features), threshold))
            for name, architecture in architectures.items()}


def _frame(columns, rows=100, seed=42):
    """Frame with the row count the filters require in order to correlate."""
    rng = np.random.default_rng(seed)
    base = {name: builder(rng, rows) for name, builder in columns.items()}
    return pd.DataFrame(base)


class TestCollinearityFilter:
    """Greedy pairwise correlation filter, as each paradigm applies it.

    This used to test a local copy that received a ready-made correlation
    matrix and *did not sort* the features. Production sorts, so
    `test_order_matters` asserted a property that production does not have: it
    was the test that was wrong, not the code.
    """

    def test_uncorrelated_features_all_kept(self, architectures):
        frame = _frame({name: (lambda rng, n: rng.normal(size=n))
                        for name in ('a', 'b', 'c')})
        for name, kept in _filter_all(architectures, frame,
                                      ['a', 'b', 'c']).items():
            assert sorted(kept) == ['a', 'b', 'c'], name

    def test_perfectly_correlated_pair(self, architectures):
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        frame = pd.DataFrame({'a': x, 'b': x + 0.001 * rng.normal(size=100),
                              'c': rng.normal(size=100)})
        for name, kept in _filter_all(architectures, frame,
                                      ['a', 'b', 'c']).items():
            assert 'a' in kept, name
            assert 'b' not in kept, name
            assert 'c' in kept, name

    def test_single_feature(self, architectures):
        frame = _frame({'a': lambda rng, n: rng.normal(size=n)})
        for name, kept in _filter_all(architectures, frame, ['a']).items():
            assert kept == ['a'], name

    def test_empty_features(self, architectures):
        frame = _frame({'a': lambda rng, n: rng.normal(size=n)})
        for name, kept in _filter_all(architectures, frame, []).items():
            assert kept == [], name

    def test_the_result_does_not_depend_on_input_order(self, architectures):
        """Production sorts before walking; the local copy did not sort.

        Without sorting, which of the two collinear features survives depends
        on the order in which correlation-based selection returned them -- and
        that order could differ between paradigms, which is exactly what Δ=0
        does not allow.
        """
        rng = np.random.default_rng(42)
        x = rng.normal(size=100)
        frame = pd.DataFrame({'a': x, 'b': x + 0.001 * rng.normal(size=100)})
        forward = _filter_all(architectures, frame, ['a', 'b'])
        reversed_order = _filter_all(architectures, frame, ['b', 'a'])
        for name in forward:
            assert forward[name] == reversed_order[name] == ['a'], name

    def test_the_three_paradigms_keep_the_same_features(self, architectures):
        """Chain a-b 0.9, b-c 0.9, a-c 0.3: keeps a and c, discards b."""
        rng = np.random.default_rng(7)
        a = rng.normal(size=200)
        c = 0.3 * a + np.sqrt(1 - 0.09) * rng.normal(size=200)
        b = 0.9 * a + np.sqrt(1 - 0.81) * rng.normal(size=200)
        frame = pd.DataFrame({'a': a, 'b': b, 'c': c})
        assert abs(np.corrcoef(a, b)[0, 1]) > 0.8
        assert abs(np.corrcoef(a, c)[0, 1]) < 0.8

        results = _filter_all(architectures, frame, ['a', 'b', 'c'])
        assert len(set(map(tuple, results.values()))) == 1, results
        for name, kept in results.items():
            assert kept == ['a', 'c'], name

    def test_the_threshold_is_strict(self, architectures):
        """At the exact threshold the feature drops: the comparison is `<`, not
        `<=`.

        An identical copy gives correlation exactly 1.0 in the three engines,
        so the threshold 1.0 is the only point at which the two operators
        disagree. With offsets of 1e-12 around an arbitrary correlation, `<`
        and `<=` decide the same and the test does not tell the two apart.
        """
        rng = np.random.default_rng(11)
        a = rng.normal(size=200)
        frame = pd.DataFrame({'a': a, 'b': a.copy()})
        for name, kept in _filter_all(architectures, frame, ['a', 'b'],
                                      threshold=1.0).items():
            assert kept == ['a'], (
                f'{name}: kept an identical copy at the exact threshold'
            )

    def test_below_the_threshold_the_feature_survives(self, architectures):
        """Counterpart: without it, always rejecting would pass the test above."""
        rng = np.random.default_rng(11)
        frame = pd.DataFrame({'a': rng.normal(size=200),
                              'b': rng.normal(size=200)})
        for name, kept in _filter_all(architectures, frame, ['a', 'b'],
                                      threshold=1.0).items():
            assert kept == ['a', 'b'], name


# 4. Scientific configuration validation

class TestScientificConfig:
    """Tests verifying that the scientific configuration has the required keys and valid values."""

    @pytest.fixture
    def config(self):
        from core.scientific_config import SCIENTIFIC_CONFIG
        return SCIENTIFIC_CONFIG

    def test_required_keys_exist(self, config):
        required = [
            'random_seed', 'collinearity_threshold', 'temporal_gap_years',
            'feature_transform', 'bootstrap_iters',
            'temporal_range_start', 'temporal_range_end',
            'folds_min_train_years', 'folds_val_len_years', 'folds_test_len_years',
            'sesoi_r2', 'sesoi_wape', 'sesoi_mase',
        ]
        for key in required:
            assert key in config, f"Missing key: {key}"

    def test_seed_is_int(self, config):
        assert isinstance(config['random_seed'], int)

    def test_collinearity_threshold_range(self, config):
        t = config['collinearity_threshold']
        assert 0.5 <= t <= 1.0, f"Threshold {t} outside reasonable range [0.5, 1.0]"

    def test_temporal_range_valid(self, config):
        assert config['temporal_range_start'] < config['temporal_range_end']
        span = config['temporal_range_end'] - config['temporal_range_start'] + 1
        min_required = (config['folds_min_train_years']
                       + config['folds_val_len_years']
                       + config['folds_test_len_years']
                       + 2 * config['temporal_gap_years'])
        assert span >= min_required, f"Temporal span {span} < min required {min_required}"

    def test_transform_is_symmetric_log(self, config):
        assert config['feature_transform'] == 'symmetric_log'

    def test_sesoi_positive(self, config):
        for key in ['sesoi_r2', 'sesoi_wape', 'sesoi_mase']:
            assert config[key] > 0, f"{key} must be positive"


# 5. Country strata completeness

class TestCountryStrata:
    """Tests for the geographic stratification configuration."""

    @pytest.fixture(autouse=True)
    def _load_strata(self):
        from core.config import COUNTRY_STRATA
        self.STRATA = COUNTRY_STRATA

    def test_total_32_countries(self):
        all_countries = [c for s in self.STRATA.values() for c in s]
        assert len(all_countries) == 32

    def test_no_duplicate_countries(self):
        all_countries = [c for s in self.STRATA.values() for c in s]
        assert len(all_countries) == len(set(all_countries))

    def test_all_iso2_codes(self):
        for stratum, countries in self.STRATA.items():
            for code in countries:
                assert len(code) == 2 and code.isalpha() and code.isupper(), \
                    f"Invalid ISO-2 code '{code}' in {stratum}"


# 6. Temporal integrity validation

class TestTemporalIntegrity:
    """Anti-leak temporal validation, through the production TemporalValidator.

    The local copy returned a boolean and checked three conditions. The
    production validator also returns the list of errors, checks mandatory
    fields, and knows about the embargo -- none of that was exercised.
    """

    @staticmethod
    def _fold(train, val, test):
        return {'train_start': train[0], 'train_end': train[1],
                'val_start': val[0], 'val_end': val[1],
                'test_start': test[0], 'test_end': test[1]}

    @staticmethod
    def _validate(train, val, test, min_gap=2, embargo=0):
        from core.validation import TemporalValidator
        validator = TemporalValidator(min_gap_years=min_gap,
                                      embargo_years=embargo)
        return validator.validate_fold_integrity(
            TestTemporalIntegrity._fold(train, val, test))

    def test_valid_split(self):
        valid, errors = self._validate((2000, 2007), (2010, 2011),
                                       (2014, 2015))
        assert valid, errors

    def test_insufficient_train_val_gap(self):
        valid, errors = self._validate((2000, 2008), (2010, 2011),
                                       (2014, 2015))
        assert not valid
        assert any('train-val' in error for error in errors), errors

    def test_insufficient_val_test_gap(self):
        valid, errors = self._validate((2000, 2007), (2010, 2011),
                                       (2013, 2014))
        assert not valid
        assert any('val-test' in error for error in errors), errors

    def test_overlapping_train_val(self):
        valid, errors = self._validate((2000, 2010), (2010, 2011),
                                       (2014, 2015))
        assert not valid
        assert any('Train-val overlap' in error for error in errors), errors

    def test_reversed_order(self):
        valid, errors = self._validate((2014, 2015), (2010, 2011),
                                       (2000, 2007))
        assert not valid

    def test_a_missing_field_is_reported(self):
        """The local copy raised KeyError; production names the field."""
        from core.validation import TemporalValidator
        fold = self._fold((2000, 2007), (2010, 2011), (2014, 2015))
        del fold['val_start']
        valid, errors = TemporalValidator(min_gap_years=2) \
            .validate_fold_integrity(fold)
        assert not valid
        assert any('val_start' in error for error in errors), errors

    def test_the_embargo_consumes_the_gap(self):
        """A parameter the local copy did not have.

        With gap 2 and embargo 3, the fold that passes without an embargo stops
        passing.
        """
        assert self._validate((2000, 2007), (2010, 2011), (2014, 2015),
                              embargo=0)[0]
        valid, errors = self._validate((2000, 2007), (2010, 2011),
                                       (2014, 2015), embargo=3)
        assert not valid
        assert any('embargo violated' in error for error in errors), errors

    def test_all_default_folds_valid(self):
        """The folds production generates pass production's own validator."""
        folds = _generate_folds(2000, 2023, 8, 2, 2, 2)
        assert folds
        for fold in folds:
            valid, errors = self._validate(
                (fold['train_start'], fold['train_end']),
                (fold['val_start'], fold['val_end']),
                (fold['test_start'], fold['test_end']))
            assert valid, f"Fold {fold['fold_id']}: {errors}"

    def test_the_generator_and_the_validator_agree_on_the_gap(self):
        """Generating with gap 2 and validating with gap 3 has to fail.

        Without this, the two sides could read different parameters and the
        test above would pass because both were equally wrong.
        """
        folds = _generate_folds(2000, 2023, 8, 2, 2, 2)
        results = [self._validate(
            (fold['train_start'], fold['train_end']),
            (fold['val_start'], fold['val_end']),
            (fold['test_start'], fold['test_end']), min_gap=3)[0]
            for fold in folds]
        assert not any(results)


# 7. Import tests (via conftest.py PYTHONPATH)

class TestImports:
    """Verifies that the core modules are importable via conftest.py's path configuration."""

    def test_import_scientific_config(self):
        from core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED
        assert RANDOM_SEED == 42
        assert isinstance(SCIENTIFIC_CONFIG, dict)

    def test_import_config(self):
        from core.config import COUNTRY_STRATA, LATIN_AMERICA_COUNTRIES
        assert len(LATIN_AMERICA_COUNTRIES) == 32
        assert 'large_economies' in COUNTRY_STRATA

    def test_import_get_project_root(self):
        from core.config import get_project_root
        root = Path(get_project_root())
        assert root.is_dir()
        assert (root / "src" / "core").is_dir()


# 8. Anti-leakage: P4 (temporal scope) and extended P3 (proxy detection)

class TestAntiLeakageP4:
    """Validates that feature selection uses only data from the training period."""

    def test_first_fold_train_end_calculation(self):
        """Checks the first fold's train_end computation from the config."""
        from core.scientific_config import SCIENTIFIC_CONFIG
        cfg = SCIENTIFIC_CONFIG
        start = cfg.get('temporal_range_start', 2000)
        min_train = cfg.get('folds_min_train_years', 8)
        val_len = cfg.get('folds_val_len_years', 2)
        gap = cfg.get('temporal_gap_years', 2)

        test_start_min = start + min_train + val_len + 2 * gap
        val_end = test_start_min - gap - 1
        val_start = val_end - val_len + 1
        expected_train_end = val_start - gap - 1

        assert expected_train_end == 2007

    def test_proxy_threshold_in_config(self):
        """proxy_correlation_threshold must exist in the scientific config."""
        from core.scientific_config import SCIENTIFIC_CONFIG
        threshold = SCIENTIFIC_CONFIG.get('proxy_correlation_threshold')
        assert threshold is not None
        assert 0.5 < threshold <= 1.0, f"Threshold outside the reasonable range: {threshold}"



# 9. Anti-leakage: P5 (preprocessing scope) and HPO

class TestPreprocessingIsolation:
    """P5 (preprocessing scope) in the code that runs, not in a library.

    The three previous tests here exercised sklearn's StandardScaler, pandas'
    .fillna() and Python's max() over literal dicts. They passed with the whole
    framework deleted, because none of them mentioned it. One of them asserted
    that max({1: 0.85, 10: 0.80}, key=...) == 1.

    The imputation contract is covered in test_imputation_scope.py, including
    the property that matters -- changing the test window does not move the
    statistic. What remains here are the two links that were missing: the
    scaler and hyperparameter selection, verified in the body of each
    paradigm's run_fold_analysis.
    """

    MODELS = sorted((Path(__file__).resolve().parents[1] / 'src'
                     / 'architectures_ml').glob('*/models/hierarchical_model.py'))

    def test_the_models_were_found(self):
        assert len(self.MODELS) == 3

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_scaler_is_fitted_on_training_data_only(self, path):
        """fit_transform on training; transform on validation and test."""
        import ast as _ast
        tree = _ast.parse(path.read_text())
        fold = next(n for n in _ast.walk(tree)
                    if isinstance(n, _ast.FunctionDef)
                    and n.name == 'run_fold_analysis')
        fitted_on = [n.args[0].id for n in _ast.walk(fold)
                     if isinstance(n, _ast.Call)
                     and getattr(n.func, 'attr', None) == 'fit_transform'
                     and n.args and hasattr(n.args[0], 'id')]
        assert fitted_on == ['X_train'], (
            f'{path.parts[-3]}: scaler fitted on {fitted_on}, and not only on '
            f'the training set'
        )
        transformed = [n.args[0].id for n in _ast.walk(fold)
                       if isinstance(n, _ast.Call)
                       and getattr(n.func, 'attr', None) == 'transform'
                       and n.args and hasattr(n.args[0], 'id')]
        assert 'X_test' in transformed and 'X_val' in transformed

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_no_fit_touches_the_test_window(self, path):
        import ast as _ast
        tree = _ast.parse(path.read_text())
        fold = next(n for n in _ast.walk(tree)
                    if isinstance(n, _ast.FunctionDef)
                    and n.name == 'run_fold_analysis')
        for call in _ast.walk(fold):
            if isinstance(call, _ast.Call) and \
                    getattr(call.func, 'attr', None) in ('fit', 'fit_transform'):
                names = [a.id for a in call.args if hasattr(a, 'id')]
                assert 'X_test' not in names and 'y_test' not in names, (
                    f'{path.parts[-3]}: fit over the test window'
                )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_hyperparameters_are_selected_on_validation(self, path):
        """The search compares validation r2; the test set is evaluated later."""
        source = path.read_text()
        fold_start = source.index('def run_fold_analysis')
        body = source[fold_start:]
        selection = body.index('best_val_r2')
        assert 'X_val_scaled' in body[:selection + 2000], (
            f'{path.parts[-3]}: the selection does not consult the validation '
            f'window'
        )
        assert "best_val_r2 = -1e9" in body, 'no initialisation of the search'

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_selected_parameters_are_applied_to_the_test_window(self, path):
        source = path.read_text()
        body = source[source.index('def run_fold_analysis'):]
        assert 'best_shrink' in body and 'best_params' in body, (
            f'{path.parts[-3]}: chosen parameters not reused'
        )



class TestTheEmbargoDocstringMatchesTheCode:
    """The docstring described removing observations; the code checks a gap.

    López de Prado (2018) defines the embargo as dropping the training
    observations adjacent to each split boundary. Nothing here drops anything:
    the validator asserts that the declared gap covers the declared embargo and
    fails the fold otherwise. On this panel the two select the same training
    set, which is why the check suffices -- but a reader taking the docstring
    literally would look for an exclusion that does not exist.
    """

    @staticmethod
    def _docstring():
        from core.validation import TemporalValidator
        return TemporalValidator.__doc__

    def test_it_does_not_claim_observations_are_excluded(self):
        text = self._docstring()
        assert 'are excluded from training' not in text, (
            'the code excludes nothing; it verifies that the gap covers the '
            'embargo'
        )

    def test_it_says_what_the_check_actually_does(self):
        text = self._docstring()
        assert ('checks that the declared gap' in text
                or 'fails the fold when it does not' in text)

    def test_no_code_path_drops_rows_for_the_embargo(self):
        """Reproduced: the fold that passes with embargo 0 fails with embargo 3,
        and no row count changes anywhere -- only the verdict does."""
        from core.validation import TemporalValidator

        fold = {'train_start': 2000, 'train_end': 2007,
                'val_start': 2010, 'val_end': 2011,
                'test_start': 2014, 'test_end': 2015}
        before = dict(fold)
        valid_without, _ = TemporalValidator(
            min_gap_years=2, embargo_years=0).validate_fold_integrity(fold)
        valid_with, errors = TemporalValidator(
            min_gap_years=2, embargo_years=3).validate_fold_integrity(fold)
        assert valid_without and not valid_with
        assert any('embargo violated' in error for error in errors)
        assert fold == before, 'the validator mutated the fold'


class TestTheFoldCountFormulaIsDerived:
    """The comment ended in "8... +1 = 9", which reads as an ad-hoc correction.

    The count is of test-window start points, so it is the size of the closed
    range [test_start_min, test_start_max]. A closed range whose endpoints
    coincide holds one element, not zero -- the "+1" is the range size, not a
    fudge.
    """

    def test_the_comment_derives_the_endpoints(self):
        source = (Path(_SRC).parent / 'src' / 'core'
                  / 'scientific_config.py').read_text()
        assert 'test_start_min' in source and 'test_start_max' in source
        assert '8... +1 = 9' not in source

    def test_the_formula_in_the_comment_gives_the_generated_count(self):
        """A comment that disagrees with the generator is worse than none."""
        start, end = 2000, 2023
        min_train, val, test, gap, step = 8, 2, 2, 2, 1
        test_start_min = start + min_train + val + 2 * gap
        test_start_max = end - test + 1
        predicted = (test_start_max - test_start_min) // step + 1

        folds = _generate_folds(start, end, min_train, val, test, gap,
                                step=step)
        assert predicted == len(folds) == 9


class TestTheFoldMetadataDoesNotOverclaim:
    """`fit_to_test_gap` measures parameter estimation, not information.

    The field recorded six years between the last observation used and the
    first evaluated, and the comment presented that as a safety margin over the
    two years P2 requires. At prediction time a test row at 2014 carries
    dropout_rate_lag_3 -- the target at 2011, the last validation year -- and
    the naive baseline reads history up to the test year minus the gap. The
    information horizon is two years, not six.

    Nothing here is leakage: 2011 precedes 2014 and using past target values is
    the task. What was wrong is the claim, in a field that reaches the
    published fold artifacts.
    """

    @staticmethod
    def _folds():
        return _generate_folds(2000, 2023, 8, 2, 2, 2)

    def test_both_separations_are_recorded(self):
        for fold in self._folds():
            assert 'fit_to_test_gap' in fold
            assert 'information_horizon_years' in fold

    def test_the_information_horizon_is_the_smallest_lag(self):
        from core.base_architecture import BaseArchitectureML
        expected = min(BaseArchitectureML.TARGET_LAG_ORDERS)
        for fold in self._folds():
            assert fold['information_horizon_years'] == expected

    def test_the_horizon_is_smaller_than_the_fitting_gap(self):
        """If they coincided, recording both would prove nothing."""
        for fold in self._folds():
            assert fold['information_horizon_years'] < fold['fit_to_test_gap']

    def test_the_horizon_reaches_into_the_evaluation_side_of_the_gap(self):
        """Concretely: the most recent value a test row consults."""
        fold = self._folds()[0]
        consulted = fold['test_start'] - fold['information_horizon_years']
        assert consulted > fold['train_end'], (
            'the claim that nothing after train_end is consulted is what the '
            'old field implied'
        )

    def test_every_paradigm_builds_exactly_the_declared_lags(self):
        """Three implementations building different lags would break Delta=0."""
        import re
        from pathlib import Path
        from core.base_architecture import BaseArchitectureML
        from core.paradigm_registry import discover_paradigms

        declared = {f'lag_{order}'
                    for order in BaseArchitectureML.TARGET_LAG_ORDERS}
        root = Path(_SRC).parent / 'src' / 'architectures_ml'
        for paradigm in sorted(discover_paradigms()):
            source = (root / paradigm / 'setup.py').read_text()
            built = set(re.findall(r'lag_(\d+)', source))
            assert {f'lag_{order}' for order in built} == declared, (
                f'{paradigm} builds lag orders {sorted(built)} against the '
                f'declared {sorted(BaseArchitectureML.TARGET_LAG_ORDERS)}'
            )


class TestTheWarmupCountIsHonoured:
    """`--warmup 0` fell through to the configured default.

    `warmup_runs or default` treats zero as absent, so asking for no warmup ran
    the configured number anyway -- and the operator had no signal that the
    request was ignored.
    """

    def test_zero_is_respected(self):
        source = (Path(_SRC).parent / 'src' / 'benchmarking'
                  / 'architectural_benchmark.py').read_text()
        assert 'warmup_runs or int(' not in source
        assert 'if warmup_runs is None else int(warmup_runs)' in source

    def test_none_still_takes_the_configured_default(self):
        from core.config import BENCHMARK_CONFIG
        for requested, expected in ((None, int(BENCHMARK_CONFIG['warmup_runs'])),
                                    (0, 0), (3, 3)):
            resolved = (int(BENCHMARK_CONFIG.get('warmup_runs', 1))
                        if requested is None else int(requested))
            assert resolved == expected


class TestTheBestBaselineIgnoresUndefinedScores:
    """`max` with NaN returns whatever came first.

    Comparisons against NaN are False, so a first entry with an undefined
    validation R2 was elected best, and best_test_r2 and generalization_gap
    followed from it. The choice depended on dict insertion order rather than
    on performance.
    """

    @staticmethod
    def _select(scores):
        import sys
        from pathlib import Path
        module_root = Path(_SRC).parent / 'src'
        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
        from architectures_ml.sql_engine.models.baseline_analysis import (
            _best_by_val_r2)
        return _best_by_val_r2({name: {'val_r2': value, 'test_r2': 0.0}
                                for name, value in scores.items()})

    def test_a_leading_nan_does_not_win(self):
        name, value = self._select({'global_mean': float('nan'),
                                    'linear_trend': 0.55,
                                    'naive_with_lag': 0.31})
        assert name == 'linear_trend' and value == 0.55

    def test_a_trailing_nan_does_not_win(self):
        name, _ = self._select({'global_mean': 0.10,
                                'linear_trend': float('nan'),
                                'naive_with_lag': 0.55})
        assert name == 'naive_with_lag'

    def test_all_undefined_halts(self):
        with pytest.raises(ValueError, match='no best baseline to report'):
            self._select({'global_mean': float('nan'),
                          'linear_trend': float('nan')})

    def test_the_plain_max_would_have_failed_these(self):
        """Pins the defect, so the tests above cannot pass vacuously."""
        scores = [('global_mean', float('nan')), ('linear_trend', 0.55)]
        assert max(scores, key=lambda pair: pair[1])[0] == 'global_mean'

    def test_all_three_paradigms_use_the_helper(self):
        from pathlib import Path
        from core.paradigm_registry import discover_paradigms
        root = Path(_SRC).parent / 'src' / 'architectures_ml'
        for paradigm in sorted(discover_paradigms()):
            source = (root / paradigm / 'models'
                      / 'baseline_analysis.py').read_text()
            assert '_best_by_val_r2(fold_results)' in source, paradigm
            assert 'max(val_scores' not in source, paradigm
