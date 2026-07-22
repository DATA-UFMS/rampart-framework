#!/usr/bin/env python3
"""How far a temporal fill may carry an observation, and what the artifact says.

Two defects, both in the part of the pipeline a reviewer inspects to judge how
much of the panel is real.

The reach was the sum of chained steps rather than the number written down.
A lag-1 fill ran first; then a ``ffill(limit=3)`` over the already-filled
series, so the value the lag had just written became the anchor for three more;
then, for unemployment, a three-year rolling mean that averaged imputed cells
as though they were observations. Declared limit: three years. Measured reach
from a single observation: seven.

And ``target_coverage.json`` reported ``observed_fraction`` computed after the
imputation, so every filled cell counted as observed and the fraction came out
near 1.0 by construction. On a synthetic panel the same column reads 0.579
observed against the 0.794 the old field would have printed.
"""

import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from collection.raw_data_collector import (CARRY_LIMIT_YEARS,
                                           LOW_FREQUENCY_CARRY_LIMIT_YEARS,
                                           LOW_FREQUENCY_COLUMNS,
                                           RawDataCollector,
                                           _years_since_observed)

TARGET = 'lower_secondary_completion_rate'


def _limit_for(column):
    return (LOW_FREQUENCY_CARRY_LIMIT_YEARS if column in LOW_FREQUENCY_COLUMNS
            else CARRY_LIMIT_YEARS)


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings('ignore')
        return function(*args, **kwargs)


class TestYearsSinceObserved:

    def test_zero_where_observed(self):
        series = pd.Series([1.0, np.nan, 2.0, np.nan])
        assert list(_years_since_observed(series)) == [0.0, 1.0, 0.0, 1.0]

    def test_counts_up_across_a_run_of_gaps(self):
        series = pd.Series([1.0, np.nan, np.nan, np.nan])
        assert list(_years_since_observed(series)) == [0.0, 1.0, 2.0, 3.0]

    def test_undefined_before_the_first_observation(self):
        """Nothing to carry forward from, so no distance exists."""
        series = pd.Series([np.nan, np.nan, 1.0])
        gaps = _years_since_observed(series)
        assert gaps.isna().iloc[:2].all()
        assert gaps.iloc[2] == 0.0

    def test_it_restarts_after_a_later_observation(self):
        series = pd.Series([1.0, np.nan, np.nan, 5.0, np.nan])
        assert list(_years_since_observed(series)) == [0.0, 1.0, 2.0, 0.0, 1.0]


class TestTheReachIsTheDeclaredLimit:
    """One observation, then a long gap, run through the collector."""

    @staticmethod
    def _reach(column, years=14):
        import tempfile
        collector = _quiet(RawDataCollector)
        collector.output_dir = tempfile.mkdtemp()
        frame = pd.DataFrame({
            'country_code': ['BRA'] * years,
            'country_name': ['Brazil'] * years,
            'year': list(range(2000, 2000 + years)),
            column: [10.0] + [np.nan] * (years - 1),
            TARGET: np.linspace(60, 80, years),
        })
        imputed = _quiet(collector.apply_conservative_imputation, frame)
        return int(imputed[column].notna().sum()) - 1

    @pytest.mark.parametrize('column', ['gini_index',
                                        'gdp_per_capita_constant_2015',
                                        'unemployment_total'])
    def test_reach_equals_the_limit(self, column):
        assert self._reach(column) == _limit_for(column), column

    def test_the_low_frequency_columns_reach_further(self):
        """Otherwise one limit would do and the distinction proves nothing."""
        assert LOW_FREQUENCY_CARRY_LIMIT_YEARS > CARRY_LIMIT_YEARS

    def test_unemployment_no_longer_reaches_seven(self):
        """The chained steps carried it to seven years from one observation."""
        assert self._reach('unemployment_total') < 7

    def test_a_long_gap_is_not_bridged(self):
        """Beyond the limit the cell stays missing for the fold-scoped layer."""
        collector = _quiet(RawDataCollector)
        import tempfile
        collector.output_dir = tempfile.mkdtemp()
        frame = pd.DataFrame({
            'country_code': ['BRA'] * 12,
            'country_name': ['Brazil'] * 12,
            'year': list(range(2000, 2012)),
            'gini_index': [10.0] + [np.nan] * 11,
            TARGET: np.linspace(60, 80, 12),
        })
        imputed = _quiet(collector.apply_conservative_imputation, frame)
        assert imputed['gini_index'].isna().any()


class TestNothingCrossesEntities:
    """The property that makes the fill P5-safe, kept under the rewrite."""

    def test_one_entity_does_not_fill_another(self):
        collector = _quiet(RawDataCollector)
        import tempfile
        collector.output_dir = tempfile.mkdtemp()
        years = list(range(2000, 2008))
        frame = pd.DataFrame({
            'country_code': ['AAA'] * len(years) + ['BBB'] * len(years),
            'country_name': ['A'] * len(years) + ['B'] * len(years),
            'year': years * 2,
            'gini_index': [40.0] * len(years) + [np.nan] * len(years),
            TARGET: list(np.linspace(60, 80, len(years))) * 2,
        })
        imputed = _quiet(collector.apply_conservative_imputation, frame)
        second = imputed[imputed['country_code'] == 'BBB']
        assert second['gini_index'].isna().all(), (
            'BBB has no observation of its own; anything filled there came '
            'from another entity'
        )

    def test_nothing_is_filled_backwards(self):
        """A later observation must not reach an earlier gap."""
        collector = _quiet(RawDataCollector)
        import tempfile
        collector.output_dir = tempfile.mkdtemp()
        frame = pd.DataFrame({
            'country_code': ['BRA'] * 6,
            'country_name': ['Brazil'] * 6,
            'year': list(range(2000, 2006)),
            'gini_index': [np.nan, np.nan, np.nan, 40.0, np.nan, np.nan],
            TARGET: np.linspace(60, 80, 6),
        })
        imputed = _quiet(collector.apply_conservative_imputation, frame)
        ordered = imputed.sort_values('year')['gini_index'].tolist()
        assert all(pd.isna(value) for value in ordered[:3])


class TestTheCoverageArtifactMeasuresTheInput:

    @pytest.fixture(scope='class')
    def coverage(self):
        import tempfile
        rng = np.random.default_rng(4)
        rows = [{'country_code': entity, 'country_name': entity, 'year': year,
                 'gini_index': rng.normal(40, 5) if rng.random() > 0.45
                 else np.nan,
                 'unemployment_total': rng.normal(8, 2) if rng.random() > 0.6
                 else np.nan,
                 TARGET: rng.normal(70, 10) if rng.random() > 0.2 else np.nan}
                for entity in [f'C{index:02d}' for index in range(8)]
                for year in range(2000, 2016)]
        collector = _quiet(RawDataCollector)
        directory = tempfile.mkdtemp()
        collector.output_dir = directory
        imputed = _quiet(collector.apply_conservative_imputation,
                         pd.DataFrame(rows))
        payload = json.loads(
            (Path(directory) / 'target_coverage.json').read_text())
        return payload, pd.DataFrame(rows), imputed

    def test_an_imputed_cell_is_not_counted_as_observed(self, coverage):
        payload, raw, imputed = coverage
        column = 'gini_index'
        assert payload['imputed_fraction'][column] > 0, (
            'nothing was imputed, so the distinction cannot be tested here'
        )
        assert payload['observed_fraction'][column] < \
            float(imputed[column].notna().mean()), (
            'the observed fraction equals the post-imputation coverage, which '
            'is what counted filled cells as real'
        )

    def test_observed_plus_imputed_is_the_post_imputation_coverage(self,
                                                                   coverage):
        payload, _, imputed = coverage
        for column in ('gini_index', 'unemployment_total'):
            total = (payload['observed_fraction'][column]
                     + payload['imputed_fraction'][column])
            assert total == pytest.approx(
                float(imputed[column].notna().mean()), abs=1e-9), column

    def test_the_observed_fraction_matches_the_input_panel(self, coverage):
        payload, raw, imputed = coverage
        for column in ('gini_index', 'unemployment_total'):
            expected = float(
                raw.loc[imputed.index, column].notna().mean())
            assert payload['observed_fraction'][column] == pytest.approx(
                expected, abs=1e-9), column

    def test_the_carry_limit_is_recorded_per_column(self, coverage):
        payload, _, _ = coverage
        for column, limit in payload['carry_limit_years'].items():
            assert limit == _limit_for(column), column

    def test_the_target_column_is_fully_observed(self, coverage):
        """Rows without a target are removed, not filled."""
        payload, _, _ = coverage
        assert payload['observed_fraction'][TARGET] == pytest.approx(1.0)
        assert payload['imputed_fraction'][TARGET] == pytest.approx(0.0)
        assert payload['rows_removed_missing_target'] > 0


class TestTheCarriedValue:
    """What gets written, not only how far it reaches.

    For unemployment the carried value is the mean of the previous observations
    inside the window, which is the point of the step: smoothing cycles rather
    than repeating the last reading. Replacing it with a plain forward fill
    reaches exactly as far, so a test on the reach alone cannot tell them
    apart.
    """

    @staticmethod
    def _imputed(column, values):
        import tempfile
        collector = _quiet(RawDataCollector)
        collector.output_dir = tempfile.mkdtemp()
        frame = pd.DataFrame({
            'country_code': ['BRA'] * len(values),
            'country_name': ['Brazil'] * len(values),
            'year': list(range(2000, 2000 + len(values))),
            column: values,
            TARGET: np.linspace(60, 80, len(values)),
        })
        result = _quiet(collector.apply_conservative_imputation, frame)
        return result.sort_values('year')[column].tolist()

    def test_unemployment_carries_the_mean_of_the_window(self):
        observed = [4.0, 10.0, 16.0]
        filled = self._imputed('unemployment_total',
                               observed + [np.nan] * 4)
        assert filled[3] == pytest.approx(np.mean(observed)), (
            f'carried {filled[3]}, expected the window mean '
            f'{np.mean(observed)}; a plain forward fill would write '
            f'{observed[-1]}'
        )

    def test_it_is_not_the_last_observation(self):
        """Pins the difference the reach test cannot see."""
        observed = [4.0, 10.0, 16.0]
        filled = self._imputed('unemployment_total',
                               observed + [np.nan] * 4)
        assert filled[3] != pytest.approx(observed[-1])

    def test_a_regular_column_carries_the_last_observation(self):
        """The smoothing applies to unemployment only; the rest repeat."""
        filled = self._imputed('gini_index', [4.0, 10.0, 16.0, np.nan, np.nan])
        assert filled[3] == pytest.approx(16.0)

    def test_the_window_follows_the_declared_limit(self, monkeypatch):
        """A window that merely coincides with the limit is a coincidence.

        With the limit lowered, both the reach and the set of observations
        averaged must follow it.
        """
        import collection.raw_data_collector as module
        monkeypatch.setattr(module, 'LOW_FREQUENCY_CARRY_LIMIT_YEARS', 2)
        observed = [4.0, 10.0, 16.0]
        filled = self._imputed('unemployment_total',
                               observed + [np.nan] * 4)
        assert filled[3] == pytest.approx(np.mean(observed[-2:])), (
            'the averaged window did not follow the limit'
        )
        assert int(np.sum(~pd.isna(filled))) - len(observed) == 2, (
            'the reach did not follow the limit'
        )


class TestTheDistanceBoundStandsAlone:
    """The bound must hold whatever the source offers.

    Both sources in use are incidentally bounded, so removing the distance
    check changes nothing through the collector. Exercised directly against an
    unbounded source, it is the only thing stopping a carry from crossing an
    arbitrary gap -- and the source has already changed once.
    """

    @staticmethod
    def _series():
        observed = pd.Series([10.0] + [np.nan] * 9)
        unbounded = observed.ffill()          # no limit at all
        gap = _years_since_observed(observed)
        return observed, unbounded, gap

    def test_an_unbounded_source_is_still_bounded(self):
        from collection.raw_data_collector import _fillable_cells
        observed, unbounded, gap = self._series()
        assert unbounded.notna().all(), (
            'the source is not unbounded, so this proves nothing'
        )
        selected = _fillable_cells(observed, unbounded, gap, limit=3)
        assert list(np.flatnonzero(selected)) == [1, 2, 3]

    def test_the_limit_is_inclusive(self):
        from collection.raw_data_collector import _fillable_cells
        observed, unbounded, gap = self._series()
        assert bool(_fillable_cells(observed, unbounded, gap, limit=1).iloc[1])

    def test_an_observed_cell_is_never_refilled(self):
        from collection.raw_data_collector import _fillable_cells
        observed = pd.Series([10.0, 20.0, np.nan])
        gap = _years_since_observed(observed)
        selected = _fillable_cells(observed, observed.ffill(), gap, limit=3)
        assert not selected.iloc[0] and not selected.iloc[1]

    def test_nothing_is_filled_before_the_first_observation(self):
        """The distance is undefined there, and undefined must not pass."""
        from collection.raw_data_collector import _fillable_cells
        observed = pd.Series([np.nan, np.nan, 10.0, np.nan])
        gap = _years_since_observed(observed)
        selected = _fillable_cells(observed, observed.bfill(), gap, limit=3)
        assert not selected.iloc[0] and not selected.iloc[1]

    def test_the_collector_uses_it(self):
        source = (_SRC / 'collection' / 'raw_data_collector.py').read_text()
        assert '_fillable_cells(df_sorted[column], source, gap, limit)' in source


class TestImputationNeverAltersAnObservation:
    """The invariant the variance-stabilising transform broke.

    Two columns were pushed through a Yeo-Johnson before the carry and pulled
    back after. It changed no result by design -- both are filled by a carry,
    which selects a value already present and therefore commutes with any
    monotone transform -- but the round trip is not exact. Measured on a
    synthetic panel: observed cells came back altered by 1.5e-11, and imputed
    cells differed by 1.1e-11 from the direct carry. The imputation was
    perturbing observations in exchange for nothing.
    """

    @staticmethod
    def _panel(column, seed=9, entities=('AAA', 'BBB', 'CCC')):
        rng = np.random.default_rng(seed)
        return pd.DataFrame([
            {'country_code': entity, 'country_name': entity, 'year': year,
             column: (rng.normal(0, 5000) if rng.random() > 0.35 else np.nan),
             TARGET: rng.normal(70, 10)}
            for entity in entities for year in range(2000, 2016)])

    @pytest.mark.parametrize('column', [
        'gdp_per_capita_constant_2015',      # was Yeo-Johnson (has negatives)
        'intentional_homicides_per_100k',    # was in the same set
        'gini_index',                        # never transformed
    ])
    def test_observed_cells_pass_through_bitwise(self, column):
        import tempfile
        raw = self._panel(column)
        collector = _quiet(RawDataCollector)
        collector.output_dir = tempfile.mkdtemp()
        imputed = _quiet(collector.apply_conservative_imputation, raw.copy())

        observed = raw[column].notna() & raw.index.isin(imputed.index)
        assert observed.sum() > 0
        assert np.array_equal(imputed.loc[observed, column].to_numpy(),
                              raw.loc[observed, column].to_numpy()), (
            f'{column}: the imputation changed values it did not impute; '
            f'max |delta| = '
            f'{np.nanmax(np.abs(imputed.loc[observed, column].to_numpy() - raw.loc[observed, column].to_numpy())):.3e}'
        )

    def test_the_panel_carries_negatives(self):
        """Yeo-Johnson was reached only for columns with a non-positive value."""
        raw = self._panel('gdp_per_capita_constant_2015')
        assert (raw['gdp_per_capita_constant_2015'] <= 0).any(), (
            'without a non-positive value the old code took the log1p branch '
            'and this test would not exercise the one that was inexact'
        )

    def test_no_transform_survives_in_the_collector(self):
        source = (_SRC / 'collection' / 'raw_data_collector.py').read_text()
        for stale in ('yeojohnson', 'financial_log_transforms',
                      'log1p_shifted'):
            assert stale not in source, stale

    def test_the_carried_value_is_one_that_was_observed(self):
        """What makes the carry commute with a monotone transform.

        Also the reason removing the transform cannot have changed the method:
        a carry selects, it does not compute.
        """
        import tempfile
        column = 'gdp_per_capita_constant_2015'
        raw = self._panel(column)
        collector = _quiet(RawDataCollector)
        collector.output_dir = tempfile.mkdtemp()
        imputed = _quiet(collector.apply_conservative_imputation, raw.copy())

        filled = imputed[column].notna() & raw.loc[imputed.index, column].isna()
        assert filled.sum() > 0
        observed_values = set(raw[column].dropna().to_numpy().tolist())
        for value in imputed.loc[filled, column]:
            assert value in observed_values, (
                f'{value} was carried but never observed'
            )
