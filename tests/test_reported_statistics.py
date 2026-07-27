#!/usr/bin/env python3
"""An undefined statistic is reported as null, not as zero.

With a single observation the standard deviation does not exist; with the
column entirely missing, neither does the mean. DuckDB returns NULL, Polars
returns None, pandas returns NaN.

Two of the three paradigms converted that to 0.0, which is a claim about the
data -- "there is no variation" -- rather than an absence of one. The third
wrote NaN, which is not valid JSON under a strict parser. So the three
disagreed about the same degenerate input, in an artifact the usage guide
tells a reviewer to open.

The engines themselves agree on the statistics that are defined: mean, sample
standard deviation and median match across DuckDB, Polars and pandas to within
a few units in the last place. They are not bitwise identical, and they are
not claimed to be -- the bitwise claim is about predictions, which every
paradigm computes in pandas after materialising.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.base_architecture import BaseArchitectureML
from core.paradigm_registry import discover_paradigms

PARADIGMS = sorted(discover_paradigms())


class TestReportedStatistic:

    @pytest.mark.parametrize('value,expected', [
        (5.0, 5.0), (0.0, 0.0), (-3.25, -3.25), ('7.5', 7.5),
    ])
    def test_a_defined_value_passes_through(self, value, expected):
        assert BaseArchitectureML.reported_statistic(value) == expected

    @pytest.mark.parametrize('value', [None, float('nan'), float('inf'),
                                       float('-inf'), 'abc', object()])
    def test_an_undefined_value_becomes_null(self, value):
        assert BaseArchitectureML.reported_statistic(value) is None

    def test_zero_is_not_treated_as_missing(self):
        """`x or 0` conflated them; a genuine zero must survive."""
        assert BaseArchitectureML.reported_statistic(0.0) == 0.0
        assert BaseArchitectureML.reported_statistic(0.0) is not None

    def test_the_result_is_valid_json(self):
        payload = json.dumps({'std': BaseArchitectureML.reported_statistic(
            float('nan'))})
        assert payload == '{"std": null}'
        assert 'NaN' not in payload

    def test_a_raw_nan_would_not_be(self):
        """Pins why null rather than NaN: json.dump emits a non-standard token."""
        assert 'NaN' in json.dumps({'std': float('nan')})


class TestNoParadigmCoercesToZero:

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_no_statistic_falls_back_to_zero(self, paradigm):
        import ast as ast_module
        source = (_SRC / 'architectures_ml' / paradigm / 'setup.py').read_text()
        tree = ast_module.parse(source)
        statistics = next(
            (node for node in ast_module.walk(tree)
             if isinstance(node, ast_module.FunctionDef)
             and node.name == '_compute_target_statistics'), None)
        assert statistics is not None, paradigm
        body = ast_module.unparse(statistics)
        for pattern in ('or 0)', 'else 0.0', 'else 0)'):
            assert pattern not in body, (
                f'{paradigm} still turns an undefined statistic into zero '
                f'({pattern})'
            )

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_it_routes_through_the_shared_helper(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'setup.py').read_text()
        assert 'reported_statistic(' in source, paradigm


class TestTheEnginesAgreeOnDefinedStatistics:
    """Same definition, not the same last bit."""

    @pytest.fixture(scope='class')
    def sample(self):
        return pd.DataFrame({'t': np.random.default_rng(7).normal(50, 12, 997)})

    def test_the_standard_deviation_is_the_sample_one_everywhere(self, sample):
        import duckdb
        import polars as pl

        connection = duckdb.connect()
        connection.register('d', sample)
        from_sql = connection.execute('SELECT STDDEV(t) FROM d').fetchone()[0]
        from_polars = pl.from_pandas(sample)['t'].std()
        from_pandas = float(sample['t'].std())

        assert from_sql == pytest.approx(from_pandas, rel=1e-12)
        assert from_polars == pytest.approx(from_pandas, rel=1e-12)

    def test_it_is_not_the_population_one(self, sample):
        """Otherwise the agreement above would not distinguish the definitions."""
        assert float(sample['t'].std(ddof=0)) != pytest.approx(
            float(sample['t'].std()), rel=1e-6)

    def test_the_median_agrees(self, sample):
        import duckdb
        import polars as pl

        connection = duckdb.connect()
        connection.register('d', sample)
        from_sql = connection.execute(
            'SELECT PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY t) FROM d'
        ).fetchone()[0]
        assert from_sql == pytest.approx(float(sample['t'].median()), rel=1e-12)
        assert pl.from_pandas(sample)['t'].median() == pytest.approx(
            from_sql, rel=1e-12)

    def test_the_median_interpolates_on_an_even_count(self):
        """Where implementations most often diverge."""
        import duckdb
        import polars as pl

        frame = pd.DataFrame({'t': [1.0, 2.0, 3.0, 4.0]})
        connection = duckdb.connect()
        connection.register('d', frame)
        from_sql = connection.execute(
            'SELECT PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY t) FROM d'
        ).fetchone()[0]
        assert from_sql == pytest.approx(2.5)
        assert float(frame['t'].median()) == pytest.approx(2.5)
        assert pl.from_pandas(frame)['t'].median() == pytest.approx(2.5)

    def test_they_are_not_claimed_to_be_bitwise_identical(self, sample):
        """They differ in the last bits, and nothing depends on them not doing so.

        The bitwise claim covers predictions, which every paradigm computes in
        pandas after materialising.
        """
        import duckdb
        connection = duckdb.connect()
        connection.register('d', sample)
        from_sql = connection.execute('SELECT STDDEV(t) FROM d').fetchone()[0]
        assert from_sql != float(sample['t'].std())
        assert abs(from_sql - float(sample['t'].std())) < 1e-13


class TestUndefinedStatisticsInPractice:

    def test_one_observation_leaves_the_deviation_undefined(self):
        import duckdb
        connection = duckdb.connect()
        connection.register('d', pd.DataFrame({'t': [5.0]}))
        assert connection.execute('SELECT STDDEV(t) FROM d').fetchone()[0] is None
        assert math.isnan(pd.Series([5.0]).std())

    def test_both_become_null_in_the_artifact(self):
        import duckdb
        connection = duckdb.connect()
        connection.register('d', pd.DataFrame({'t': [5.0]}))
        from_sql = connection.execute('SELECT STDDEV(t) FROM d').fetchone()[0]
        assert BaseArchitectureML.reported_statistic(from_sql) is None
        assert BaseArchitectureML.reported_statistic(
            pd.Series([5.0]).std()) is None

    def test_the_mean_of_one_observation_survives(self):
        """Only the deviation is undefined there; the mean is not."""
        assert BaseArchitectureML.reported_statistic(
            float(pd.Series([5.0]).mean())) == 5.0
