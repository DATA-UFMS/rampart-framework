#!/usr/bin/env python3
"""The gap is wide enough for the dependence it exists to outrun.

P2 puts a gap between the training window and the evaluation windows, and the
value was 2 years with nothing behind it. The README cited Roberts et al.
(2017), which gives the criterion and not the number, and no code measured the
quantity the criterion is about -- a grep for autocorrelation, acf or variogram
returned nothing. A reviewer reproducing the citation would have found that.

The criterion, in the words of `blockCV`, the package three of that paper's
authors wrote to implement it: blocks should be bigger than the range of
autocorrelation *in the model residual*. The qualifier decides the answer here.
Measured on the raw target the dependence at lag 2 is 0.56 and the gap looks
too narrow; measured on what the model fails to explain it is 0.03 and the gap
is exactly right. The model reads lags of the target as features, so the
difference between the two is the structure it already absorbs.
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
sys.path.insert(0, str(_ROOT / 'scripts' / 'validation'))

import measure_dependence_range as mdr  # noqa: E402
from core.scientific_config import SCIENTIFIC_CONFIG  # noqa: E402


def _panel(rho, entity_offset, entities=40, years=14, seed=3):
    """A residual series with a known AR(1) decay and a known entity offset."""
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(entities):
        offset = rng.normal(0, entity_offset) if entity_offset else 0.0
        value = rng.normal()
        for step in range(years):
            value = rho * value + rng.normal(0, np.sqrt(1 - rho ** 2))
            rows.append({'entity': entity, 'year': 2000 + step,
                         'residual': offset + value})
    frame = pd.DataFrame(rows)
    frame['within_entity'] = (frame['residual']
                              - frame.groupby('entity')['residual']
                              .transform('mean'))
    return frame


class TestTheMeasurement:
    """The instrument reads what it claims to read."""

    def test_it_recovers_a_known_decay(self):
        """An AR(1) at rho decays to rho^k; the estimate has to follow."""
        frame = _panel(rho=0.3, entity_offset=0.0)
        first = mdr._autocorrelation(frame, 'within_entity', 1)
        second = mdr._autocorrelation(frame, 'within_entity', 2)
        assert 0.15 < first < 0.45, first
        assert abs(second) < 0.20, second

    def test_it_separates_the_entity_offset_from_the_decay(self):
        """The distinction the whole diagnostic turns on.

        A large per-entity offset makes the raw autocorrelation high at every
        lag while the temporal structure is gone by the second. Reading the
        total would call for a buffer that no buffer can supply, because the
        offset is non-independence between rows and not dependence over time.
        """
        frame = _panel(rho=0.05, entity_offset=3.0)
        total = mdr._autocorrelation(frame, 'residual', 4)
        within = mdr._autocorrelation(frame, 'within_entity', 4)
        assert total > 0.5, f'the offset should dominate the total: {total}'
        assert abs(within) < 0.20, f'the decay should be spent: {within}'

    def test_too_few_pairs_yields_no_estimate(self):
        """Silence rather than a number computed from nothing."""
        frame = _panel(rho=0.3, entity_offset=0.0, entities=2, years=3)
        assert mdr._autocorrelation(frame, 'within_entity', 2) is None


class TestTheGapCoversTheDependence:
    """The configured gap against the measurement, on the real panels."""

    #: Runs of the published artifacts. Skipped rather than failed when absent:
    #: a fresh clone has no completed run, and this test is a check on the
    #: configuration, not on whether someone has executed the pipeline.
    RUNS = {
        'worldbank': Path('/home/eos/pesquisa/eos/dw-vs-dl-dropout-prediction'
                          '-latam/azure_results_v7_wb'),
        'inep_censo': Path('/home/eos/pesquisa/eos/dw-vs-dl-dropout-prediction'
                           '-latam/azure_results_v7_inep'),
    }

    def _measure(self, run: Path):
        prep = next((run / 'ml_pipeline' / 'architectures').glob('*/prep'),
                    None)
        if prep is None:
            pytest.skip(f'no completed run under {run}')
        paradigm = prep.parent.name
        folds = prep / f'temporal_folds_{paradigm}.json'
        results = next((prep.parent / 'models' / 'hierarchical_results')
                       .glob('hierarchical_analysis_*.json'), None)
        if not (folds.exists() and results):
            pytest.skip(f'no hierarchical results under {run}')
        return mdr.measure(results, folds)

    @pytest.mark.parametrize('dataset', sorted(RUNS))
    def test_the_dependence_is_spent_within_the_gap(self, dataset):
        measured = self._measure(self.RUNS[dataset])
        gap = int(SCIENTIFIC_CONFIG['temporal_gap_years'])
        spent = measured['dependence_spent_at_lag']
        assert spent is not None, (
            f'{dataset}: residual dependence still above '
            f'{mdr.NEGLIGIBLE} at lag {mdr.MAX_LAG}; no configured gap covers it')
        assert spent <= gap, (
            f'{dataset}: dependence spent at lag {spent}, gap is {gap}. '
            f'Either widen the gap or stop citing the buffer criterion for it.')

    @pytest.mark.parametrize('dataset', sorted(RUNS))
    def test_the_entity_offset_is_reported_not_hidden(self, dataset):
        """What the gap cannot fix has to come out as a number.

        Roughly two thirds of the residual variance is a per-entity offset in
        both panels. That is L3.2, which this framework declares as needing an
        argument from the author rather than claiming to solve; the measurement
        turns the declaration into a quantity.
        """
        measured = self._measure(self.RUNS[dataset])
        share = measured['entity_share_of_variance']
        assert share is not None and share > 0.0, dataset
        assert measured['autocorrelation_by_lag'][1]['total'] is not None

    def test_the_criterion_is_the_residual_not_the_raw_series(self, tmp_path):
        """Pinned because measuring the raw series condemns an adequate gap.

        On the World Bank panel the target's own autocorrelation at lag 2 is
        about 0.56 after removing entity means, against 0.03 in the residual.
        A future edit that points this diagnostic at the target would fail the
        gap for structure the model already models.
        """
        import json as _json

        # Behavioural, not textual. Feeding known values and checking the
        # arithmetic is the only form that survives a rename, and the earlier
        # source-text version did not catch pointing the diagnostic at the
        # target: on these panels the target and the residual happen to give
        # the same verdict, so the empirical test below cannot separate them
        # either. This can.
        results = tmp_path / 'hierarchical_analysis_probe.json'
        folds = tmp_path / 'temporal_folds_probe.json'
        results.write_text(_json.dumps({'folds': [{
            'fold_id': 0,
            'models': {'simple_hierarchical': {'test': {
                'y_true': [10.0, 20.0, 30.0, 40.0],
                'predictions': [1.0, 2.0, 3.0, 4.0]}}}}]}))
        folds.write_text(_json.dumps({'folds': [
            {'fold_id': 0, 'test_start': 2020, 'test_end': 2021}]}))

        frame = mdr._residuals(results, folds)
        assert sorted(frame['residual']) == [9.0, 18.0, 27.0, 36.0], (
            f'the residual is not actual minus predicted: '
            f'{sorted(frame["residual"])}')
        assert sorted(frame['year'].unique()) == [2020, 2021]
        assert sorted(frame['entity'].unique()) == [0, 1], (
            'the entity of each row is not recovered from the row ordering')
