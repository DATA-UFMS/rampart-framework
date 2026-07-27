#!/usr/bin/env python3
"""Does the hierarchical model beat the naive baseline?

Kapoor & Narayanan's case study is exactly this measurement: they took four
civil war papers claiming complex ML beat logistic regression, corrected the
leakage, and found the advantage gone in every case but one.

It answers the question L2 leaves open. K&N decline to subdivide L2 because
legitimacy needs domain judgment, and they name two ways a feature can be
illegitimate: being a proxy for the outcome, and making the prediction trivial
because it is already known at prediction time. The automated screen catches
the first. This comparison is what measures the second.

The gap is informative in both directions and neither is good unqualified. Near
zero means the ML adds nothing over carrying the last observed value forward.
Large means it is worth checking whether one feature is doing the work
trivially -- and which baseline won says a great deal.

It reads the prediction vectors rather than each paradigm's aggregate metrics:
one source with one schema against three different baseline JSON layouts, the
metric computed here the same way for both stages, and those vectors are the
ones the bitwise claim is about, so the comparison inherits that guarantee.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import discover_paradigms
from statistical_validation import baseline_comparison as bc

PARADIGMS = sorted(discover_paradigms())
FOLDS = [0, 1, 2, 3, 4, 5, 6, 7]


def _vectors(rows, seed, model_error):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=rows)
    return truth, truth + rng.normal(0, model_error, rows)


def _write(tmp_path, monkeypatch, *, model_error, baseline_error,
           paradigms=None, per_paradigm_error=None, second_model=True,
           model_error_by_fold=None):
    """Prediction artifacts for both stages, for every paradigm."""
    monkeypatch.setattr(
        bc, 'predictions_path',
        lambda paradigm, stage: str(
            tmp_path / f'predictions_{stage}_{paradigm}.parquet'))

    for paradigm in (paradigms or PARADIGMS):
        error = (per_paradigm_error or {}).get(paradigm, model_error)
        for stage, spread, names in (
                ('baseline', baseline_error,
                 ['naive_with_lag', 'global_mean']),
                ('hierarchical', error,
                 ['hierarchical'] + (['random_forest'] if second_model
                                     else []))):
            frames = []
            for fold in FOLDS:
                fold_spread = spread
                if stage == 'hierarchical' and model_error_by_fold:
                    fold_spread = model_error_by_fold.get(fold, spread)
                for index, name in enumerate(names):
                    truth, predicted = _vectors(
                        40, seed=100 * fold + index,
                        model_error=fold_spread + 0.5 * index)
                    frames.append(pd.DataFrame({
                        'fold': fold, 'model': name,
                        'row': np.arange(40), 'entity': 'E',
                        'y_true': truth, 'y_pred': predicted}))
            pd.concat(frames, ignore_index=True).to_parquet(
                tmp_path / f'predictions_{stage}_{paradigm}.parquet')


class TestTheMetric:

    def test_a_perfect_prediction_scores_one(self):
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        assert bc._r_squared(truth, truth) == pytest.approx(1.0)

    def test_predicting_the_mean_scores_zero(self):
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        assert bc._r_squared(truth, np.full(4, truth.mean())) == \
            pytest.approx(0.0)

    def test_worse_than_the_mean_goes_negative(self):
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        assert bc._r_squared(truth, np.array([4.0, 3.0, 2.0, 1.0])) < 0

    def test_a_constant_target_yields_no_score(self):
        """Nothing to explain, so R2 is undefined rather than zero."""
        assert bc._r_squared(np.full(5, 7.0), np.full(5, 7.0)) is None

    def test_too_few_points_yield_no_score(self):
        assert bc._r_squared(np.array([1.0]), np.array([1.0])) is None


class TestTheComparison:

    def test_a_better_model_shows_a_positive_gap(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, model_error=0.2, baseline_error=1.0)
        report = bc.compare(PARADIGMS[0], bootstrap_iters=400)
        assert report['mean_gap'] > 0
        assert report['beats_baseline'] is True

    def test_a_worse_model_shows_a_negative_gap(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, model_error=1.5, baseline_error=0.3)
        report = bc.compare(PARADIGMS[0], bootstrap_iters=400)
        assert report['mean_gap'] < 0
        assert report['beats_baseline'] is False
        assert report['folds_where_baseline_wins'] == len(FOLDS)

    def test_a_positive_point_estimate_is_not_superiority(self, tmp_path,
                                                          monkeypatch):
        """Superiority is the interval excluding zero, not the mean's sign.

        Measured here: the model wins seven folds of eight, the mean gap is
        about +0.22, and the interval runs from -0.17 to +0.36. Reading the
        point estimate alone would report an advantage the folds do not
        support, which is the reporting K&N found across the field.
        """
        # Wins in most folds and loses in two, which is the real shape of a
        # small advantage. A uniform gain across folds gives almost zero
        # variance and a narrow CI, and does not tell the two readings apart.
        _write(tmp_path, monkeypatch, model_error=0.85, baseline_error=1.0,
               model_error_by_fold={2: 1.3})
        report = bc.compare(PARADIGMS[0], bootstrap_iters=800)
        assert report['mean_gap'] > 0
        low, high = report['gap_ci95']
        assert low < 0 < high
        assert report['beats_baseline'] is False

    def test_a_clear_advantage_is_reported_as_one(self, tmp_path, monkeypatch):
        """Otherwise refusing superiority always would satisfy the test above."""
        _write(tmp_path, monkeypatch, model_error=0.85, baseline_error=1.0)
        report = bc.compare(PARADIGMS[0], bootstrap_iters=800)
        assert report['gap_ci95'][0] > 0
        assert report['beats_baseline'] is True

    def test_the_best_model_is_compared_not_the_average(self, tmp_path,
                                                        monkeypatch):
        """Best baseline against mean model would compare unlike with unlike.

        The stage writes more than one model, and the baseline side is reduced
        by taking its best. The model side has to be reduced the same way.
        """
        _write(tmp_path, monkeypatch, model_error=0.2, baseline_error=1.0)
        scores = bc._stage_scores(PARADIGMS[0], 'hierarchical')
        assert len(scores[0]) > 1, 'one model per fold cannot tell them apart'
        best = max(scores[0].values())
        average = sum(scores[0].values()) / len(scores[0])
        assert best != pytest.approx(average)

        report = bc.compare(PARADIGMS[0], bootstrap_iters=400)
        first = next(row for row in report['per_fold'] if row['fold'] == 0)
        assert first['model_r2'] == pytest.approx(best)

    def test_the_winning_baseline_is_named(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, model_error=0.2, baseline_error=1.0)
        report = bc.compare(PARADIGMS[0], bootstrap_iters=400)
        assert sum(report['baseline_wins'].values()) == len(FOLDS)
        assert 'naive_with_lag' in report['baseline_wins']

    def test_the_interval_records_its_method(self, tmp_path, monkeypatch):
        """Every interval in this repository names how it was produced."""
        _write(tmp_path, monkeypatch, model_error=0.2, baseline_error=1.0)
        report = bc.compare(PARADIGMS[0], bootstrap_iters=400)
        assert report['gap_ci95_method']

    def test_it_uses_the_shared_interval(self):
        """A second bootstrap would be a second policy for the same question."""
        source = (_SRC / 'statistical_validation'
                  / 'baseline_comparison.py').read_text()
        assert 'from statistical_validation.equivalence_estimation import' in source
        assert 'bootstrap_ci' in source
        assert 'def bootstrap_ci' not in source

    def test_absent_artifacts_yield_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bc, 'predictions_path',
                            lambda paradigm, stage: str(tmp_path / 'absent'))
        assert bc.compare(PARADIGMS[0], bootstrap_iters=100) is None


class TestCrossParadigmAgreement:
    """With Delta=0 the gap must be identical in all three."""

    def test_identical_predictions_give_an_identical_gap(self, tmp_path,
                                                         monkeypatch):
        _write(tmp_path, monkeypatch, model_error=0.2, baseline_error=1.0)
        report = bc.analyze(bootstrap_iters=400)
        agreement = report['cross_paradigm_agreement']
        assert agreement['checked'] is True
        assert agreement['consistent'] is True
        assert agreement['max_absolute_difference'] == pytest.approx(0.0)

    def test_a_diverging_paradigm_is_caught(self, tmp_path, monkeypatch):
        """It would contradict the bitwise claim, so it cannot pass quietly."""
        _write(tmp_path, monkeypatch, model_error=0.2, baseline_error=1.0,
               per_paradigm_error={PARADIGMS[0]: 0.9})
        report = bc.analyze(bootstrap_iters=400)
        assert report['cross_paradigm_agreement']['consistent'] is False

    def test_the_run_halts_on_divergence(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, model_error=0.2, baseline_error=1.0,
               per_paradigm_error={PARADIGMS[0]: 0.9})
        monkeypatch.setattr(bc, 'RESULTS_DIR', str(tmp_path / 'stats'))
        with pytest.raises(ValueError,
                           match='paradigms disagree about the difference'):
            bc.main()

    def test_agreement_needs_two_paradigms(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, model_error=0.2, baseline_error=1.0,
               paradigms=[PARADIGMS[0]])
        report = bc.analyze(bootstrap_iters=200)
        assert report['cross_paradigm_agreement']['checked'] is False


class TestTheOutputs:

    @pytest.fixture
    def written(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, model_error=0.2, baseline_error=1.0)
        monkeypatch.setattr(bc, 'RESULTS_DIR', str(tmp_path / 'stats'))
        report = bc.analyze(bootstrap_iters=400)
        json_path, tex_path = bc.write_outputs(report)
        return Path(json_path), Path(tex_path), report

    def test_the_json_carries_every_fold(self, written):
        json_path, _, _ = written
        payload = json.loads(json_path.read_text())
        for entry in payload['by_paradigm'].values():
            assert len(entry['per_fold']) == len(FOLDS)

    def test_the_latex_escapes_the_paradigm_names(self, written):
        import re
        _, tex_path, _ = written
        for line in tex_path.read_text().splitlines():
            if line.strip().startswith('%'):
                continue
            assert not re.search(r'(?<!\\)_', line), line

    def test_the_latex_column_count_agrees(self, written):
        import re
        _, tex_path, _ = written
        table = tex_path.read_text()
        spec = re.search(r'\\begin\{tabular\}\{([lrc|]+)\}', table)
        columns = len([c for c in spec.group(1) if c in 'lrc'])
        body = [line for line in table.splitlines()
                if '&' in line and not line.strip().startswith('%')]
        assert {line.count('&') + 1 for line in body} == {columns}

    def test_the_metric_is_named(self, written):
        _, _, report = written
        assert report['metric'] == 'r2_out_of_sample'


class TestItIsWiredIntoThePipeline:

    def test_the_stage_is_registered(self):
        source = (_ROOT / 'pipeline.py').read_text()
        assert 'baseline_comparison.py' in source

    def test_it_runs_after_the_stages_it_depends_on(self):
        """It reads prediction vectors, which the model stages write."""
        source = (_ROOT / 'pipeline.py').read_text()
        assert source.index('architectural_benchmark.py') < source.index(
            'baseline_comparison.py')
