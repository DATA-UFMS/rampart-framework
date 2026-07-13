#!/usr/bin/env python3
"""Cross-paradigm prediction equivalence verifier."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from statistical_validation.prediction_equivalence import _compare_vector, verify
from core.scientific_config import SCIENTIFIC_CONFIG

TOLERANCE = float(SCIENTIFIC_CONFIG.get('float_precision_tolerance', 1e-9))


def _vector(y_true, y_pred, entities=None):
    n = len(y_true)
    return pd.DataFrame({
        'fold': [0] * n,
        'model': ['m'] * n,
        'row': list(range(n)),
        'entity': entities if entities is not None else [None] * n,
        'y_true': np.asarray(y_true, dtype=float),
        'y_pred': np.asarray(y_pred, dtype=float),
    })


class TestVectorComparison:

    def test_identical_vectors_are_equivalent(self):
        v = _vector([1.0, 2.0], [1.5, 2.5], ['AR', 'BR'])
        assert _compare_vector(v, v.copy(), TOLERANCE) is None

    def test_row_count_difference_is_misalignment(self):
        verdict = _compare_vector(
            _vector([1.0, 2.0], [1.0, 2.0]), _vector([1.0], [1.0]), TOLERANCE
        )
        assert verdict['kind'] == 'misaligned'
        assert 'row count' in verdict['reason']

    def test_entity_sequence_difference_is_misalignment(self):
        verdict = _compare_vector(
            _vector([1.0, 2.0], [1.0, 2.0], ['AR', 'BR']),
            _vector([1.0, 2.0], [1.0, 2.0], ['AR', 'CL']),
            TOLERANCE,
        )
        assert verdict['kind'] == 'misaligned'
        assert verdict['first_mismatch_row'] == 1

    def test_observed_target_difference_is_misalignment_not_divergence(self):
        """Different evaluated rows must not be reported as differing predictions."""
        verdict = _compare_vector(
            _vector([1.0, 2.0], [9.0, 9.0]),
            _vector([1.0, 7.0], [9.0, 9.0]),
            TOLERANCE,
        )
        assert verdict['kind'] == 'misaligned'
        assert 'observed targets' in verdict['reason']

    def test_prediction_difference_is_divergence(self):
        verdict = _compare_vector(
            _vector([1.0, 2.0], [1.0, 2.0]),
            _vector([1.0, 2.0], [1.0, 2.5]),
            TOLERANCE,
        )
        assert verdict['kind'] == 'divergent'
        assert verdict['differing_rows'] == 1
        assert verdict['max_abs_difference'] == pytest.approx(0.5)
        assert verdict['within_float_tolerance'] is False

    def test_subtolerance_difference_is_still_reported(self):
        """A difference within tolerance is characterised, never excused."""
        verdict = _compare_vector(
            _vector([1.0], [1.0]),
            _vector([1.0], [1.0 + TOLERANCE / 10]),
            TOLERANCE,
        )
        assert verdict is not None
        assert verdict['kind'] == 'divergent'
        assert verdict['within_float_tolerance'] is True


class TestReport:

    def test_without_predictions_the_status_is_insufficient(self, monkeypatch):
        import statistical_validation.prediction_equivalence as module
        monkeypatch.setattr(module, 'load_predictions', lambda paradigm: None)
        report = verify()
        assert report['status'] == 'insufficient_data'
        assert report['paradigms_without_predictions']

    def test_equivalent_paradigms_report_equivalent(self, monkeypatch):
        import statistical_validation.prediction_equivalence as module
        shared = _vector([1.0, 2.0, 3.0], [1.1, 2.1, 3.1], ['AR', 'BR', 'CL'])
        monkeypatch.setattr(module, 'load_predictions',
                            lambda paradigm: shared.copy())
        report = verify()
        assert report['status'] == 'equivalent'
        assert report['vectors_compared'] == len(report['comparisons'])
        assert not report['violations']

    def test_one_divergent_paradigm_reports_violation(self, monkeypatch):
        import statistical_validation.prediction_equivalence as module
        clean = _vector([1.0, 2.0], [1.0, 2.0], ['AR', 'BR'])
        dirty = _vector([1.0, 2.0], [1.0, 2.9], ['AR', 'BR'])

        def loader(paradigm):
            return dirty.copy() if paradigm == 'task_graph' else clean.copy()

        monkeypatch.setattr(module, 'load_predictions', loader)
        report = verify()
        assert report['status'] == 'violation'
        assert any(v['kind'] == 'divergent' for v in report['violations'])

    def test_missing_vector_in_one_paradigm_is_disjoint(self, monkeypatch):
        import statistical_validation.prediction_equivalence as module
        full = _vector([1.0], [1.0], ['AR'])
        extra = pd.concat([full, full.assign(fold=1)], ignore_index=True)

        def loader(paradigm):
            return extra.copy() if paradigm == 'task_graph' else full.copy()

        monkeypatch.setattr(module, 'load_predictions', loader)
        report = verify()
        assert report['status'] == 'violation'
        assert any(v['kind'] == 'disjoint' for v in report['violations'])
