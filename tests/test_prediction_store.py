#!/usr/bin/env python3
"""Prediction vector persistence."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.prediction_store import PredictionRecorder


class TestRecorder:

    def test_records_one_row_per_prediction(self):
        recorder = PredictionRecorder('task_graph')
        recorder.record(fold=0, model='simple_hierarchical',
                        y_true=[1.0, 2.0, 3.0], y_pred=[1.1, 2.1, 3.1],
                        entities=['AR', 'AR', 'BR'])
        frame = recorder.frame()
        assert len(frame) == 3
        assert list(frame['row']) == [0, 1, 2]
        assert list(frame['entity']) == ['AR', 'AR', 'BR']
        assert frame['fold'].unique().tolist() == [0]

    def test_accumulates_across_folds_and_models(self):
        recorder = PredictionRecorder('sql_engine')
        for fold in range(2):
            for model in ('simple_hierarchical', 'random_forest_hierarchical'):
                recorder.record(fold=fold, model=model,
                                y_true=[1.0, 2.0], y_pred=[1.0, 2.0])
        frame = recorder.frame()
        assert len(frame) == 8
        assert set(frame['fold']) == {0, 1}
        assert set(frame['model']) == {'simple_hierarchical',
                                       'random_forest_hierarchical'}

    def test_length_mismatch_is_rejected(self):
        recorder = PredictionRecorder('dataframe_lib')
        with pytest.raises(ValueError, match='y_true has'):
            recorder.record(fold=0, model='m', y_true=[1.0, 2.0], y_pred=[1.0])

    def test_entity_count_mismatch_is_rejected(self):
        recorder = PredictionRecorder('dataframe_lib')
        with pytest.raises(ValueError, match='entities for'):
            recorder.record(fold=0, model='m', y_true=[1.0, 2.0],
                            y_pred=[1.0, 2.0], entities=['AR'])

    def test_missing_entities_are_allowed(self):
        recorder = PredictionRecorder('task_graph')
        recorder.record(fold=0, model='m', y_true=[1.0], y_pred=[1.0])
        assert recorder.frame()['entity'].isna().all()


class TestArtifact:

    def test_write_is_a_noop_when_empty(self, tmp_path):
        recorder = PredictionRecorder('task_graph')
        assert recorder.write(str(tmp_path / 'p.parquet')) is None

    def test_float_values_survive_the_round_trip_bitwise(self, tmp_path):
        """Equivalence is asserted bitwise, so storage must not perturb values."""
        pytest.importorskip('pyarrow')
        rng = np.random.default_rng(11)
        values = rng.normal(size=256)
        recorder = PredictionRecorder('task_graph')
        recorder.record(fold=0, model='m', y_true=values, y_pred=values / 3.0)

        path = str(tmp_path / 'predictions.parquet')
        recorder.write(path)
        restored = pd.read_parquet(path)

        assert np.array_equal(restored['y_true'].to_numpy(), values)
        assert np.array_equal(restored['y_pred'].to_numpy(), values / 3.0)
