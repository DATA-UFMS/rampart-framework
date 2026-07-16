#!/usr/bin/env python3
"""A failed benchmark phase must not become a measurement.

A phase that raised was recorded as duration_ns = -1 and the benchmark carried on.
That value reached the CSV as duration_s = -1e-09, which no consumer filtered: it
entered the paired vectors, the means, the percentiles and the paired tests as a
negative latency.

The published runs contain no such row, so no published number is affected. The
INEP re-run takes about twenty-nine hours, where a transient failure in one
repetition is a realistic outcome -- and the failure mode is silent.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from benchmarking.architectural_benchmark import PhaseResult


class TestNonPositiveDurationIsNotALatency:

    def test_valid_duration_converts(self):
        result = PhaseResult(run_id=0, phase='baseline', architecture='x',
                             step='s', duration_ns=2_500_000_000, records=10)
        assert result.duration_s == pytest.approx(2.5)

    @pytest.mark.parametrize('duration_ns', [-1, 0])
    def test_non_positive_duration_refuses_to_convert(self, duration_ns):
        result = PhaseResult(run_id=0, phase='baseline', architecture='x',
                             step='s', duration_ns=duration_ns, records=10)
        with pytest.raises(ValueError, match='non-positive duration'):
            _ = result.duration_s

    def test_the_error_names_the_phase(self):
        result = PhaseResult(run_id=3, phase='hierarchical',
                             architecture='task_graph', step='fit',
                             duration_ns=-1, records=None)
        with pytest.raises(ValueError) as exc:
            _ = result.duration_s
        message = str(exc.value)
        assert 'hierarchical' in message and 'task_graph' in message

    def test_throughput_stays_none_rather_than_raising(self):
        """Throughput already guarded; it reports absence, not an error."""
        result = PhaseResult(run_id=0, phase='baseline', architecture='x',
                             step='s', duration_ns=-1, records=10)
        assert result.throughput_rps is None


class TestNoSentinelIsWritten:

    def test_benchmark_does_not_record_a_negative_duration(self):
        source = (_SRC / 'benchmarking'
                  / 'architectural_benchmark.py').read_text()
        assert 'duration_ns=-1' not in source, (
            'a failed phase is being recorded as a measurable row'
        )

    def test_failure_raises_rather_than_returning_a_row(self):
        source = (_SRC / 'benchmarking'
                  / 'architectural_benchmark.py').read_text()
        block = source[source.index('def measure('):]
        block = block[:block.index('\n        # ---')]
        assert 'raise RuntimeError' in block, (
            'a failed phase must abort: the paradigms stop having the same '
            'paired observations'
        )


class TestConsumersRejectContaminatedInput:
    """A CSV written before the fix must not be consumed silently."""

    @pytest.fixture
    def csv_with_failure(self, tmp_path):
        rows = []
        for run_id in range(3):
            for arch in ('sql_engine', 'task_graph'):
                rows.append({'run_id': run_id, 'phase': 'baseline',
                             'architecture': arch, 'duration_s': 1.5,
                             'duration_ns': 1_500_000_000, 'records': 10})
        # One repetition where a phase failed, as the old code recorded it.
        rows.append({'run_id': 3, 'phase': 'baseline',
                     'architecture': 'task_graph', 'duration_s': -1e-09,
                     'duration_ns': -1, 'records': None})
        path = tmp_path / 'architectural_benchmark_results.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    @pytest.fixture
    def clean_csv(self, tmp_path):
        rows = [{'run_id': r, 'phase': 'baseline', 'architecture': a,
                 'duration_s': 1.5, 'duration_ns': 1_500_000_000,
                 'records': 10}
                for r in range(3) for a in ('sql_engine', 'task_graph')]
        path = tmp_path / 'clean.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    @pytest.mark.parametrize('module_name', [
        'statistical_validation.significance_tests',
        'statistical_validation.effect_analysis',
    ])
    def test_loader_rejects_the_contaminated_csv(self, module_name,
                                                 csv_with_failure):
        import importlib
        module = importlib.import_module(module_name)
        with pytest.raises(ValueError, match='non-positive duration'):
            module.load_benchmark(csv_with_failure)

    @pytest.mark.parametrize('module_name', [
        'statistical_validation.significance_tests',
        'statistical_validation.effect_analysis',
    ])
    def test_loader_accepts_a_clean_csv(self, module_name, clean_csv):
        import importlib
        module = importlib.import_module(module_name)
        frame = module.load_benchmark(clean_csv)
        assert len(frame) == 6

    def test_error_reports_how_many_rows(self, csv_with_failure):
        from statistical_validation import significance_tests as st
        with pytest.raises(ValueError) as exc:
            st.load_benchmark(csv_with_failure)
        assert '1 rows' in str(exc.value)
