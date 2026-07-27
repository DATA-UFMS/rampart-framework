#!/usr/bin/env python3
"""The resource table describes the measured runs, and only them.

Two ways it described something else:

  - warmup runs were in it. The latency CSV excluded them, the resource log did
    not, and warmups are exactly the runs with cold caches and unpaged memory --
    the atypical profile the repetitions exist to avoid. Two warmups against ten
    repetitions is a sixth of the rows.

  - the log was opened in append mode and nothing truncated it, so a second
    pipeline run stacked on the first. The published table then averaged runs
    from different versions of the code, and the count grew with how many times
    the benchmark had been run on that machine.
"""

import ast
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from benchmarking import derive_resource_usage_table as module


def _record(run_id, is_warmup, rss):
    return {
        'run_id': run_id, 'phase': 'setup', 'architecture': 'sql_engine',
        'step': 'setup', 'is_warmup': is_warmup,
        'cpu_proc': {'mean': 10.0, 'max': 20.0, 'n': 5},
        'cpu_sys': {'mean': 30.0, 'max': 40.0},
        'rss_mb': {'mean': rss, 'max': rss},
        'mem_sys_percent': {'mean': 50.0, 'max': 60.0},
        'io_read_mb': 1.0, 'io_write_mb': 2.0,
    }


def _write(path, records):
    path.write_text('\n'.join(json.dumps(r) for r in records) + '\n')


class TestWarmupsAreExcluded:

    def test_a_warmup_row_does_not_reach_the_table(self, tmp_path):
        path = tmp_path / 'log.jsonl'
        _write(path, [_record(0, True, 9999.0), _record(1, False, 100.0)])
        frame = module._load_jsonl(path)
        assert len(frame) == 1
        assert frame['rss_mb_mean'].iloc[0] == 100.0

    def test_the_measured_rows_survive(self, tmp_path):
        """Otherwise dropping everything would satisfy the test above."""
        path = tmp_path / 'log.jsonl'
        _write(path, [_record(index, False, 100.0) for index in range(10)])
        assert len(module._load_jsonl(path)) == 10

    def test_a_warmup_would_move_the_mean(self, tmp_path):
        """Without this the exclusion could be filtering nothing that matters."""
        path = tmp_path / 'log.jsonl'
        measured = [_record(index, False, 100.0) for index in range(10)]
        _write(path, [_record(99, True, 9999.0)] + measured)
        assert module._load_jsonl(path)['rss_mb_mean'].mean() == 100.0

    def test_an_unmarked_record_halts(self, tmp_path):
        """It predates the distinction, so which side it falls on is unknown."""
        path = tmp_path / 'log.jsonl'
        stale = _record(0, False, 100.0)
        del stale['is_warmup']
        _write(path, [stale])
        with pytest.raises(ValueError, match='warmup'):
            module._load_jsonl(path)


class TestTheLogBelongsToThisRun:

    def test_the_benchmark_truncates_it_before_the_loop(self):
        source = (_SRC / 'benchmarking' / 'architectural_benchmark.py') \
            .read_text()
        tree = ast.parse(source)
        run = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == 'run')
        body = ast.get_source_segment(source, run)
        truncate = body.index('open(self.resource_log_path, "w")')
        loop = body.index('for run_id in range(total_runs)')
        assert truncate < loop, (
            'the log must be emptied before the first run, not after'
        )

    def test_the_monitor_still_appends_within_a_run(self):
        """Truncating per record would leave only the last one."""
        source = (_SRC / 'benchmarking' / 'architectural_benchmark.py') \
            .read_text()
        assert 'open(self.log_path, "a")' in source

    def test_the_monitor_records_the_warmup_flag(self):
        source = (_SRC / 'benchmarking' / 'architectural_benchmark.py') \
            .read_text()
        assert '"is_warmup": bool(self.is_warmup)' in source
        assert 'is_warmup=run_id < self.warmup_runs' in source
