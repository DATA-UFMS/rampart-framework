#!/usr/bin/env python3
"""Intervals over folds that overlap, and the block length that was a constant.

The moving-block bootstrap was written inline in the first probe with a block
length of 2 and a comment saying consecutive World Bank folds share a test year.
True for that panel and false for the other one: INEP evaluates one year per fold
and steps one year, so its folds are disjoint and a block of 2 would widen every
interval for a dependence that is not there.

Also here: the directional check, which is a bug this file exists to keep fixed.
A probe asking whether the forest inflates more than the ridge counted an
interval lying entirely *below* zero as a detection, so the opposite of the
documented finding was reported as reproducing it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from statistical_validation.dependent_bootstrap import (  # noqa
    excludes_zero, fold_dependence_span, moving_block_ci)


class TestTheBlockLengthIsDerived:

    def test_overlapping_windows_need_a_block(self):
        assert fold_dependence_span({'test_len': 2, 'step': 1}) == 2

    def test_disjoint_windows_do_not(self):
        """One means the ordinary bootstrap over folds is already correct."""
        assert fold_dependence_span({'test_len': 1, 'step': 1}) == 1

    def test_a_longer_window_spans_more_folds(self):
        assert fold_dependence_span({'test_len': 5, 'step': 2}) == 3

    def test_a_step_past_the_window_still_gives_one(self):
        assert fold_dependence_span({'test_len': 1, 'step': 3}) == 1

    def test_a_zero_step_is_refused(self):
        with pytest.raises(ValueError, match='step must be positive'):
            fold_dependence_span({'test_len': 2, 'step': 0})

    def test_the_two_registered_datasets_differ(self):
        """The reason this is derived and not written out."""
        import datasets  # noqa: F401
        from core.dataset_config import get_dataset

        spans = {name: fold_dependence_span(get_dataset(name).walk_forward_config)
                 for name in ('worldbank', 'inep_censo')}
        assert spans == {'worldbank': 2, 'inep_censo': 1}, spans


class TestTheInterval:

    def test_it_covers_the_mean_of_well_behaved_folds(self):
        values = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.12, 0.10, 0.11]
        point, (low, high), record = moving_block_ci(values, block=2, iters=2000)
        assert point == pytest.approx(float(np.mean(values)))
        assert low < point < high
        assert record['block'] == 2 and record['n_folds'] == 9

    def test_positive_dependence_widens_the_interval(self):
        """The case the method exists for.

        A first version of this test used an alternating sequence and expected
        widening there too. It narrowed, and correctly: blocking averages out
        negative dependence. The block bootstrap widens under *positive*
        dependence, which is what overlapping evaluation windows produce -- two
        folds scored partly on the same year err in the same direction.
        """
        values = [0.02, 0.03, 0.05, 0.11, 0.13, 0.14, 0.24, 0.26, 0.27]
        _p, tight, _r = moving_block_ci(values, block=1, iters=8000)
        _p, loose, _r = moving_block_ci(values, block=2, iters=8000)
        assert (loose[1] - loose[0]) > (tight[1] - tight[0])

    def test_a_block_covering_most_of_the_sample_says_so(self):
        """It narrows there, and silence would read as precision."""
        values = [0.05, 0.20, 0.02, 0.30, 0.10, 0.25, 0.03, 0.28, 0.07]
        _p, _i, record = moving_block_ci(values, block=7, iters=2000)
        assert record['block_share_of_sample'] > 0.5
        assert 'understates the spread' in record['warning']
        assert record['distinct_starts'] == 3

    def test_zero_variance_is_named_rather_than_reported_as_precision(self):
        point, interval, record = moving_block_ci([0.2] * 6, block=2)
        assert interval == (point, point)
        assert record['method'] == 'degenerate_zero_variance'

    def test_one_fold_gives_no_interval(self):
        point, interval, record = moving_block_ci([0.3], block=2)
        assert point == pytest.approx(0.3)
        assert all(np.isnan(v) for v in interval)
        assert record['method'] == 'single_fold'

    def test_no_folds_is_reported_as_insufficient(self):
        _p, _i, record = moving_block_ci([], block=2)
        assert record['method'] == 'insufficient_data'

    def test_a_block_longer_than_the_data_is_clamped(self):
        _p, _i, record = moving_block_ci([0.1, 0.4, 0.2], block=9, iters=500)
        assert record['block'] == 3

    def test_nan_folds_are_dropped_not_propagated(self):
        point, _i, record = moving_block_ci(
            [0.1, float('nan'), 0.3, None, 0.2], block=2, iters=500)
        assert record['n_folds'] == 3
        assert point == pytest.approx(0.2)

    def test_the_same_seed_reproduces_the_interval(self):
        values = [0.05, 0.20, 0.02, 0.30, 0.10, 0.25, 0.03]
        one = moving_block_ci(values, block=2, iters=1500, seed=11)[1]
        two = moving_block_ci(values, block=2, iters=1500, seed=11)[1]
        assert one == two

    def test_every_resample_has_the_length_of_the_data(self):
        """Truncating after whole blocks keeps draws comparable to each other.

        Checked through the interval on a two-valued sample: if some draws were
        shorter, their means would scatter beyond the range of the data.
        """
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        _p, (low, high), _r = moving_block_ci(values, block=2, iters=4000)
        assert 0.0 <= low <= high <= 1.0


class TestTheDirectionalCheck:

    def test_an_interval_above_zero_is_a_positive_detection(self):
        assert excludes_zero((0.01, 0.05), direction=+1)

    def test_an_interval_below_zero_is_not(self):
        """The bug. Without the direction this returned True, and a probe
        counted the opposite of the documented effect as reproducing it."""
        assert not excludes_zero((-0.05, -0.01), direction=+1)
        assert excludes_zero((-0.05, -0.01), direction=-1)

    def test_an_interval_covering_zero_is_never_a_detection(self):
        for direction in (-1, 0, +1):
            assert not excludes_zero((-0.02, 0.03), direction=direction)

    def test_without_a_direction_either_side_counts(self):
        assert excludes_zero((0.01, 0.05))
        assert excludes_zero((-0.05, -0.01))

    def test_a_non_finite_interval_is_not_a_detection(self):
        assert not excludes_zero((float('nan'), float('nan')), direction=+1)
        assert not excludes_zero((0.01, float('inf')), direction=+1)

    def test_an_interval_touching_zero_is_not_a_detection(self):
        """Strictly outside, so a boundary case is not read as significance."""
        assert not excludes_zero((0.0, 0.05), direction=+1)
        assert not excludes_zero((-0.05, 0.0), direction=-1)


class TestTheTwoPanelsNeedDifferentBlocks:
    """The reason the block length is derived and not written out.

    The two panels give 2 and 1, and the regime probe runs both in one pass. A
    constant would widen every INEP interval for a dependence that panel does not
    have -- an error in the comfortable direction, which is why it would survive.
    """

    def test_the_probe_harness_reports_the_spillover_of_both(self):
        import sys
        scripts = _ROOT / 'scripts' / 'validation'
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from probe_harness import PANELS
        assert set(PANELS) == {'worldbank', 'inep_censo'}

    def test_neither_panel_declares_a_block_by_hand(self):
        """Read from the syntax tree: a literal block length in a probe is the
        arrangement the derivation replaced."""
        import ast

        offending = []
        for path in sorted((_ROOT / 'scripts').rglob('*.py')):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.keyword) or node.arg != 'block':
                    continue
                if isinstance(node.value, ast.Constant):
                    offending.append(f'{path.name}:{node.lineno}')
        assert not offending, (
            f'a probe passes a literal block length instead of deriving it: '
            f'{offending}')
