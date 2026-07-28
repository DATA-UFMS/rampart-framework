#!/usr/bin/env python3
"""Splitting severity into memorisation and a shift in generalisation.

Aggregate severity adds the two, and measured on this panel they order models
oppositely: the random forest wins the local channel at every dose, the ridge wins
the global one, and the aggregate follows the global because the rows that did not
leak outnumber those that did. So the decomposition is what carries the finding,
and its arithmetic has to be checkable rather than trusted.

Most of what is below is constructed so the right answer is known in advance --
a model made perfect on handed rows must read local = 1, a model untouched must
read 0 on both -- plus the identity that ties the parts to the whole.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from statistical_validation.leakage_channels import (  # noqa
    check_identity, decompose_fold, handed_mask, summarise)


def window(rows=20, seed=0):
    """Truth and a clean prediction with ordinary error on every row."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=rows) * 10.0
    clean = truth + rng.normal(size=rows) * 2.0
    return truth, clean


class TestThePartition:

    def test_it_selects_exactly_the_recorded_keys(self):
        entities = ['BRA', 'BRA', 'ARG', 'CHL']
        years = [2020, 2021, 2020, 2021]
        mask = handed_mask(entities, years, [['BRA', 2021], ['CHL', 2021]])
        assert list(mask) == [False, True, False, True]

    def test_the_year_type_does_not_matter(self):
        """Artifacts round-trip through JSON, where a year may arrive as a string."""
        mask = handed_mask(['BRA'], [2020], [['BRA', '2020']])
        assert list(mask) == [True]

    def test_an_empty_key_list_hands_over_nothing(self):
        mask = handed_mask(['BRA', 'ARG'], [2020, 2020], [])
        assert not mask.any()


class TestTheChannels:

    def test_a_model_made_perfect_on_handed_rows_reads_one_locally(self):
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        mask[:5] = True
        leaked = clean.copy()
        leaked[mask] = truth[mask]          # memorised exactly, nothing else
        result = decompose_fold(truth, clean, leaked, mask=mask)
        assert result['local'] == pytest.approx(1.0)
        assert result['global_uncontrolled'] == pytest.approx(0.0)

    def test_a_model_unchanged_reads_zero_on_both(self):
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        mask[:4] = True
        result = decompose_fold(truth, clean, clean, mask=mask)
        assert result['local'] == pytest.approx(0.0)
        assert result['global_uncontrolled'] == pytest.approx(0.0)
        assert result['aggregate'] == pytest.approx(0.0)

    def test_a_global_shift_shows_up_only_globally(self):
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        mask[:6] = True
        leaked = clean.copy()
        held = ~mask
        leaked[held] = truth[held] + (clean[held] - truth[held]) * 0.5
        result = decompose_fold(truth, clean, leaked, mask=mask)
        assert result['local'] == pytest.approx(0.0)
        assert result['global_uncontrolled'] == pytest.approx(0.75)

    def test_a_worse_held_out_fit_reads_negative(self):
        """Contamination can distort a model into generalising worse, and the
        measure has to be able to say so rather than flooring at zero."""
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        mask[:5] = True
        leaked = clean.copy()
        held = ~mask
        leaked[held] = truth[held] + (clean[held] - truth[held]) * 2.0
        result = decompose_fold(truth, clean, leaked, mask=mask)
        assert result['global_uncontrolled'] < 0

    def test_the_row_counts_are_reported(self):
        truth, clean = window(rows=30)
        mask = np.zeros(30, dtype=bool)
        mask[:7] = True
        result = decompose_fold(truth, clean, clean, mask=mask)
        assert result['rows_handed'] == 7 and result['rows_held'] == 23

    def test_a_thin_handed_partition_is_flagged(self):
        """Three rows is what a 5% dose on a 64-row window produces. Shown, not
        hidden: an average over three rows is not an average over sixty."""
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        mask[:3] = True
        assert decompose_fold(truth, clean, clean,
                              mask=mask)['thin_handed_partition'] is True
        mask[:9] = True
        assert decompose_fold(truth, clean, clean,
                              mask=mask)['thin_handed_partition'] is False

    def test_mismatched_vector_lengths_are_refused(self):
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        with pytest.raises(ValueError, match='lengths disagree'):
            decompose_fold(truth, clean, clean[:-1], mask=mask)


class TestTheControlArm:

    def test_without_it_the_global_channel_is_left_unattributed(self):
        """Calling an uncontrolled number `global` would let a sample-size effect
        be read as leakage."""
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        mask[:5] = True
        result = decompose_fold(truth, clean, clean, mask=mask)
        assert result['global'] is None
        assert result['sample_size_effect'] is None
        assert result['global_uncontrolled'] is not None

    def test_it_is_subtracted_from_the_held_out_improvement(self):
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        mask[:5] = True
        held = ~mask
        leaked = clean.copy()
        leaked[held] = truth[held] + (clean[held] - truth[held]) * 0.5   # 0.75
        control = clean.copy()
        control[held] = truth[held] + (clean[held] - truth[held]) * 0.8  # 0.36
        result = decompose_fold(truth, clean, leaked, mask=mask,
                               control=control)
        assert result['sample_size_effect'] == pytest.approx(0.36)
        assert result['global'] == pytest.approx(0.75 - 0.36)

    def test_a_control_that_explains_everything_leaves_nothing(self):
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        mask[:5] = True
        held = ~mask
        leaked = clean.copy()
        leaked[held] = truth[held] + (clean[held] - truth[held]) * 0.5
        result = decompose_fold(truth, clean, leaked, mask=mask,
                               control=leaked)
        assert result['global'] == pytest.approx(0.0)

    def test_a_control_of_the_wrong_length_is_refused(self):
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)
        with pytest.raises(ValueError, match='control arm has'):
            decompose_fold(truth, clean, clean, mask=mask, control=clean[:-2])


class TestTheIdentity:
    """The aggregate is the channels weighted by their share of the clean error.

    Exact, so it is an arithmetic check on the implementation rather than a
    plausibility argument.
    """

    def test_it_holds_on_a_constructed_fold(self):
        truth, clean = window(rows=40, seed=3)
        mask = np.zeros(40, dtype=bool)
        mask[:11] = True
        rng = np.random.default_rng(9)
        leaked = clean + rng.normal(size=40)
        assert check_identity(decompose_fold(truth, clean, leaked, mask=mask))

    def test_it_holds_when_one_channel_is_negative(self):
        truth, clean = window(rows=40, seed=4)
        mask = np.zeros(40, dtype=bool)
        mask[:8] = True
        leaked = clean.copy()
        leaked[~mask] = truth[~mask] + (clean[~mask] - truth[~mask]) * 3.0
        leaked[mask] = truth[mask]
        assert check_identity(decompose_fold(truth, clean, leaked, mask=mask))

    def test_it_is_false_when_a_channel_is_undefined(self):
        truth, clean = window()
        mask = np.zeros(len(truth), dtype=bool)   # nothing handed
        assert not check_identity(decompose_fold(truth, clean, clean, mask=mask))


class TestTheSummary:

    def _folds(self, count=9, seed=0):
        rng = np.random.default_rng(seed)
        made = []
        for index in range(count):
            truth, clean = window(rows=40, seed=seed + index)
            mask = np.zeros(40, dtype=bool)
            mask[:10] = True
            leaked = clean.copy()
            leaked[mask] = truth[mask]
            held = ~mask
            leaked[held] = truth[held] + (clean[held] - truth[held]) * (
                0.9 + 0.02 * rng.normal())
            control = clean.copy()
            control[held] = truth[held] + (clean[held] - truth[held]) * 0.98
            made.append(decompose_fold(truth, clean, leaked, mask=mask,
                                       control=control))
        return made

    def test_every_channel_gets_a_point_and_an_interval(self):
        summary = summarise(self._folds(), block=2, iters=2000)
        for channel in ('local', 'global', 'aggregate',
                        'sample_size_effect', 'global_uncontrolled'):
            assert np.isfinite(summary[channel]['point']), channel
            assert summary[channel]['inference']['block'] == 2

    def test_the_local_channel_recovers_the_constructed_truth(self):
        summary = summarise(self._folds(), block=2, iters=2000)
        assert summary['local']['point'] == pytest.approx(1.0, abs=1e-9)

    def test_it_reports_whether_the_identity_held_everywhere(self):
        summary = summarise(self._folds(), block=2, iters=1000)
        assert summary['coverage']['identity_holds'] is True
        assert summary['coverage']['folds'] == 9

    def test_it_counts_the_folds_with_a_thin_partition(self):
        folds = self._folds()
        folds[0]['thin_handed_partition'] = True
        summary = summarise(folds, block=2, iters=500)
        assert summary['coverage']['folds_with_thin_handed_partition'] == 1

    def test_an_absent_control_does_not_break_the_summary(self):
        truth, clean = window(rows=40)
        mask = np.zeros(40, dtype=bool)
        mask[:10] = True
        folds = [decompose_fold(truth, clean, clean, mask=mask)
                 for _ in range(4)]
        summary = summarise(folds, block=2, iters=500)
        assert summary['global']['inference']['method'] == 'insufficient_data'
        assert np.isfinite(summary['aggregate']['point'])
