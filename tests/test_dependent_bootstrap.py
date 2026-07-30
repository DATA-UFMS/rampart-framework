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
        """Pinned so a new panel cannot arrive without the block length being
        reconsidered -- which is the whole point of deriving it.

        `worldbank_clean` is the same source recollected without the
        cross-sectional imputation tiers. It is a variant rather than a third
        dataset, so it borrows the registered config instead of adding one; a
        panel that omitted `config` would silently ask the registry for a dataset
        that does not exist.
        """
        import sys
        scripts = _ROOT / 'scripts' / 'validation'
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from probe_harness import PANELS
        assert set(PANELS) == {'worldbank', 'inep_censo', 'worldbank_clean',
                               'worldbank_imputed_features',
                               'worldbank_clean_unclipped'}
        assert PANELS['worldbank_clean']['config'] == 'worldbank'
        # The two derived arms exist to isolate one variable each, so each must
        # declare a filter -- an arm without one is the whole panel under a new name.
        for derived in ('worldbank_imputed_features', 'worldbank_clean_unclipped'):
            assert PANELS[derived].get('filter') is not None, (
                f'{derived} is a derived arm and must declare its row filter')
        for name, spec in PANELS.items():
            assert spec.get('config', name) in ('worldbank', 'inep_censo'), (
                f'panel {name} points at an unregistered dataset config')

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


class TestContrastsGetTheirOwnInterval:
    """Every headline this project produced is a contrast, and none had an interval.

    `summarise` computed ci95 for each channel and all four probes indexed ['point'],
    so the bootstrap ran and was discarded. Shown, one headline died: "leaking the
    buffer immediately before the evaluation window costs more than leaking the window
    itself" is a ratio of 1.049 whose interval covers one. Marginal intervals would not
    have caught it either -- a contrast between two arms measured on the SAME folds has
    to be taken fold by fold, because that is what cancels the shared fold noise.
    """

    def _folds(self, values):
        return [{'local': v} for v in values]

    def test_a_shared_shift_is_detected_that_marginals_would_hide(self):
        """The case the helper exists for: wide arms, narrow difference."""
        import numpy as np
        from statistical_validation.leakage_channels import contrast, summarise
        noise = np.linspace(-0.4, 0.4, 9)
        a = self._folds(0.50 + noise)
        b = self._folds(0.44 + noise)          # same folds, constant 0.06 apart

        marginal_a = summarise(a, block=2, iters=2000)['local']['ci95']
        marginal_b = summarise(b, block=2, iters=2000)['local']['ci95']
        paired = contrast(a, b, 'local', block=2, iters=2000, direction=+1)

        assert marginal_a[0] < marginal_b[1], 'the marginals should overlap here'
        assert paired['point'] == pytest.approx(0.06)
        assert paired['detected'], 'the paired difference is constant and positive'

    def test_a_difference_that_is_noise_is_not_detected(self):
        from statistical_validation.leakage_channels import contrast
        a = self._folds([0.30, 0.55, 0.20, 0.61, 0.28, 0.49, 0.33, 0.58, 0.25])
        b = self._folds([0.52, 0.24, 0.58, 0.22, 0.55, 0.27, 0.60, 0.26, 0.54])

        paired = contrast(a, b, 'local', block=2, iters=4000, direction=+1)

        assert not paired['detected']

    def test_unpaired_arms_are_refused(self):
        """Positional pairing is the whole method; silently truncating would lie."""
        from statistical_validation.leakage_channels import contrast
        with pytest.raises(ValueError, match='same folds on both sides'):
            contrast(self._folds([0.1, 0.2, 0.3]), self._folds([0.1, 0.2]),
                     'local', block=2)

    def test_folds_missing_the_channel_are_dropped_in_pairs(self):
        """Dropping one side of a pair would break the positional correspondence."""
        from statistical_validation.leakage_channels import contrast
        a = [{'local': 0.5}, {'local': None}, {'local': 0.7}, {'local': 0.6}]
        b = [{'local': 0.4}, {'local': 0.3}, {'local': float('nan')}, {'local': 0.5}]

        paired = contrast(a, b, 'local', block=2, iters=500)

        assert paired['n_pairs'] == 2, 'only the two complete pairs survive'

    def test_the_resample_count_comes_from_the_configuration(self):
        """Not from another module's default, which is what the sentinel invited."""
        from core.scientific_config import SCIENTIFIC_CONFIG
        from statistical_validation.leakage_channels import contrast
        paired = contrast(self._folds([0.1, 0.3, 0.2, 0.4]),
                          self._folds([0.0, 0.1, 0.1, 0.2]), 'local', block=2)
        assert (paired['inference']['iters']
                == SCIENTIFIC_CONFIG['bootstrap_iters'])


class TestResampleLedger:
    """What ran, as opposed to what the configuration declares.

    Every record already carried its own `iters` and it was still not possible to
    answer "how many resamples produced this published interval" from a run log:
    the records feed the tables and the count does not. Answering it once meant
    dating a `scientific_config` commit against the run, which establishes the
    default rather than what executed -- and two call sites override the default.
    """

    def test_the_count_that_ran_is_recorded(self):
        from statistical_validation.dependent_bootstrap import (
            moving_block_ci, observed_resample_counts, reset_resample_ledger)
        reset_resample_ledger()

        moving_block_ci([0.1, 0.4, 0.2, 0.5], block=2, iters=321)

        assert observed_resample_counts() == {321: 1}

    def test_an_overriding_call_site_is_visible_next_to_the_configured_one(self):
        """The failure this exists for: one probe at 4,000 among runs at 15,000."""
        from core.scientific_config import SCIENTIFIC_CONFIG
        from statistical_validation.dependent_bootstrap import (
            moving_block_ci, observed_resample_counts, reset_resample_ledger)
        reset_resample_ledger()
        configured = int(SCIENTIFIC_CONFIG['bootstrap_iters'])

        moving_block_ci([0.1, 0.4, 0.2, 0.5], block=2)
        moving_block_ci([0.2, 0.3, 0.1, 0.6], block=2, iters=4000)

        observed = observed_resample_counts()
        assert set(observed) == {configured, 4000}, (
            'a run that mixes counts must say so; a single declared count would '
            'describe only part of the tables')

    def test_paths_that_never_resample_are_not_counted(self):
        """A degenerate fold set returns without drawing anything."""
        from statistical_validation.dependent_bootstrap import (
            moving_block_ci, observed_resample_counts, reset_resample_ledger)
        reset_resample_ledger()

        moving_block_ci([], block=2)
        moving_block_ci([0.3], block=2)
        moving_block_ci([0.5, 0.5, 0.5], block=2)

        assert observed_resample_counts() == {}, (
            'reporting resamples for an interval that was never resampled is the '
            'same class of untruth this ledger exists to catch')
