#!/usr/bin/env python3
"""Absorption: does the instrument measure what it claims to?

The quantity is how much of the squared error on a few handed answers disappears
when those answers are appended to the frame the model is fitted on. It replaced a
borrowed ordering that did not transfer, and it carries the study's mechanistic
claim, so the instrument needs checking before its readings are used.

Three checks pin it down with no judgement involved. An unbounded decision tree
must keep a training row exactly, so it must read one. A ridge with a large
penalty must shrink it away, so it must read near zero. And k-nearest neighbours
has an analytic value, (2k-1)/k^2, which the instrument was not fitted to and
should reproduce.

Also here: that it is a ratio of sums. The first version averaged per-row ratios
with a residual in each denominator and read 6.87 for one configuration and -0.23
with a standard deviation of 2.13 for another, for a quantity bounded by
construction.
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

from core.models.absorption import (  # noqa
    absorption_coefficient, knn_expected_absorption)
from core.models.ladder import LADDER, RUNGS  # noqa
from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG  # noqa


def frames(rows=90, seed=0):
    """A fit frame and an evaluation frame with genuine signal and noise.

    Noise matters: with a target the models fit exactly, every probe row has a
    residual near zero and the ratio divides by nothing.
    """
    rng = np.random.default_rng(seed)
    X_fit = pd.DataFrame(rng.normal(size=(rows, 4)),
                         columns=[f'f{i}' for i in range(4)])
    y_fit = pd.Series(X_fit['f0'] * 2 - X_fit['f1']
                      + rng.normal(scale=1.0, size=rows))
    X_eval = pd.DataFrame(rng.normal(size=(30, 4)),
                          columns=[f'f{i}' for i in range(4)])
    y_eval = pd.Series(X_eval['f0'] * 2 - X_eval['f1']
                       + rng.normal(scale=1.0, size=30))
    return X_fit, y_fit, X_eval, y_eval


def make_tree():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(min_samples_leaf=1, random_state=0)


def make_heavy_ridge():
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1e6)


class TestTheInstrument:

    def test_an_unbounded_tree_keeps_the_row_exactly(self):
        """A leaf holding one row predicts that row's label. Absorption is one.

        The check that would catch a broken measurement: any sign error, any
        swapped before and after, any misindexed probe row moves this off one.
        """
        X_fit, y_fit, X_eval, y_eval = frames()
        result = absorption_coefficient(make_tree, X_fit, y_fit, X_eval, y_eval)
        assert result['absorption'] == pytest.approx(1.0, abs=1e-12)
        assert result['error_after'] == pytest.approx(0.0, abs=1e-18)

    def test_a_heavily_penalised_ridge_reads_the_global_floor(self):
        """It cannot memorise, so what it reads is the other component.

        A penalty of 1e6 drives the coefficients to nothing, so the model predicts
        its intercept. Appending probe rows moves that intercept, and the
        improvement on the probe rows is exactly the part of the reading that is
        not memorisation. So this is not a test that the reading is zero -- an
        earlier version asserted that and passed only because it used five probe
        rows instead of twelve. It is a test that the floor is small next to what
        a memorising model reads.
        """
        X_fit, y_fit, X_eval, y_eval = frames()
        floor = absorption_coefficient(make_heavy_ridge, X_fit, y_fit,
                                       X_eval, y_eval)['absorption']
        memorising = absorption_coefficient(make_tree, X_fit, y_fit,
                                            X_eval, y_eval)['absorption']
        assert abs(floor) < 0.35, f'the global floor is not small: {floor}'
        assert floor < memorising / 2.0

    def test_the_probe_dose_is_recorded(self):
        """Absorption read at a 19% dose is not absorption at the single-row
        margin, and the count alone does not say which it is."""
        X_fit, y_fit, X_eval, y_eval = frames()
        result = absorption_coefficient(make_tree, X_fit, y_fit, X_eval, y_eval,
                                        probes=6)
        assert result['probe_dose'] == pytest.approx(6 / len(X_eval))

    def test_the_two_are_ordered(self):
        """The only property the axis has to have to be an axis."""
        X_fit, y_fit, X_eval, y_eval = frames()
        tree = absorption_coefficient(make_tree, X_fit, y_fit, X_eval, y_eval)
        ridge = absorption_coefficient(make_heavy_ridge, X_fit, y_fit,
                                       X_eval, y_eval)
        assert tree['absorption'] > ridge['absorption']

    def test_it_uses_the_configured_number_of_probes(self):
        X_fit, y_fit, X_eval, y_eval = frames()
        result = absorption_coefficient(make_tree, X_fit, y_fit, X_eval, y_eval)
        assert (result['probes_used']
                == SCIENTIFIC_CONFIG['in_context_models']['absorption_probes'])

    def test_it_does_not_ask_for_more_probes_than_there_are_rows(self):
        X_fit, y_fit, X_eval, y_eval = frames()
        result = absorption_coefficient(make_tree, X_fit, y_fit,
                                        X_eval.head(2), y_eval.head(2),
                                        probes=50)
        assert result['probes_used'] == 2

    def test_the_same_seed_gives_the_same_reading(self):
        X_fit, y_fit, X_eval, y_eval = frames()
        one = absorption_coefficient(make_heavy_ridge, X_fit, y_fit,
                                     X_eval, y_eval, seed=7)
        two = absorption_coefficient(make_heavy_ridge, X_fit, y_fit,
                                     X_eval, y_eval, seed=7)
        assert one['absorption'] == two['absorption']

    def test_a_different_seed_draws_different_rows(self):
        X_fit, y_fit, X_eval, y_eval = frames()
        one = absorption_coefficient(make_heavy_ridge, X_fit, y_fit,
                                     X_eval, y_eval, seed=1)
        two = absorption_coefficient(make_heavy_ridge, X_fit, y_fit,
                                     X_eval, y_eval, seed=2)
        assert one['error_before'] != two['error_before']

    def test_it_is_a_ratio_of_sums_and_not_a_mean_of_ratios(self):
        """The defect this replaced, checked on the shape that exposed it.

        A mean of per-row ratios has a residual in each denominator, so a row the
        model nearly fits sends one term to infinity and the average with it. This
        construction gives one probe row an almost-zero residual and several
        ordinary ones; a mean of ratios blows up here, a ratio of sums does not.
        """
        rng = np.random.default_rng(0)
        X_fit = pd.DataFrame(rng.normal(size=(80, 2)), columns=['a', 'b'])
        y_fit = pd.Series(X_fit['a'] + rng.normal(scale=0.5, size=80))
        X_eval = pd.DataFrame(rng.normal(size=(6, 2)), columns=['a', 'b'])
        y_eval = pd.Series(X_eval['a'] + rng.normal(scale=0.5, size=6))

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_fit, y_fit)
        # One evaluation row given exactly the value the model predicts for it.
        y_eval.iloc[0] = float(model.predict(X_eval.iloc[[0]])[0])

        result = absorption_coefficient(lambda: Ridge(alpha=1.0), X_fit, y_fit,
                                        X_eval, y_eval, probes=6)
        assert np.isfinite(result['absorption'])
        assert -1.0 < result['absorption'] < 1.0001, result


class TestTheBaselineShortcut:

    def test_a_supplied_baseline_matches_a_computed_one(self):
        """Callers with clean predictions in hand should not pay for the fit twice."""
        X_fit, y_fit, X_eval, y_eval = frames()
        model = make_tree()
        model.fit(X_fit, y_fit)
        supplied = absorption_coefficient(
            make_tree, X_fit, y_fit, X_eval, y_eval,
            baseline=model.predict(X_eval))
        computed = absorption_coefficient(make_tree, X_fit, y_fit, X_eval, y_eval)
        assert supplied['absorption'] == pytest.approx(computed['absorption'])
        assert supplied['error_before'] == pytest.approx(computed['error_before'])

    def test_a_baseline_of_the_wrong_length_is_refused(self):
        """Silently accepting it would redefine the quantity being measured."""
        X_fit, y_fit, X_eval, y_eval = frames()
        with pytest.raises(ValueError, match='redefine what is being measured'):
            absorption_coefficient(make_tree, X_fit, y_fit, X_eval, y_eval,
                                   baseline=np.zeros(len(X_eval) - 1))


class TestDegenerateProbes:

    def test_no_error_to_remove_is_reported_rather_than_divided(self):
        """A model already exact on every probe row leaves nothing to absorb.

        The denominator is the total squared error on those rows, so this is the
        only way it can vanish -- which is the point of summing before dividing.
        """
        rows = 40
        X_fit = pd.DataFrame({'f0': np.arange(rows, dtype=float)})
        y_fit = pd.Series(np.arange(rows, dtype=float))
        result = absorption_coefficient(
            make_tree, X_fit, y_fit, X_fit.head(5), y_fit.head(5), probes=5)
        assert np.isnan(result['absorption'])
        assert result['error_before'] < 1e-12
        assert 'note' in result


class TestTheClosedFormCalibration:
    """k-nearest neighbours has an analytic absorption, so the instrument can be
    checked against a prediction it was not fitted to."""

    def test_one_neighbour_absorbs_the_answer_entirely(self):
        """Exact, not approximate: the duplicate is the only neighbour."""
        from sklearn.neighbors import KNeighborsRegressor
        X_fit, y_fit, X_eval, y_eval = frames(rows=120, seed=1)
        result = absorption_coefficient(
            lambda: KNeighborsRegressor(n_neighbors=1),
            X_fit, y_fit, X_eval, y_eval, probes=8)
        assert result['absorption'] == pytest.approx(1.0, abs=1e-12)
        assert knn_expected_absorption(1) == 1.0

    def test_the_curve_matches_within_tolerance(self):
        from sklearn.neighbors import KNeighborsRegressor
        X_fit, y_fit, X_eval, y_eval = frames(rows=300, seed=2)
        for k in (2, 3, 5, 10):
            measured = absorption_coefficient(
                lambda k=k: KNeighborsRegressor(n_neighbors=k),
                X_fit, y_fit, X_eval, y_eval, probes=20)['absorption']
            expected = knn_expected_absorption(k)
            assert abs(measured - expected) < 0.12, (
                f'k={k}: expected about {expected:.4f}, measured {measured:.4f}')

    def test_the_prediction_decreases_in_k(self):
        values = [knn_expected_absorption(k) for k in (1, 2, 3, 5, 10, 20)]
        assert values == sorted(values, reverse=True)

    def test_a_nonsensical_k_is_refused(self):
        with pytest.raises(ValueError, match='at least one'):
            knn_expected_absorption(0)


class TestTheLadderRungsSpread:

    def test_the_rungs_do_not_all_read_the_same(self):
        """An axis on which every rung sits at one point is not an axis.

        This is what disqualified the borrowed ordering, so the replacement has
        to be checked for the same failure rather than assumed free of it.
        """
        X_fit, y_fit, X_eval, y_eval = frames(rows=150, seed=3)
        readings = {}
        for rung in LADDER:
            readings[rung.name] = absorption_coefficient(
                rung.make, X_fit, y_fit, X_eval, y_eval)['absorption']
        values = np.array([v for v in readings.values() if np.isfinite(v)])
        assert len(values) >= 4
        assert values.max() - values.min() > 0.3, readings

    def test_the_regularised_rung_keeps_least(self):
        X_fit, y_fit, X_eval, y_eval = frames(rows=150, seed=3)
        ridge = absorption_coefficient(RUNGS['ladder_ridge'].make,
                                       X_fit, y_fit, X_eval, y_eval)['absorption']
        tree = absorption_coefficient(RUNGS['ladder_decision_tree'].make,
                                      X_fit, y_fit, X_eval, y_eval)['absorption']
        assert ridge < tree


class TestReplicatesRemoveTheRowOrderDependence:
    """Twelve probe rows are drawn by position, so the reading depends on which twelve.

    It depended on it enough to move the headline number fivefold. When `prepared`
    began sorting the evaluation frame by year -- a change made so the context cap
    could live in the estimator -- the same panel, the same count and the same seed
    went from a largest closed-form gap of 0.0108 to 0.0577. Nothing about the panel
    or the estimator changed; only which rows the generator landed on. Replicates
    average that away at the same dose, and pooling them is a ratio of sums because
    a residual in the denominator has diverged three times in this repository.
    """

    def _frames(self, rows=200, cols=4, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(rows, cols)))
        y = pd.Series(X.iloc[:, 0] * 2.0 + rng.normal(scale=0.5, size=rows))
        Xe = pd.DataFrame(rng.normal(size=(40, cols)))
        ye = pd.Series(Xe.iloc[:, 0] * 2.0 + rng.normal(scale=0.5, size=40))
        return X, y, Xe, ye

    def test_one_replicate_is_the_single_draw_reading(self):
        """Bit for bit against the pre-replicates path, so every number read before
        the knob existed can still be reproduced by turning it to 1.

        Compared against a hand-rolled single draw rather than against the configured
        default, because the default is no longer 1 -- and an assertion pinned to the
        default would silently stop testing this the moment the default moved, which
        is exactly what happened to the first version of it.
        """
        from sklearn.neighbors import KNeighborsRegressor
        X, y, Xe, ye = self._frames()
        make = lambda: KNeighborsRegressor(n_neighbors=3)
        seed, count = RANDOM_SEED, 12

        one = absorption_coefficient(make, X, y, Xe, ye, probes=count, seed=seed,
                                     replicates=1)

        rng = np.random.default_rng(seed)
        picked = np.sort(rng.choice(len(Xe), size=count, replace=False))
        truth = np.asarray(ye.iloc[picked], dtype=float)
        clean = make().fit(X, y)
        before = float(np.sum((truth - np.asarray(clean.predict(Xe), dtype=float)[picked]) ** 2))
        widened = make().fit(pd.concat([X, Xe.iloc[picked]], ignore_index=True),
                             pd.concat([y, ye.iloc[picked]], ignore_index=True))
        after = float(np.sum((truth - np.asarray(widened.predict(Xe.iloc[picked]),
                                                dtype=float)) ** 2))

        assert one['absorption'] == pytest.approx(1.0 - after / before)

    def test_the_configured_default_is_five_and_is_not_one(self):
        """Pinned because the default is the reading the paper reports.

        Five, from measurement: across the two row orderings of the World Bank panel
        the gap between the two calibration readings is 0.0469 at one replicate and
        0.005-0.015 from five upward, where what is left is the size of the noise in
        that gap statistic itself. More costs linearly on the in-context side for
        nothing measurable.
        """
        from sklearn.neighbors import KNeighborsRegressor
        assert SCIENTIFIC_CONFIG['in_context_models']['absorption_replicates'] == 5

        X, y, Xe, ye = self._frames()
        make = lambda: KNeighborsRegressor(n_neighbors=3)
        default = absorption_coefficient(make, X, y, Xe, ye, probes=12)

        assert default['replicates'] == 5
        assert default['absorption'] != pytest.approx(
            absorption_coefficient(make, X, y, Xe, ye, probes=12,
                                   replicates=1)['absorption'], abs=1e-12)

    def test_replicates_do_not_change_the_dose(self):
        """The reason to prefer replicates over more probes: the dose is the estimand."""
        from sklearn.neighbors import KNeighborsRegressor
        X, y, Xe, ye = self._frames()
        make = lambda: KNeighborsRegressor(n_neighbors=3)

        one = absorption_coefficient(make, X, y, Xe, ye, probes=12, replicates=1)
        many = absorption_coefficient(make, X, y, Xe, ye, probes=12, replicates=8)

        assert many['probe_dose'] == one['probe_dose'] == 12 / 40
        assert many['replicates'] == 8
        assert many['probes_used'] == 12

    def test_pooling_is_a_ratio_of_sums_not_a_mean_of_ratios(self):
        """The distinction that has bitten this repository three times.

        Reproduced here by pooling by hand: eight replicates drawn from one generator,
        their squared errors summed, one division at the end. A mean of the eight
        per-replicate absorptions is a different number, and the test says so rather
        than trusting that it is close.
        """
        from sklearn.neighbors import KNeighborsRegressor
        X, y, Xe, ye = self._frames()
        make = lambda: KNeighborsRegressor(n_neighbors=3)
        seed, count, reps = 12345, 12, 8

        pooled = absorption_coefficient(make, X, y, Xe, ye, probes=count,
                                        seed=seed, replicates=reps)

        rng = np.random.default_rng(seed)
        before_sum = after_sum = 0.0
        per_replicate = []
        clean = make().fit(X, y)
        base = np.asarray(clean.predict(Xe), dtype=float)
        for _ in range(reps):
            picked = np.sort(rng.choice(len(Xe), size=count, replace=False))
            truth = np.asarray(ye.iloc[picked], dtype=float)
            b = float(np.sum((truth - base[picked]) ** 2))
            widened = make().fit(
                pd.concat([X, Xe.iloc[picked]], ignore_index=True),
                pd.concat([y, ye.iloc[picked]], ignore_index=True))
            a = float(np.sum((truth - np.asarray(widened.predict(Xe.iloc[picked]),
                                                 dtype=float)) ** 2))
            before_sum += b
            after_sum += a
            per_replicate.append(1.0 - a / b)

        assert pooled['absorption'] == pytest.approx(1.0 - after_sum / before_sum)
        assert pooled['error_before'] == pytest.approx(before_sum)
        # Stated, not assumed: the two aggregations are genuinely different numbers.
        assert pooled['absorption'] != pytest.approx(float(np.mean(per_replicate)),
                                                     abs=1e-9)

    def test_replicates_shrink_the_spread_across_row_orderings(self):
        """The property the fix exists for, measured rather than argued.

        Two orderings of the same evaluation frame are two different probe draws.
        With one replicate the readings differ; averaging over replicates brings them
        together, because each ordering then samples much more of the window.
        """
        from sklearn.neighbors import KNeighborsRegressor
        X, y, Xe, ye = self._frames()
        make = lambda: KNeighborsRegressor(n_neighbors=3)
        order = np.random.default_rng(7).permutation(len(Xe))
        Xs = Xe.iloc[order].reset_index(drop=True)
        ys = ye.iloc[order].reset_index(drop=True)

        def spread(reps):
            a = absorption_coefficient(make, X, y, Xe, ye, probes=12,
                                       replicates=reps)['absorption']
            b = absorption_coefficient(make, X, y, Xs, ys, probes=12,
                                       replicates=reps)['absorption']
            return abs(a - b)

        assert spread(40) < spread(1)


class TestFractionMakesTwoPanelsComparable:
    """A fixed probe COUNT makes the perturbation 12/n, and 1/n is the answer.

    Absorption appends a fixed twelve rows, so the perturbation as a share of the
    training frame is 12/n: 3.13% on the World Bank at n about 384 against 0.12% on
    the larger panel, whose models read 10,000 rows under the context cap. Twenty-six
    times apart, and any quantity that declines with n is then reporting the
    perturbation rather than the model. Measured on the two panels with the share
    matched at 3.13%, the ridge goes from n^-0.75 to n^-0.20 and the random forest
    from n^-0.43 to n^-0.15, while 1-NN and the unbounded tree stay at exactly 1.0000.
    That is what structural invariance looks like, and it is the half of the n-scaling
    story that survives.
    """

    def _frames(self, rows, cols=4, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(rows, cols)))
        y = pd.Series(X.iloc[:, 0] * 2.0 + rng.normal(scale=0.5, size=rows))
        Xe = pd.DataFrame(rng.normal(size=(60, cols)))
        ye = pd.Series(Xe.iloc[:, 0] * 2.0 + rng.normal(scale=0.5, size=60))
        return X, y, Xe, ye

    def test_the_count_follows_the_training_frame(self):
        from sklearn.linear_model import Ridge
        make = lambda: Ridge(alpha=1.0)
        small = absorption_coefficient(make, *self._frames(200), fraction=0.05)
        large = absorption_coefficient(make, *self._frames(400), fraction=0.05)

        assert small['probes_used'] == 10
        assert large['probes_used'] == 20
        assert small['perturbation_share'] == pytest.approx(0.05)
        assert large['perturbation_share'] == pytest.approx(0.05)

    def test_the_share_is_recorded_even_when_a_count_was_given(self):
        """The quantity that makes panels comparable cannot be left to the reader."""
        from sklearn.linear_model import Ridge
        record = absorption_coefficient(lambda: Ridge(alpha=1.0),
                                        *self._frames(384), probes=12)

        assert record['probes_used'] == 12
        assert record['perturbation_share'] == pytest.approx(12 / 384)
        assert record['fraction_requested'] is None

    @pytest.mark.parametrize('bad', [0.0, -0.1, 1.5])
    def test_a_share_outside_the_unit_interval_is_refused(self, bad):
        from sklearn.linear_model import Ridge
        with pytest.raises(ValueError, match='fraction must be in'):
            absorption_coefficient(lambda: Ridge(alpha=1.0),
                                   *self._frames(200), fraction=bad)

    def test_at_a_matched_share_the_ridge_stops_declining_with_n(self):
        """The finding itself, at small scale: matched share removes most of the drop.

        A fixed count shrinks the perturbation as n grows, so the ridge appears to
        dilute. Holding the share fixed removes that, and what is left is much
        smaller than what the fixed count reports.
        """
        from sklearn.linear_model import Ridge
        make = lambda: Ridge(alpha=1.0)
        small, large = self._frames(200), self._frames(800)

        by_count = (absorption_coefficient(make, *small, probes=12)['absorption'],
                    absorption_coefficient(make, *large, probes=12)['absorption'])
        by_share = (absorption_coefficient(make, *small, fraction=0.06)['absorption'],
                    absorption_coefficient(make, *large, fraction=0.06)['absorption'])

        assert by_count[1] < by_count[0], 'a fixed count should show the decline'
        assert abs(by_share[1] - by_share[0]) < abs(by_count[1] - by_count[0])


class TestSingleProbeMode:
    """Batch and single modes must differ in the appending and in nothing else.

    The kNN closed form is derived for one appended copy; the instrument appends
    twelve. The single mode exists to measure that difference, which only works
    if both modes read the same probe rows -- so they share the generator draw
    -- and agree exactly wherever the batch cannot matter.
    """

    @staticmethod
    def _panel(rows=60, seed=5):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(rows, 3)), columns=list('abc'))
        y = pd.Series(X['a'] * 2 + rng.normal(0, 0.1, rows))
        return X.iloc[:40], y.iloc[:40], X.iloc[40:].reset_index(drop=True), \
            y.iloc[40:].reset_index(drop=True)

    def test_shared_draw_means_identical_denominators(self):
        from sklearn.linear_model import Ridge
        from core.models.absorption import absorption_coefficient
        Xf, yf, Xe, ye = self._panel()
        kw = dict(probes=6, seed=11, replicates=3)
        b = absorption_coefficient(lambda: Ridge(alpha=1.0), Xf, yf, Xe, ye, **kw)
        s = absorption_coefficient(lambda: Ridge(alpha=1.0), Xf, yf, Xe, ye,
                                   batch=False, **kw)
        assert b['error_before'] == s['error_before'], (
            'different denominators mean different probe rows were drawn, and '
            'then batch minus single measures the draw, not the appending')

    def test_one_nearest_neighbour_reads_one_in_both_modes(self):
        from sklearn.neighbors import KNeighborsRegressor
        from core.models.absorption import absorption_coefficient
        Xf, yf, Xe, ye = self._panel()
        kw = dict(probes=6, seed=11, replicates=2)
        for batch in (True, False):
            got = absorption_coefficient(
                lambda: KNeighborsRegressor(n_neighbors=1), Xf, yf, Xe, ye,
                batch=batch, **kw)
            assert got['absorption'] == 1.0, (
                'a duplicated query is its own nearest neighbour regardless of '
                'how many other probes rode along; anything else is a bug in '
                'the appending, not a finding')

    def test_a_single_probe_count_makes_the_modes_coincide(self):
        from sklearn.linear_model import Ridge
        from core.models.absorption import absorption_coefficient
        Xf, yf, Xe, ye = self._panel()
        kw = dict(probes=1, seed=7, replicates=4)
        b = absorption_coefficient(lambda: Ridge(alpha=1.0), Xf, yf, Xe, ye, **kw)
        s = absorption_coefficient(lambda: Ridge(alpha=1.0), Xf, yf, Xe, ye,
                                   batch=False, **kw)
        assert abs(b['absorption'] - s['absorption']) < 1e-12, (
            'with one probe per replicate there is no batch to differ by')
