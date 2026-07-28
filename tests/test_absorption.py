#!/usr/bin/env python3
"""Absorption: does the instrument measure what it claims to?

The quantity is the fraction of a single handed answer a model keeps when that
answer is appended to the frame it is fitted on. It replaced a borrowed ordering
that did not transfer, and it carries the study's mechanistic claim, so the
instrument needs checking before its readings are used.

Two readings pin it down without any judgement being involved. An unbounded
decision tree must keep a training row exactly, so it must read one. A ridge with
a large penalty must shrink it away, so it must read near zero. Anything that
reports those two the same way is not measuring absorption.
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

from core.models.absorption import absorption_coefficient  # noqa
from core.models.ladder import LADDER, RUNGS  # noqa
from core.scientific_config import SCIENTIFIC_CONFIG  # noqa


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
        assert result['absorption'] == pytest.approx(1.0, abs=1e-9)
        assert result['absorption_sd'] == pytest.approx(0.0, abs=1e-9)

    def test_a_heavily_penalised_ridge_shrinks_it_away(self):
        X_fit, y_fit, X_eval, y_eval = frames()
        result = absorption_coefficient(make_heavy_ridge, X_fit, y_fit,
                                        X_eval, y_eval)
        assert abs(result['absorption']) < 0.05, (
            'a ridge with a 1e6 penalty cannot be moved by one row')

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
        assert (result['probes_used'] + result['probes_skipped']
                == SCIENTIFIC_CONFIG['in_context_models']['absorption_probes'])

    def test_it_does_not_ask_for_more_probes_than_there_are_rows(self):
        X_fit, y_fit, X_eval, y_eval = frames()
        result = absorption_coefficient(make_tree, X_fit, y_fit,
                                        X_eval.head(2), y_eval.head(2),
                                        probes=50)
        assert result['probes_used'] + result['probes_skipped'] == 2

    def test_the_same_seed_gives_the_same_probes(self):
        X_fit, y_fit, X_eval, y_eval = frames()
        one = absorption_coefficient(make_tree, X_fit, y_fit, X_eval, y_eval,
                                     seed=7)
        two = absorption_coefficient(make_tree, X_fit, y_fit, X_eval, y_eval,
                                     seed=7)
        assert one['per_probe'] == two['per_probe']

    def test_a_different_seed_draws_different_rows(self):
        X_fit, y_fit, X_eval, y_eval = frames()
        one = absorption_coefficient(make_heavy_ridge, X_fit, y_fit,
                                     X_eval, y_eval, seed=1)
        two = absorption_coefficient(make_heavy_ridge, X_fit, y_fit,
                                     X_eval, y_eval, seed=2)
        assert one['per_probe'] != two['per_probe']


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
        assert supplied['per_probe'] == pytest.approx(computed['per_probe'])

    def test_a_baseline_of_the_wrong_length_is_refused(self):
        """Silently accepting it would redefine the quantity being measured."""
        X_fit, y_fit, X_eval, y_eval = frames()
        with pytest.raises(ValueError, match='redefine what is being measured'):
            absorption_coefficient(make_tree, X_fit, y_fit, X_eval, y_eval,
                                   baseline=np.zeros(len(X_eval) - 1))


class TestDegenerateProbes:

    def test_a_row_already_predicted_exactly_is_skipped_not_averaged(self):
        """Dividing by a residual of zero would produce an infinity, and an
        infinity in a mean silently destroys the estimate."""
        rows = 40
        X_fit = pd.DataFrame({'f0': np.arange(rows, dtype=float)})
        y_fit = pd.Series(np.arange(rows, dtype=float))
        result = absorption_coefficient(
            make_tree, X_fit, y_fit, X_fit.head(5), y_fit.head(5), probes=5)
        assert result['probes_used'] == 0
        assert result['probes_skipped'] == 5
        assert np.isnan(result['absorption'])
        assert 'note' in result


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
