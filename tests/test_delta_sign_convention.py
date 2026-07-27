#!/usr/bin/env python3
"""The paired effect is A minus B, on both paths of the module.

`_paired_deltas_for_metric` computed `vb - va` while everything else assumes
A−B: the module docstring, `_decision_equivalence`, and `paradigm_pairs`, which
promises "the effect of a pair is measured as A minus B". The latency path
already used log(A/B). The two branches of the same module disagreed, which on
its own proves one of them was wrong.

Consequence: the `advantage` field named the **worse** paradigm on all three
predictive metrics, and the same delta feeds `bootstrap_sensitivity`.

The interesting case is that r2 improves upwards and mase/wape improve
downwards: a test covering only r2 would pass with the sign inverted on half
the metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from statistical_validation.equivalence_estimation import (
    HIGHER_IS_BETTER, _advantage, bootstrap_ci, _decision_equivalence,
    _paired_deltas_for_metric)

# A is better on all three: higher r2, lower mase and wape.
BETTER = {'r2': 0.9, 'mase': 0.5, 'wape': 0.1}
WORSE = {'r2': 0.5, 'mase': 1.5, 'wape': 0.4}
SESOI = {'r2': 0.01, 'mase': 0.05, 'wape': 0.05}


def _pairs(a_values, b_values, n=9):
    return {'archA': {i: dict(a_values) for i in range(n)},
            'archB': {i: dict(b_values) for i in range(n)}}


def _decide(metric, pairs, arch_a='archA', arch_b='archB'):
    deltas = _paired_deltas_for_metric(pairs, metric, arch_a, arch_b)
    _, (low, high), _ = bootstrap_ci(deltas, iters=400)
    decision = _decision_equivalence(low, high, SESOI[metric])
    return decision, _advantage(decision, metric, arch_a, arch_b)


class TestSignConvention:

    @pytest.mark.parametrize('metric', ['r2', 'mase', 'wape'])
    def test_the_delta_is_a_minus_b(self, metric):
        pairs = _pairs(BETTER, WORSE)
        deltas = _paired_deltas_for_metric(pairs, metric, 'archA', 'archB')
        expected = BETTER[metric] - WORSE[metric]
        assert deltas[0] == pytest.approx(expected), (
            f'{metric}: delta {deltas[0]} is not A−B ({expected})'
        )

    def test_the_latency_branch_uses_the_same_order(self):
        """log(A/B) is positive when A is larger, like A−B."""
        source = (_SRC / 'statistical_validation'
                  / 'equivalence_estimation.py').read_text()
        assert 'np.log(x[mask] / y[mask])' in source
        index = source.index('np.log(x[mask] / y[mask])')
        window = source[max(0, index - 400):index]
        assert 'vals[arch_a]' in window and 'vals[arch_b]' in window


class TestAdvantageNamesTheBetterParadigm:

    @pytest.mark.parametrize('metric', ['r2', 'mase', 'wape'])
    def test_the_better_paradigm_is_named(self, metric):
        """It was inverted on 3 out of 3."""
        _, advantage = _decide(metric, _pairs(BETTER, WORSE))
        assert advantage == 'archA', (
            f'{metric}: named {advantage}, but archA is better'
        )

    @pytest.mark.parametrize('metric', ['r2', 'mase', 'wape'])
    def test_swapping_the_data_swaps_the_winner(self, metric):
        _, advantage = _decide(metric, _pairs(WORSE, BETTER))
        assert advantage == 'archB'

    @pytest.mark.parametrize('metric', ['r2', 'mase', 'wape'])
    def test_swapping_the_pair_order_keeps_the_winner(self, metric):
        """paradigm_pairs promises invariance to the order of the pair."""
        pairs = _pairs(BETTER, WORSE)
        _, direct = _decide(metric, pairs, 'archA', 'archB')
        _, reversed_order = _decide(metric, pairs, 'archB', 'archA')
        assert direct == reversed_order == 'archA'

    def test_metrics_with_opposite_directions_agree_on_the_winner(self):
        """The point of the field: r2 goes up, mase and wape go down, the
        winner is the same."""
        pairs = _pairs(BETTER, WORSE)
        winners = {metric: _decide(metric, pairs)[1]
                   for metric in ('r2', 'mase', 'wape')}
        assert set(winners.values()) == {'archA'}, winners

    def test_the_decisions_do_differ_by_direction(self):
        """If the three decisions were the same, the invariance would be
        vacuous."""
        pairs = _pairs(BETTER, WORSE)
        decisions = {metric: _decide(metric, pairs)[0]
                     for metric in ('r2', 'mase', 'wape')}
        assert len(set(decisions.values())) > 1, decisions

    def test_equivalent_data_names_nobody(self):
        pairs = _pairs(BETTER, BETTER)
        for metric in ('r2', 'mase', 'wape'):
            decision, advantage = _decide(metric, pairs)
            assert decision == 'equivalent'
            assert advantage is None


class TestSensitivityInheritsTheFix:
    """bootstrap_sensitivity calls the same function over the same deltas."""

    def test_it_uses_the_shared_delta_function(self):
        source = (_SRC / 'statistical_validation'
                  / 'bootstrap_sensitivity.py').read_text()
        assert '_paired_deltas_for_metric' in source
        assert '_advantage' in source

    def test_a_grid_row_names_the_better_paradigm(self):
        from statistical_validation import bootstrap_sensitivity as module

        deltas = _paired_deltas_for_metric(_pairs(BETTER, WORSE), 'mase',
                                           'archA', 'archB')
        rows = module._sensitivity_grid({'archA_vs_archB_mase': deltas},
                                        [1.0], [400], 42)
        assert rows and rows[0]['advantage'] == 'archA', rows
