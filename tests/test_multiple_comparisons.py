#!/usr/bin/env python3
"""Benjamini-Hochberg adjustment.

The previous implementation enforced monotonicity in the order the tests were
listed rather than in the order of the sorted p-values. That produced adjusted
values below the raw ones -- which BH cannot produce -- in 5 of 15 World Bank
rows and 2 of 15 INEP rows of the published artifacts, in one case turning
t_p = 2.9e-11 into 8.1e-22.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from statistical_validation.effect_analysis import benjamini_hochberg


class TestProperties:
    """Properties BH satisfies by construction."""

    @pytest.mark.parametrize('seed', range(8))
    def test_adjusted_is_never_below_raw(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.uniform(1e-12, 1.0, 15)
        adj = np.array(benjamini_hochberg(list(p)))
        assert np.all(adj >= p - 1e-12), (
            f"adjusted below raw at {np.where(adj < p - 1e-12)[0].tolist()}"
        )

    @pytest.mark.parametrize('seed', range(8))
    def test_never_exceeds_one(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.5, 1.0, 12)
        assert max(benjamini_hochberg(list(p))) <= 1.0

    @pytest.mark.parametrize('seed', range(8))
    def test_order_is_preserved(self, seed):
        """A smaller raw p-value cannot receive a larger adjusted value."""
        rng = np.random.default_rng(seed)
        p = rng.uniform(1e-6, 1.0, 12)
        adj = np.array(benjamini_hochberg(list(p)))
        order = np.argsort(p)
        assert np.all(np.diff(adj[order]) >= -1e-12)

    def test_result_does_not_depend_on_input_order(self):
        """The regression: the adjustment depended on how tests were listed."""
        p = [0.001, 0.5, 0.02, 0.9, 0.04, 0.31, 0.007, 0.12]
        adj = dict(zip(p, benjamini_hochberg(p)))
        shuffled = list(reversed(p))
        adj_shuffled = dict(zip(shuffled, benjamini_hochberg(shuffled)))
        for key in adj:
            assert adj[key] == pytest.approx(adj_shuffled[key])


class TestAgreesWithReference:

    @pytest.mark.parametrize('seed', range(8))
    def test_matches_scipy(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.uniform(1e-9, 1.0, 15)
        assert np.allclose(benjamini_hochberg(list(p)),
                           sps.false_discovery_control(p, method='bh'))

    def test_matches_scipy_on_the_published_family_size(self):
        """15 tests: 3 pairs by 5 stages."""
        rng = np.random.default_rng(99)
        p = np.concatenate([rng.uniform(1e-12, 1e-7, 8),
                            rng.uniform(0.01, 0.9, 7)])
        assert np.allclose(benjamini_hochberg(list(p)),
                           sps.false_discovery_control(p, method='bh'))


class TestMissingPValues:
    """A test that produced no p-value is not part of the family."""

    def test_nan_stays_nan(self):
        adj = benjamini_hochberg([0.01, float('nan'), 0.5])
        assert np.isnan(adj[1])
        assert not np.isnan(adj[0]) and not np.isnan(adj[2])

    def test_family_size_excludes_missing(self):
        """Adjustment of the valid entries matches a family without the gap."""
        with_gap = benjamini_hochberg([0.01, float('nan'), 0.5, 0.2])
        without = benjamini_hochberg([0.01, 0.5, 0.2])
        assert [with_gap[0], with_gap[2], with_gap[3]] == pytest.approx(without)

    def test_all_missing_yields_all_nan(self):
        assert all(np.isnan(v) for v in benjamini_hochberg([float('nan')] * 3))

    def test_empty_family(self):
        assert benjamini_hochberg([]) == []
