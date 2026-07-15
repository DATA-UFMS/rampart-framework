#!/usr/bin/env python3
"""Which method produced each bootstrap interval.

Three methods can produce an interval: BCa, a percentile fallback, and the
degenerate case of zero variance. They are not interchangeable, so the method is
recorded alongside the interval. On the observed data BCa succeeds down to n=2,
which means the fallback path is rarely exercised -- hence the injected failure
below, so it is known to work when it is needed.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from statistical_validation import equivalence_estimation as ee


class TestMethodIsRecorded:

    def test_zero_variance_is_reported_as_degenerate(self):
        """The Delta = 0 case the paper reports as [0, 0]."""
        point, (lo, hi), method = ee._bootstrap_ci(np.zeros(9), iters=500)
        assert method == 'degenerate_zero_variance'
        assert (point, lo, hi) == (0.0, 0.0, 0.0)

    def test_degenerate_interval_is_exact_not_approximate(self):
        """Every resample of a constant sample has the same mean."""
        values = np.full(9, 0.37)
        point, (lo, hi), _ = ee._bootstrap_ci(values, iters=500)
        assert lo == hi == point == pytest.approx(0.37)

    def test_ordinary_sample_uses_bca(self):
        rng = np.random.default_rng(11)
        values = rng.normal(0.02, 0.01, 9)
        point, (lo, hi), method = ee._bootstrap_ci(values, iters=2000)
        assert method == 'bca'
        assert lo < point < hi

    @pytest.mark.parametrize('values', [np.array([]), np.full(5, np.nan)])
    def test_absent_data_is_reported_as_insufficient(self, values):
        point, (lo, hi), method = ee._bootstrap_ci(values, iters=500)
        assert method == 'insufficient_data'
        assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)

    def test_fallback_is_used_and_named_when_bca_fails(self, monkeypatch):
        import scipy.stats

        def explode(*args, **kwargs):
            raise RuntimeError('BCa unavailable')

        monkeypatch.setattr(scipy.stats, 'bootstrap', explode)
        rng = np.random.default_rng(3)
        values = rng.normal(0.02, 0.01, 9)
        point, (lo, hi), method = ee._bootstrap_ci(values, iters=2000)
        assert method == 'percentile_fallback:RuntimeError'
        assert lo < point < hi

    def test_non_finite_bca_interval_falls_back(self, monkeypatch):
        """An infinite endpoint is not a usable interval."""
        import scipy.stats

        class _Result:
            confidence_interval = type(
                'CI', (), {'low': float('-inf'), 'high': 1.0})()

        monkeypatch.setattr(scipy.stats, 'bootstrap',
                            lambda *a, **k: _Result())
        rng = np.random.default_rng(5)
        values = rng.normal(0.02, 0.01, 9)
        _, (lo, hi), method = ee._bootstrap_ci(values, iters=2000)
        assert method.startswith('percentile_fallback:')
        assert np.isfinite(lo) and np.isfinite(hi)


class TestMethodReachesTheArtifact:
    """An interval recorded without its method is unattributable."""

    @staticmethod
    def _string_keys(node):
        return {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}

    def test_every_reported_interval_is_attributable(self):
        """A nested dict may inherit the method from the dict that holds it.

        'ci95_pct' is the log-ratio interval rescaled to percent, so it is the
        same interval and the same method.
        """
        source = (_ROOT / 'src' / 'statistical_validation'
                  / 'equivalence_estimation.py').read_text()
        tree = ast.parse(source)

        parent = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node

        def attributed(node):
            while node is not None:
                if isinstance(node, ast.Dict) and \
                        'ci95_method' in self._string_keys(node):
                    return True
                node = parent.get(node)
            return False

        reporting = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = self._string_keys(node)
            if not any(k.startswith('ci95') and k != 'ci95_method'
                       for k in keys):
                continue
            reporting += 1
            assert attributed(node), (
                f"line {node.lineno}: reports an interval "
                f"({sorted(k for k in keys if k.startswith('ci95'))}) with no "
                f"ci95_method on it or on any dict containing it"
            )

        assert reporting == 3, (
            f"expected 3 interval-reporting dicts (predictive, latency, and "
            f"the rescaled latency interpretation), found {reporting}"
        )
