#!/usr/bin/env python3
"""P3 joint-reconstruction check.

Pairwise correlation cannot detect a target that partitions across several
features: each correlates moderately while together they determine it exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.base_architecture import BaseArchitectureML
from core.scientific_config import SCIENTIFIC_CONFIG


class _Probe(BaseArchitectureML):
    """Concrete subclass exposing the reconstruction check in isolation."""

    def __init__(self, target_column: str):
        self.config = SCIENTIFIC_CONFIG
        self.target_column = target_column

    def _compute_target_statistics(self, *a, **k): pass
    def _validate_temporal_folds(self, *a, **k): pass
    def apply_collinearity_filter(self, *a, **k): pass
    def compute_feature_correlations(self, *a, **k): pass
    def create_target_implementation(self, *a, **k): pass
    def get_numeric_features(self, *a, **k): pass
    def load_data(self, *a, **k): pass
    def prepare_features(self, *a, **k): pass
    def save_folds(self, *a, **k): pass
    def setup_environment(self, *a, **k): pass
    def validate_data(self, *a, **k): pass


@pytest.fixture
def probe():
    return _Probe('target')


def _partitioned_panel(n=400, seed=7):
    """Rates that sum to 100, so the target is an exact linear function.

    The components carry comparable variance, which is what keeps each pairwise
    correlation near sqrt(1/k) and below the proxy threshold while their joint
    fit is exact. This is the shape of INEP's rendimento rates.
    """
    rng = np.random.default_rng(seed)
    approved = rng.uniform(20.0, 60.0, n)
    failed = rng.uniform(20.0, 60.0, n)
    return pd.DataFrame({
        'approved': approved,
        'failed': failed,
        'target': 100.0 - approved - failed,
    })


def _independent_panel(n=400, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'a': rng.normal(size=n),
        'b': rng.normal(size=n),
        'target': rng.normal(size=n),
    })


class TestJointReconstruction:

    def test_exact_partition_is_detected(self, probe):
        panel = _partitioned_panel()
        r2 = probe._linear_reconstruction_r2(panel, ['approved', 'failed'])
        assert r2 == pytest.approx(1.0, abs=1e-9)
        assert r2 > SCIENTIFIC_CONFIG['identity_r2_threshold']

    def test_exact_partition_is_invisible_to_pairwise_correlation(self, probe):
        """Why the check exists: no single rate clears the proxy threshold."""
        panel = _partitioned_panel()
        threshold = SCIENTIFIC_CONFIG['proxy_correlation_threshold']
        pairwise = panel[['approved', 'failed']].corrwith(panel['target']).abs()
        assert (pairwise < threshold).all(), (
            f"fixture no longer demonstrates the blind spot: {dict(pairwise)}"
        )

    def test_independent_features_pass(self, probe):
        panel = _independent_panel()
        r2 = probe._linear_reconstruction_r2(panel, ['a', 'b'])
        assert r2 < SCIENTIFIC_CONFIG['identity_r2_threshold']

    def test_no_features_yields_no_verdict(self, probe):
        assert probe._linear_reconstruction_r2(_independent_panel(), []) is None

    def test_underdetermined_fit_yields_no_verdict(self, probe):
        panel = _independent_panel(n=2)
        assert probe._linear_reconstruction_r2(panel, ['a', 'b']) is None

    def test_constant_target_yields_no_verdict(self, probe):
        panel = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'target': [5.0] * 4})
        assert probe._linear_reconstruction_r2(panel, ['a']) is None


class TestMaterialisation:
    """The check must accept the frame type of every paradigm."""

    def test_pandas(self, probe):
        panel = _partitioned_panel(n=50)
        out = probe._materialise_pandas(panel, ['approved', 'target'])
        assert list(out.columns) == ['approved', 'target']
        assert len(out) == 50

    def test_polars(self, probe):
        pl = pytest.importorskip('polars')
        panel = pl.from_pandas(_partitioned_panel(n=50))
        out = probe._materialise_pandas(panel, ['approved', 'target'])
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 50

    def test_dask(self, probe):
        dd = pytest.importorskip('dask.dataframe')
        panel = dd.from_pandas(_partitioned_panel(n=50), npartitions=2)
        out = probe._materialise_pandas(panel, ['approved', 'target'])
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 50

    def test_unsupported_type_is_rejected(self, probe):
        with pytest.raises(TypeError):
            probe._materialise_pandas(object(), ['a'])
