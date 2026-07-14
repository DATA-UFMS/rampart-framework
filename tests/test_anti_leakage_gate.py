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
    def discover_numeric_columns(self, *a, **k): pass
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


class TestViolationIsUnrecoverable:
    """A protocol violation must reach the caller and stop the run."""

    @staticmethod
    def _probe(error):
        import pandas as pd
        from core.base_architecture import BaseArchitectureML

        class Probe(BaseArchitectureML):
            PARADIGM_META = {'name': '_probe_enforcement'}
            def setup_environment(self): pass
            def load_data(self): return pd.DataFrame({'year': [2000]})
            def validate_data(self, data):
                if error is not None:
                    raise error
            def create_target_implementation(self, data): return data
            def _compute_target_statistics(self, data): pass
            def _validate_temporal_folds(self, data, folds): pass
            def save_folds(self, data, folds): pass
            def compute_feature_correlations(self, data, features): return {}
            def apply_collinearity_filter(self, data, features): return features
            def discover_numeric_columns(self, data): return []
            def prepare_features(self, data, features): return data

        return Probe

    def test_violation_propagates(self, tmp_path):
        from core.base_architecture import BaseArchitectureML
        from core.validation import AntiLeakageViolation

        Probe = self._probe(AntiLeakageViolation('Anti-leakage violation (P1)'))
        try:
            with pytest.raises(AntiLeakageViolation):
                Probe('_probe_enforcement', str(tmp_path)).run_setup()
        finally:
            BaseArchitectureML._registry.pop('_probe_enforcement', None)

    def test_operational_failure_stays_recoverable(self, tmp_path):
        """Only violations are unrecoverable; I/O errors still report status."""
        from core.base_architecture import BaseArchitectureML

        Probe = self._probe(FileNotFoundError('missing input'))
        try:
            result = Probe('_probe_enforcement', str(tmp_path)).run_setup()
            assert result['status'] == 'failed'
            assert 'missing input' in result['error']
        finally:
            BaseArchitectureML._registry.pop('_probe_enforcement', None)

    def test_violation_is_a_value_error(self):
        """Existing handlers and tests match on ValueError."""
        from core.validation import AntiLeakageViolation
        assert issubclass(AntiLeakageViolation, ValueError)


class TestSetupExitStatus:
    """Each setup must report failure through its exit status."""

    def test_every_setup_exits_on_failure(self):
        import ast
        from pathlib import Path
        from core.paradigm_registry import discover_paradigms

        root = Path(__file__).resolve().parents[1]
        for name, meta in sorted(discover_paradigms().items()):
            if 'setup_script' not in meta:
                continue
            source = (root / meta['setup_script']).read_text()
            guard = source[source.index("if __name__"):]
            assert 'exit(' in guard, (
                f"{name}: module guard does not propagate an exit status, so a "
                f"failed setup reports success to the pipeline"
            )


class TestFinalFeatureSetAudit:
    """P3 applied to the set the models train on, including appended lags."""

    @staticmethod
    def _panel(n=300, seed=13):
        rng = np.random.default_rng(seed)
        target = rng.normal(size=n)
        return pd.DataFrame({
            'gini': rng.normal(size=n),
            'internet': rng.normal(size=n),
            # A lag correlates with the target by construction.
            'dropout_rate_lag_2': target * 0.85 + rng.normal(0, 0.3, n),
            'target': target,
        })

    def test_autoregressive_feature_is_exempt_and_recorded(self):
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import audit_feature_set

        panel = self._panel()
        report = audit_feature_set(
            panel, ['gini', 'internet', 'dropout_rate_lag_2'],
            'target', SCIENTIFIC_CONFIG)

        exemptions = report['autoregressive_exemptions']
        assert 'dropout_rate_lag_2' in exemptions
        assert exemptions['dropout_rate_lag_2'] > \
            SCIENTIFIC_CONFIG['proxy_correlation_threshold'], (
            'the fixture no longer demonstrates an exemption that matters')

    def test_non_autoregressive_proxy_still_aborts(self):
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import AntiLeakageViolation, audit_feature_set

        panel = self._panel()
        panel['sneaky'] = panel['target'] * 0.97
        with pytest.raises(AntiLeakageViolation, match='proxy detection'):
            audit_feature_set(panel, ['gini', 'sneaky'], 'target',
                              SCIENTIFIC_CONFIG)

    def test_exemption_does_not_cover_joint_reconstruction(self):
        """A lag that reconstructs the target must still abort."""
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import AntiLeakageViolation, audit_feature_set

        panel = self._panel()
        panel['dropout_rate_lag_0'] = panel['target']
        with pytest.raises(AntiLeakageViolation, match='joint reconstruction'):
            audit_feature_set(panel, ['gini', 'dropout_rate_lag_0'], 'target',
                              SCIENTIFIC_CONFIG)

    @pytest.mark.parametrize('kind', ['pandas', 'polars', 'polars_lazy', 'dask'])
    def test_verdict_is_identical_across_frame_types(self, kind):
        """A cross-paradigm gate must not depend on the frame it is handed."""
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import audit_feature_set

        panel = self._panel()
        features = ['gini', 'internet', 'dropout_rate_lag_2']
        expected = audit_feature_set(panel, features, 'target',
                                     SCIENTIFIC_CONFIG)

        if kind == 'pandas':
            data = panel
        elif kind == 'polars':
            pl = pytest.importorskip('polars')
            data = pl.from_pandas(panel)
        elif kind == 'polars_lazy':
            pl = pytest.importorskip('polars')
            data = pl.from_pandas(panel).lazy()
        else:
            dd = pytest.importorskip('dask.dataframe')
            data = dd.from_pandas(panel, npartitions=3)

        report = audit_feature_set(data, features, 'target', SCIENTIFIC_CONFIG)
        assert report['joint_reconstruction_r2'] == pytest.approx(
            expected['joint_reconstruction_r2'], abs=1e-12)
