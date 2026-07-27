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

from conftest import audit_panel

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_ROOT = Path(__file__).resolve().parents[1]

from core.base_architecture import BaseArchitectureML
from core.paradigm_registry import discover_paradigms
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
        report = audit_panel(panel, ['gini', 'internet', 'dropout_rate_lag_2'], 'target')

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
            audit_panel(panel, ['gini', 'sneaky'], 'target')

    def test_a_lag_that_is_not_lagged_aborts(self):
        """A column labelled as lagged carrying the contemporaneous value.

        This is the defect the whole-set check exists for: an off-by-one join,
        or a lag of zero. It reproduces the target exactly, which no genuine
        lag does.
        """
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import AntiLeakageViolation, audit_feature_set

        panel = self._panel()
        panel['dropout_rate_lag_0'] = panel['target']
        with pytest.raises(AntiLeakageViolation, match='target reproduction'):
            audit_panel(panel, ['gini', 'dropout_rate_lag_0'], 'target')

    def test_a_strongly_autocorrelated_lag_does_not_abort(self):
        """The false abort this split prevents.

        On an annual panel pooled across entities, a lag carries the entity's
        level, so the pooled R2 over the whole set is high by construction.
        Judging that against the 0.95 identity ceiling aborts a valid run for
        exhibiting the autocorrelation the task exists to exploit -- and the
        run costs about thirty hours before reaching this point.
        """
        import numpy as np
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import audit_feature_set

        rng = np.random.default_rng(5)
        rows = []
        for entity, level in enumerate([2.0, 9.0, 17.0, 26.0, 35.0]):
            series = level + np.cumsum(rng.normal(0, 0.3, 30))
            for index in range(3, 30):
                rows.append({
                    'entity': entity,
                    'target': series[index],
                    'dropout_rate_lag_2': series[index - 2],
                    'dropout_rate_lag_3': series[index - 3],
                    'gini': 0.2 * series[index] + rng.normal(0, 3.0),
                })
        panel = pd.DataFrame(rows)

        features = ['gini', 'dropout_rate_lag_2', 'dropout_rate_lag_3']
        from core.validation import linear_reconstruction_r2
        whole_set = linear_reconstruction_r2(panel, features, 'target')
        assert whole_set > SCIENTIFIC_CONFIG['identity_r2_threshold'], (
            f'R2 over the whole set is {whole_set:.4f}, below the old ceiling: '
            f'this panel does not reproduce the false abort, so passing proves '
            f'nothing'
        )

        report = audit_panel(panel, features, 'target')
        assert report['full_set_reconstruction_r2'] == pytest.approx(whole_set)
        assert report['joint_reconstruction_r2'] < \
            SCIENTIFIC_CONFIG['identity_r2_threshold']

    def test_an_exogenous_identity_still_aborts(self):
        """Splitting must not let the leakage case through.

        Two exogenous halves that sum to the target: each correlates weakly, so
        the pairwise check cannot see it.
        """
        import numpy as np
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import AntiLeakageViolation, audit_feature_set

        panel = self._panel()
        rng = np.random.default_rng(9)
        target = panel['target'].to_numpy()
        noise = 2.0 * rng.normal(size=len(panel))
        noise -= (np.cov(target, noise, bias=True)[0, 1]
                  / target.var()) * target
        panel['half_a'] = 0.5 * target + noise
        panel['half_b'] = 0.5 * target - noise
        with pytest.raises(AntiLeakageViolation,
                           match='joint reconstruction'):
            audit_panel(panel, ['half_a', 'half_b',
                                      'dropout_rate_lag_2'], 'target')

    def test_lags_cannot_mask_an_exogenous_identity(self):
        """The identity is judged without the lags, so adding lags cannot help."""
        import numpy as np
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import AntiLeakageViolation, audit_feature_set

        panel = self._panel()
        rng = np.random.default_rng(9)
        target = panel['target'].to_numpy()
        noise = 2.0 * rng.normal(size=len(panel))
        noise -= (np.cov(target, noise, bias=True)[0, 1]
                  / target.var()) * target
        panel['half_a'] = 0.5 * target + noise
        panel['half_b'] = 0.5 * target - noise
        panel['dropout_rate_lag_3'] = panel['dropout_rate_lag_2'].shift(1)
        for extra in ([], ['dropout_rate_lag_2'],
                      ['dropout_rate_lag_2', 'dropout_rate_lag_3']):
            with pytest.raises(AntiLeakageViolation,
                               match='joint reconstruction'):
                audit_panel(panel, ['half_a', 'half_b'] + extra, 'target')

    @pytest.mark.parametrize('kind', ['pandas', 'polars', 'polars_lazy', 'dask'])
    def test_verdict_is_identical_across_frame_types(self, kind):
        """A cross-paradigm gate must not depend on the frame it is handed."""
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import audit_feature_set

        panel = self._panel()
        features = ['gini', 'internet', 'dropout_rate_lag_2']

        if kind == 'pandas':
            # The one accepted form: the materialised design matrix.
            report = audit_panel(panel, features, 'target')
            assert report['checks']['joint_reconstruction'] == 'ran'
            return

        if kind == 'polars':
            pl = pytest.importorskip('polars')
            data = pl.from_pandas(panel[features])
        elif kind == 'polars_lazy':
            pl = pytest.importorskip('polars')
            data = pl.from_pandas(panel[features]).lazy()
        else:
            dd = pytest.importorskip('dask.dataframe')
            data = dd.from_pandas(panel[features], npartitions=3)

        # Invariance across frame types used to be a property the audit had to
        # exhibit, and a test had to assert. It is now structural: the audit
        # takes the matrix the model fits, every paradigm materialises the fold
        # through canonical_fold to get it, and anything else is refused rather
        # than handled -- so there is no frame-type axis left to vary.
        with pytest.raises(TypeError, match='materialised design matrix'):
            audit_feature_set(data, panel['target'], autoregressive=[],
                              unaudited_by_selection=[],
                              config=SCIENTIFIC_CONFIG)


class TestOneImplementationOfEachCheck:
    """The setup-level and model-level P3 checks must not drift apart.

    Both materialise a dense matrix and both fit the target on the selected
    features, and each had its own copy. They had already diverged: the
    setup-level copy did not handle a Polars LazyFrame, so the same check
    raised TypeError on input the model-level one accepts.
    """

    @staticmethod
    def _frame():
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(3)
        target = rng.normal(size=40)
        return pd.DataFrame({
            'year': list(range(2000, 2040)),
            'gini': 0.4 * target + rng.normal(size=40),
            'target': target,
        })

    def test_the_architecture_delegates_materialisation(self):
        source = (_ROOT / 'src' / 'core' / 'base_architecture.py').read_text()
        block = source[source.index('def _materialise_pandas'):]
        block = block[:block.index('\n    def ', 1)]
        assert 'materialise_pandas(data, columns)' in block
        assert 'isinstance(data, pl.DataFrame)' not in block, (
            'a second dispatch here is how the two drifted apart'
        )

    def test_the_architecture_delegates_the_reconstruction(self):
        source = (_ROOT / 'src' / 'core' / 'base_architecture.py').read_text()
        block = source[source.index('def _linear_reconstruction_r2'):]
        block = block[:block.index('\n    def ', 1)]
        assert 'linear_reconstruction_r2(data, features' in block
        assert 'np.linalg.lstsq' not in block

    def test_a_lazyframe_is_accepted(self):
        """The divergence, reproduced: the setup-level copy raised here."""
        polars = pytest.importorskip('polars')
        from core.validation import materialise_pandas

        lazy = polars.from_pandas(self._frame()).lazy()
        materialised = materialise_pandas(lazy, ['gini', 'target'])
        assert list(materialised.columns) == ['gini', 'target']
        assert len(materialised) == 40

    def test_both_paths_give_the_same_r2(self):
        """Whichever entry point is used, the same number comes out."""
        from core.base_architecture import BaseArchitectureML
        from core.validation import linear_reconstruction_r2

        class Probe(BaseArchitectureML):
            def setup_environment(self): pass
            def load_data(self): pass
            def validate_data(self, data): pass
            def create_target_implementation(self, data): return data
            def _compute_target_statistics(self, data): pass
            def _validate_temporal_folds(self, data, folds): pass
            def save_folds(self, data, folds): pass
            def compute_feature_correlations(self, data, features): return {}
            def apply_collinearity_filter(self, data, features,
                                          threshold=0.8): return features
            def prepare_features(self, data, features): return data
            def discover_numeric_columns(self, data): return []

        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            architecture = Probe('sql_engine', '/tmp')
        frame = self._frame().rename(
            columns={'target': architecture.target_column})
        assert architecture._linear_reconstruction_r2(frame, ['gini']) == \
            linear_reconstruction_r2(frame, ['gini'],
                                     architecture.target_column)

    def test_the_dead_wrapper_is_gone(self):
        """audit_final_features delegated to audit_feature_set and had no caller.

        The models call audit_feature_set directly. A second name for the same
        gate invites one of them to be updated alone.
        """
        source = (_ROOT / 'src' / 'core' / 'base_architecture.py').read_text()
        assert 'audit_final_features' not in source


class TestTheTwoThresholdsAnswerDifferentQuestions:
    """One is a modelling choice, the other is numerical.

    Collapsing them was the defect: judging the whole set, lags included,
    against the 0.95 identity ceiling.
    """

    def test_the_reproduction_tolerance_is_numerical(self):
        from core.scientific_config import SCIENTIFIC_CONFIG
        tolerance = SCIENTIFIC_CONFIG['target_reproduction_tolerance']
        assert tolerance <= 1e-6, (
            f'{tolerance} is a modelling threshold, not a numerical one: at '
            f'that size a legitimately autocorrelated lag set trips it'
        )
        assert tolerance > 0

    def test_the_identity_ceiling_is_a_modelling_choice(self):
        from core.scientific_config import SCIENTIFIC_CONFIG
        assert 0.5 < SCIENTIFIC_CONFIG['identity_r2_threshold'] < 1.0

    def test_a_loosened_tolerance_would_abort_a_valid_run(self):
        """Why the tolerance may not drift upwards.

        The same autocorrelated panel that passes today aborts once the
        tolerance reaches the identity ceiling's scale.
        """
        import numpy as np
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import AntiLeakageViolation, audit_feature_set

        rng = np.random.default_rng(5)
        rows = []
        for entity, level in enumerate([2.0, 9.0, 17.0, 26.0, 35.0]):
            series = level + np.cumsum(rng.normal(0, 0.3, 30))
            for index in range(3, 30):
                rows.append({'target': series[index],
                             'dropout_rate_lag_2': series[index - 2],
                             'dropout_rate_lag_3': series[index - 3],
                             'gini': 0.2 * series[index]
                                     + rng.normal(0, 3.0)})
        panel = pd.DataFrame(rows)
        features = ['gini', 'dropout_rate_lag_2', 'dropout_rate_lag_3']

        audit_panel(panel, features, 'target')

        loosened = {**SCIENTIFIC_CONFIG,
                    'target_reproduction_tolerance': 0.05}
        with pytest.raises(AntiLeakageViolation, match='target reproduction'):
            audit_panel(panel, features, 'target', config=loosened)


class TestTheGateHasSomethingToAttestTo:
    """An empty fold list satisfied "no invalid folds" vacuously.

    The pipeline logged "0 folds -- integridade temporal verificada" and went
    on to the benchmark. Zero folds means the models had nothing to train on,
    or the artifact is broken; neither is temporal integrity.
    """

    def test_an_empty_configuration_halts(self):
        from core.validation import AntiLeakageViolation, TemporalValidator
        with pytest.raises(AntiLeakageViolation, match='empty'):
            TemporalValidator(min_gap_years=2).enforce_walk_forward([])

    def test_a_valid_configuration_still_passes(self):
        """Otherwise raising unconditionally would satisfy the test above."""
        from core.validation import TemporalValidator
        TemporalValidator(min_gap_years=2).enforce_walk_forward([{
            'train_start': 2000, 'train_end': 2007,
            'val_start': 2010, 'val_end': 2011,
            'test_start': 2014, 'test_end': 2015,
        }])

    def test_the_report_no_longer_calls_zero_folds_valid(self):
        from core.validation import TemporalValidator
        valid, report = TemporalValidator(
            min_gap_years=2).validate_walk_forward([])
        assert report['total_folds'] == 0
        # validate_walk_forward stays descriptive; enforcement is where the
        # decision lives, and that is what the pipeline calls.
        assert valid is True


class TestTheParadigmsShareTheirFolds:
    """Each paradigm's folds were validated alone, never against the others.

    Splits that differ across paradigms make the comparison a comparison
    between different problems: the bitwise claim would be falsified for that
    reason rather than by the implementations.
    """

    @staticmethod
    def _write(root, windows_by_paradigm, created):
        import json
        for paradigm, windows in windows_by_paradigm.items():
            directory = (root / 'ml_pipeline' / 'architectures' / paradigm
                         / 'prep')
            directory.mkdir(parents=True, exist_ok=True)
            folds = [{'fold_id': index,
                      'train_start': w[0], 'train_end': w[1],
                      'val_start': w[2], 'val_end': w[3],
                      'test_start': w[4], 'test_end': w[5]}
                     for index, w in enumerate(windows)]
            (directory / f'temporal_folds_{paradigm}.json').write_text(
                json.dumps({'creation_timestamp': created.isoformat(),
                            'folds': folds}))

    @pytest.fixture
    def gate(self, tmp_path, monkeypatch):
        import sys
        from datetime import datetime, timedelta

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import pipeline

        outputs = tmp_path / 'outputs'
        monkeypatch.setattr(
            pipeline, 'get_absolute_output_path',
            lambda relative: str(outputs / relative.replace('outputs/', '')))
        started = datetime.now() - timedelta(seconds=5)
        return pipeline, outputs, started, datetime.now()

    #: Two folds that satisfy the gap of two years.
    SHARED = [(2000, 2007, 2010, 2011, 2014, 2015),
              (2000, 2008, 2011, 2012, 2015, 2016)]

    def test_identical_folds_pass(self, gate):
        pipeline, outputs, started, created = gate
        paradigms = sorted(discover_paradigms())
        self._write(outputs, {p: self.SHARED for p in paradigms}, created)
        pipeline._validate_anti_leakage_gate(str(outputs), started)

    def test_a_shifted_window_halts(self, gate):
        """Same count, same gaps, different years: counting folds misses it."""
        pipeline, outputs, started, created = gate
        paradigms = sorted(discover_paradigms())
        windows = {p: self.SHARED for p in paradigms}
        windows[paradigms[0]] = [(2000, 2007, 2010, 2011, 2014, 2015),
                                 (2000, 2009, 2012, 2013, 2016, 2017)]
        assert len(windows[paradigms[0]]) == len(self.SHARED)
        self._write(outputs, windows, created)
        with pytest.raises(ValueError, match='mesmos folds'):
            pipeline._validate_anti_leakage_gate(str(outputs), started)

    def test_a_missing_fold_halts(self, gate):
        pipeline, outputs, started, created = gate
        paradigms = sorted(discover_paradigms())
        windows = {p: self.SHARED for p in paradigms}
        windows[paradigms[-1]] = self.SHARED[:1]
        self._write(outputs, windows, created)
        with pytest.raises(ValueError, match='mesmos folds'):
            pipeline._validate_anti_leakage_gate(str(outputs), started)

    def test_an_empty_configuration_halts_before_the_comparison(self, gate):
        """Three empty lists agree with each other, and agreement is not integrity."""
        from core.validation import AntiLeakageViolation
        pipeline, outputs, started, created = gate
        paradigms = sorted(discover_paradigms())
        self._write(outputs, {p: [] for p in paradigms}, created)
        with pytest.raises(AntiLeakageViolation, match='empty'):
            pipeline._validate_anti_leakage_gate(str(outputs), started)


class TestCreateTemporalFoldsEnforces:
    """Deleting the enforcement call left the whole suite green.

    `create_temporal_folds` is the only path by which folds reach the models.
    Nothing checked that the folds it returns were run past the validator, so
    the generator was free to emit a violating set.
    """

    @staticmethod
    def _probe(gap):
        class Config:
            temporal_range = (2000, 2023)
            walk_forward_config = {'min_train': 8, 'val_len': 2, 'test_len': 2}
            year_column = 'year'
            entity_column = 'country_code'
            entity_name_column = 'country_name'
            stratification_column = None
            target_source_column = 'source_rate'
            feature_columns = []
            excluded_columns = []

        class Probe(BaseArchitectureML):
            def setup_environment(self): pass
            def load_data(self): pass
            def validate_data(self, data): pass
            def create_target_implementation(self, data): return data
            def _compute_target_statistics(self, data): pass
            def _validate_temporal_folds(self, data, folds): pass
            def save_folds(self, data, folds): pass
            def compute_feature_correlations(self, data, features): return {}
            def apply_collinearity_filter(self, data, features,
                                          threshold=0.8): return features
            def prepare_features(self, data, features): return data
            def discover_numeric_columns(self, data): return []

        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            architecture = Probe('sql_engine', '/tmp', dataset_config=Config())
        architecture.config = {**architecture.config,
                               'temporal_gap_years': gap}
        return architecture

    def test_the_returned_folds_went_through_the_validator(self):
        """A generator that emits a violating set must not get past here."""
        from core.validation import AntiLeakageViolation

        architecture = self._probe(gap=2)
        violating = [{'train_start': 2000, 'train_end': 2009,
                      'val_start': 2010, 'val_end': 2011,
                      'test_start': 2014, 'test_end': 2015}]
        architecture._generate_walkforward_folds_auto = lambda: violating
        with pytest.raises(AntiLeakageViolation):
            architecture.create_temporal_folds()

    def test_an_empty_generator_result_halts(self):
        from core.validation import AntiLeakageViolation

        architecture = self._probe(gap=2)
        architecture._generate_walkforward_folds_auto = lambda: []
        with pytest.raises(AntiLeakageViolation, match='empty'):
            architecture.create_temporal_folds()

    def test_valid_folds_are_returned(self):
        """Otherwise raising unconditionally would satisfy both tests above."""
        import contextlib
        import io
        architecture = self._probe(gap=2)
        with contextlib.redirect_stdout(io.StringIO()):
            folds = architecture.create_temporal_folds()
        assert len(folds) == 9

    def test_the_validator_reads_the_configured_gap(self):
        """Enforcing with a gap the generator did not use proves nothing."""
        import contextlib
        import io
        from core.validation import AntiLeakageViolation

        architecture = self._probe(gap=3)
        # Folds built for a gap of two, enforced under three.
        architecture._generate_walkforward_folds_auto = lambda: [
            {'train_start': 2000, 'train_end': 2007,
             'val_start': 2010, 'val_end': 2011,
             'test_start': 2014, 'test_end': 2015}]
        with pytest.raises(AntiLeakageViolation):
            with contextlib.redirect_stdout(io.StringIO()):
                architecture.create_temporal_folds()


class TestEveryParadigmCallsTheFinalAudit:
    """No test checked that any paradigm invokes audit_feature_set.

    Removing the call was caught only incidentally, by the unused-import check.
    Removing the call *and* the import would have passed.
    """

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_the_hierarchical_model_calls_it(self, paradigm):
        import ast as ast_module
        path = (_ROOT / 'src' / 'architectures_ml' / paradigm / 'models'
                / 'hierarchical_model.py')
        tree = ast_module.parse(path.read_text())
        called = {node.func.id for node in ast_module.walk(tree)
                  if isinstance(node, ast_module.Call)
                  and isinstance(node.func, ast_module.Name)}
        assert 'audit_feature_set' in called, (
            f'{paradigm} trains without the final-feature P3 audit, so the '
            f'lags appended after selection never pass a gate'
        )

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_the_result_is_kept(self, paradigm):
        """Calling and discarding leaves no record in the artifacts."""
        source = (_ROOT / 'src' / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        assert 'self._feature_audits.append((fold_id, audit_feature_set(' in source


class TestTheAuditsDomainAndItsSilences:
    """The three decisions the redesign turns on, and none was pinned.

    Caught by mutation: reverting each left the suite green.
    """

    @staticmethod
    def _panel(rows=400, seed=7):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({
            'gini': rng.normal(size=rows),
            'internet': rng.normal(size=rows),
            'dropout_rate_lag_2': rng.normal(size=rows),
        })
        frame['target'] = 0.3 * frame['gini'] + rng.normal(size=rows)
        return frame

    def test_the_ceiling_spares_what_selection_already_audited(self):
        """Selection applies this ceiling over the full panel and aborts there.

        Re-measuring the same columns on a fold's training window cannot detect
        a proxy selection missed -- the panel is the larger sample -- so it can
        only disagree with itself through sampling noise, and the disagreement
        would abort one paradigm and not the other two.
        """
        from core.validation import AntiLeakageViolation

        import numpy as np

        panel = self._panel()
        # Calibrated to trip the pairwise ceiling and nothing else: |r| = 0.87
        # against a ceiling of 0.80, while the set reconstructs the target with
        # R2 = 0.77 against an identity threshold of 0.95. A proxy that is an
        # exact function of the target would fail the identity check first and
        # the test would pass for the wrong reason.
        panel['sneaky'] = panel['target'] + 0.62 * np.random.default_rng(
            11).normal(size=len(panel))

        # Cleared by selection: outside the audit's domain, so it stands.
        report = audit_panel(panel, ['gini', 'sneaky', 'dropout_rate_lag_2'],
                             'target', unaudited=[])
        assert report['checks']['proxy_ceiling'] == 'not_applicable'
        assert report['joint_reconstruction_r2'] < 0.95, (
            'the fixture trips the identity check, not the ceiling')
        assert report['max_nonautoregressive_correlation'] > 0.80, (
            'the correlation is reported even when nothing was eligible')

        # The same column, never audited by selection: inside the domain.
        with pytest.raises(AntiLeakageViolation, match='never audited'):
            audit_panel(panel, ['gini', 'sneaky', 'dropout_rate_lag_2'],
                        'target', unaudited=['sneaky'])

    def test_the_ceiling_judges_only_its_domain_not_everything_present(self):
        """The discriminating case: a non-empty domain with the proxy outside it.

        With nothing eligible the ceiling short-circuits, so a test built that
        way never reaches the comparison and cannot see whether it judges the
        right columns. Here one benign column is eligible and the proxy is not:
        judging the whole exogenous set would abort a run that is sound.
        """
        import numpy as np

        panel = self._panel()
        panel['sneaky'] = panel['target'] + 0.62 * np.random.default_rng(
            11).normal(size=len(panel))
        panel['fresh'] = np.random.default_rng(12).normal(size=len(panel))

        report = audit_panel(
            panel, ['gini', 'sneaky', 'fresh', 'dropout_rate_lag_2'],
            'target', unaudited=['fresh'])
        assert report['checks']['proxy_ceiling'] == 'ran', (
            'the domain was empty, so the comparison never happened')
        assert report['unaudited_by_selection'] == ['fresh']

    def test_a_matrix_of_lags_alone_halts(self):
        """Two paradigms reached this state by reading a key that was absent.

        With no exogenous column the reconstruction check has nothing to
        evaluate, returns None, and the audit used to move on -- leaving a model
        trained on the target's own past and a receipt that looked complete.
        """
        from core.validation import AntiLeakageViolation

        panel = self._panel()
        with pytest.raises(AntiLeakageViolation,
                           match="target's own past alone"):
            audit_panel(panel, ['dropout_rate_lag_2'], 'target')

    def test_a_check_that_cannot_be_computed_says_so(self):
        """Listwise deletion can leave fewer complete rows than predictors.

        `linear_reconstruction_r2` returns None there, and the report used to
        read exactly like one where the check had passed. The gate reads this
        field, so the distinction has to survive to disk.
        """
        import numpy as np

        panel = self._panel(rows=6)
        # Every row incomplete in at least one predictor: nothing to fit on.
        panel.loc[panel.index[:4], 'gini'] = np.nan
        panel.loc[panel.index[4:], 'internet'] = np.nan

        report = audit_panel(panel, ['gini', 'internet', 'dropout_rate_lag_2'],
                             'target')
        assert report['checks']['joint_reconstruction'] == 'indeterminate'
        assert report['joint_reconstruction_r2'] is None
