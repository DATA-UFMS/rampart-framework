"""
Tests for the dataset configuration system.

Verifies that the DatasetConfig Protocol works, that both datasets are
registered, and that the INEP→framework adapter is correct.
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDatasetRegistry:
    """Tests of the dataset registry and Protocol."""

    def test_both_datasets_registered(self):
        import datasets  # noqa: F401
        from core.dataset_config import list_datasets
        registered = list_datasets()
        assert 'worldbank' in registered
        assert 'inep_censo' in registered

    def test_get_worldbank(self):
        from core.dataset_config import get_dataset
        wb = get_dataset('worldbank')
        assert wb.name == 'worldbank'
        assert wb.temporal_range == (2000, 2023)
        assert wb.entity_column == 'country_code'

    def test_get_inep(self):
        from core.dataset_config import get_dataset
        inep = get_dataset('inep_censo')
        assert inep.name == 'inep_censo'
        assert inep.temporal_range == (2007, 2024)
        assert inep.entity_column == 'country_code'

    def test_get_nonexistent_raises(self):
        from core.dataset_config import get_dataset
        with pytest.raises(KeyError, match="not registered"):
            get_dataset('nonexistent_dataset')

    def test_protocol_compliance(self):
        from core.dataset_config import DatasetConfig, get_dataset
        wb = get_dataset('worldbank')
        inep = get_dataset('inep_censo')
        assert isinstance(wb, DatasetConfig)
        assert isinstance(inep, DatasetConfig)

    def test_walk_forward_config_complete(self):
        from core.dataset_config import get_dataset
        for name in ['worldbank', 'inep_censo']:
            cfg = get_dataset(name)
            wf = cfg.walk_forward_config
            assert 'min_train' in wf
            assert 'val_len' in wf
            assert 'test_len' in wf
            assert 'gap' in wf
            assert 'step' in wf

    def test_strata_non_empty(self):
        from core.dataset_config import get_dataset
        for name in ['worldbank', 'inep_censo']:
            cfg = get_dataset(name)
            assert len(cfg.strata) > 0
            for stratum, entities in cfg.strata.items():
                assert len(entities) > 0

    def test_excluded_columns_include_entity(self):
        """Excluded columns must include the entity column (keeps it from being used as a feature)."""
        from core.dataset_config import get_dataset
        for name in ['worldbank', 'inep_censo']:
            cfg = get_dataset(name)
            assert cfg.entity_column in cfg.excluded_columns


class TestInepAdapter:
    """Tests of the INEP → framework schema adapter."""

    @staticmethod
    def _make_inep_df(**extra):
        """DataFrame in the format parse_year() produces."""
        base = {
            'ano': [2019], 'regiao': ['Sudeste'], 'uf': ['SP'],
            'cod_municipio': [3550308.0], 'nome_municipio': ['São Paulo'],
            'localizacao': ['Total'], 'dependencia': ['Total'],
            'abandono_em': [6.0], 'aprov_em': [90.0], 'reprov_em': [4.0],
            'aprov_ef': [92.0], 'reprov_ef': [5.0], 'abandono_ef': [3.0],
        }
        base.update(extra)
        return pd.DataFrame(base)

    def test_adapter_renames_columns(self):
        from collection.inep_collector import adapt_to_framework_schema
        adapted = adapt_to_framework_schema(self._make_inep_df())
        assert 'country_code' in adapted.columns
        assert 'country_stratum' in adapted.columns
        assert 'country_name' in adapted.columns
        assert 'cod_municipio' not in adapted.columns

    def test_adapter_inverts_abandono(self):
        from collection.inep_collector import adapt_to_framework_schema
        adapted = adapt_to_framework_schema(self._make_inep_df(abandono_em=[6.0]))
        assert 'lower_secondary_completion_rate' in adapted.columns
        assert adapted['lower_secondary_completion_rate'].iloc[0] == pytest.approx(94.0)


    def test_adapter_preserves_lower_secondary_features(self):
        """Lower-secondary rates reach the final dataset unchanged."""
        from collection.inep_collector import adapt_to_framework_schema
        adapted = adapt_to_framework_schema(
            self._make_inep_df(aprov_ef=[92.5], reprov_ef=[4.2])
        )
        assert adapted['aprov_ef'].iloc[0] == pytest.approx(92.5)
        assert adapted['reprov_ef'].iloc[0] == pytest.approx(4.2)

    def test_adapter_drops_upper_secondary_rates(self):
        """The target's algebraic components must not reach the feature pool.

        aprovacao + reprovacao + abandono partition each level exactly, so any
        upper-secondary rate reconstructs the upper-secondary dropout target.
        """
        from collection.inep_collector import adapt_to_framework_schema
        adapted = adapt_to_framework_schema(self._make_inep_df())

        leaked = [
            col for col in adapted.columns
            if col.endswith('_em') or '_em_' in col
        ]
        assert not leaked, (
            f"upper-secondary rates reached the feature pool and reconstruct "
            f"the target: {leaked}"
        )


class TestBaseArchitectureDatasetConfig:
    """Tests that BaseArchitectureML respects DatasetConfig."""

    def test_default_is_worldbank(self):
        from core.base_architecture import BaseArchitectureML
        import inspect
        sig = inspect.signature(BaseArchitectureML.__init__)
        assert 'dataset_config' in sig.parameters
        default = sig.parameters['dataset_config'].default
        assert default is None  # None = default worldbank

    def test_excluded_features_from_config(self):
        """get_excluded_features must use dataset_config.excluded_columns."""
        from core.base_architecture import BaseArchitectureML
        from core.dataset_config import get_dataset

        # Build a minimal mock for the test
        class MockArch(BaseArchitectureML):
            PARADIGM_META = {
                'name': '_test_dataset_excl',
                'label': 'Test',
                'processor_module': 'x', 'setup_module': 'x',
                'baseline_module': 'x', 'hierarchical_module': 'x',
            }
            def setup_environment(self): pass
            def load_data(self): pass
            def validate_data(self, d): pass
            def create_target_implementation(self, d): pass
            def _compute_target_statistics(self, d): pass
            def _validate_temporal_folds(self, d, f): pass
            def save_folds(self, d, f): pass
            def compute_feature_correlations(self, d, f): pass
            def apply_collinearity_filter(self, d, f): pass
            def discover_numeric_columns(self, d): pass
            def prepare_features(self, d, f, t): pass

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inep_cfg = get_dataset('inep_censo')
            arch = MockArch('test', tmpdir, dataset_config=inep_cfg)
            try:
                excluded = arch.get_excluded_features()
                assert 'country_code' in excluded
                assert 'country_stratum' in excluded
            finally:
                # A failing assertion must not leave the mock in the global
                # registry, where it would break unrelated tests.
                BaseArchitectureML._registry.pop('_test_dataset_excl', None)


class TestInepConfigMatchesCollector:
    """The declared schema must be the one the collector produces."""

    @staticmethod
    def _produced_columns():
        from collection.inep_collector import FEATURE_COLS
        framework_schema = {
            'country_code', 'country_name', 'country_stratum', 'year',
            'lower_secondary_completion_rate',
        }
        return framework_schema | set(FEATURE_COLS)

    def test_declared_columns_exist(self):
        from datasets.inep_censo import InepCensoDatasetConfig as cfg
        produced = self._produced_columns()
        for field in ('entity_column', 'entity_name_column',
                      'stratification_column', 'target_source_column'):
            column = getattr(cfg, field)
            assert column in produced, (
                f"{field}={column!r} is not produced by the collector"
            )

    def test_declared_features_exist(self):
        from datasets.inep_censo import InepCensoDatasetConfig as cfg
        missing = [c for c in cfg.feature_columns
                   if c not in self._produced_columns()]
        assert not missing, f"declared features absent from the data: {missing}"

    def test_exclusion_list_reaches_real_columns(self):
        """An exclusion list of absent names would disable P3 silently."""
        from datasets.inep_censo import InepCensoDatasetConfig as cfg
        effective = set(cfg.excluded_columns) & self._produced_columns()
        assert effective, (
            "no excluded column exists in the data, so the P3 exclusion list "
            "has no effect"
        )


class TestTargetSubstitutionIsRejected:
    """A missing target must abort, never fall back to a similar name."""

    def test_no_setup_substitutes_the_target(self):
        import ast
        from pathlib import Path
        from core.paradigm_registry import discover_paradigms

        root = Path(__file__).resolve().parents[1]
        for name, meta in sorted(discover_paradigms().items()):
            if 'setup_script' not in meta:
                continue
            source = (root / meta['setup_script']).read_text()
            tree = ast.parse(source)
            validate = next(
                (n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == 'validate_data'),
                None,
            )
            assert validate is not None, f"{name}: no validate_data"

            body = ast.unparse(validate)
            assert 'Target column' in body, (
                f"{name}: validate_data does not abort on a missing target"
            )
            # Reassigning source_column here is how the substitution happened.
            reassigns = [
                n for n in ast.walk(validate)
                if isinstance(n, ast.Assign)
                and any(getattr(t, 'attr', None) == 'source_column'
                        for t in n.targets)
            ]
            assert not reassigns, (
                f"{name}: validate_data reassigns source_column, substituting "
                f"the declared target"
            )
