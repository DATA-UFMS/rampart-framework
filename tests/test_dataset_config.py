"""
Testes para o sistema de configuração de datasets.

Verifica que o DatasetConfig Protocol funciona, que ambos datasets
estão registrados, e que o adapter INEP→framework é correto.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDatasetRegistry:
    """Testes do registry e Protocol de datasets."""

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
        assert wb.target_formula == 'invert'

    def test_get_inep(self):
        from core.dataset_config import get_dataset
        inep = get_dataset('inep_censo')
        assert inep.name == 'inep_censo'
        assert inep.temporal_range == (2012, 2023)
        assert inep.entity_column == 'municipality_code'
        assert inep.target_formula == 'direct'

    def test_get_nonexistent_raises(self):
        from core.dataset_config import get_dataset
        with pytest.raises(KeyError, match="não registrado"):
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
        """Excluded columns devem incluir a entity column (evita usar como feature)."""
        from core.dataset_config import get_dataset
        for name in ['worldbank', 'inep_censo']:
            cfg = get_dataset(name)
            assert cfg.entity_column in cfg.excluded_columns


class TestInepAdapter:
    """Testes do adapter INEP → schema do framework."""

    @staticmethod
    def _make_inep_df(**extra):
        """DataFrame no formato que parse_year() produz."""
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

    def test_adapter_creates_enrollment_rate(self):
        from collection.inep_collector import adapt_to_framework_schema
        adapted = adapt_to_framework_schema(self._make_inep_df())
        assert 'enrollment_rate_secondary_net' in adapted.columns

    def test_adapter_preserves_features(self):
        """Features INEP (aprov_ef, reprov_em, etc.) passam inalteradas."""
        from collection.inep_collector import adapt_to_framework_schema
        adapted = adapt_to_framework_schema(
            self._make_inep_df(aprov_ef=[92.5], reprov_em=[3.7])
        )
        assert adapted['aprov_ef'].iloc[0] == pytest.approx(92.5)
        assert adapted['reprov_em'].iloc[0] == pytest.approx(3.7)


class TestBaseArchitectureDatasetConfig:
    """Testa que BaseArchitectureML respeita DatasetConfig."""

    def test_default_is_worldbank(self):
        from core.base_architecture import BaseArchitectureML
        # Não podemos instanciar ABC, mas verificamos que o __init__
        # aceita dataset_config como parâmetro
        import inspect
        sig = inspect.signature(BaseArchitectureML.__init__)
        assert 'dataset_config' in sig.parameters
        default = sig.parameters['dataset_config'].default
        assert default is None  # None = default worldbank

    def test_excluded_features_from_config(self):
        """get_excluded_features deve usar dataset_config.excluded_columns."""
        from core.base_architecture import BaseArchitectureML
        from core.dataset_config import get_dataset

        # Criar mock mínimo para testar
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
            def get_numeric_features(self, d): pass
            def prepare_features(self, d, f, t): pass

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inep_cfg = get_dataset('inep_censo')
            arch = MockArch('test', tmpdir, dataset_config=inep_cfg)
            excluded = arch.get_excluded_features()
            assert 'municipality_code' in excluded
            assert 'state_code' in excluded

        # Cleanup registry
        BaseArchitectureML._registry.pop('_test_dataset_excl', None)
