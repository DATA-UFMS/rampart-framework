"""
Tests for the dataset configuration system.

Verifies that the DatasetConfig Protocol works, that both datasets are
registered, and that the INEP→framework adapter is correct.
"""

import sys
import os
import pytest
import pandas as pd

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
sys.path.insert(0, str(_SRC))


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
        assert wb.entity_column == 'entity_id'

    def test_get_inep(self):
        from core.dataset_config import get_dataset
        inep = get_dataset('inep_censo')
        assert inep.name == 'inep_censo'
        assert inep.temporal_range == (2007, 2024)
        assert inep.entity_column == 'entity_id'

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
        assert 'entity_id' in adapted.columns
        assert 'entity_stratum' in adapted.columns
        assert 'entity_name' in adapted.columns
        assert 'cod_municipio' not in adapted.columns

    def test_adapter_inverts_abandono(self):
        from collection.inep_collector import adapt_to_framework_schema
        adapted = adapt_to_framework_schema(self._make_inep_df(abandono_em=[6.0]))
        assert 'target_source_rate' in adapted.columns
        assert adapted['target_source_rate'].iloc[0] == pytest.approx(94.0)


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
                assert 'entity_id' in excluded
                assert 'entity_stratum' in excluded
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
            'entity_id', 'entity_name', 'entity_stratum', 'year',
            'target_source_rate',
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


class TestTheRegistryIsTheDispatch:
    """The registry existed, was tested, and production never read it.

    Every dataset registered itself on import and `get_dataset` had unit tests,
    while `BaseArchitectureML.__init__` resolved the config with an if/else on
    the name. An extension point declared, exercised by the suite, and unused --
    the same shape as an artifact nobody reads.

    The branch it replaced ended in `else: worldbank`, so a typo in
    DATASET_NAME ran the World Bank panel and wrote it under a directory named
    after the typo. Nothing downstream distinguishes that from a correct run:
    the outputs root is derived from the same variable.
    """

    def test_production_resolves_through_the_registry(self):
        source = (_SRC / 'core' / 'base_architecture.py').read_text()
        assert 'get_dataset(' in source, (
            'the dispatch no longer goes through the registry')
        assert 'InepCensoDatasetConfig()' not in source, (
            'the core names a concrete dataset again')

    def test_an_unknown_name_halts_instead_of_falling_back(self):
        import datasets  # noqa: F401
        from core.dataset_config import get_dataset

        with pytest.raises(KeyError) as caught:
            get_dataset('inep_cens')
        assert 'inep_cens' in str(caught.value)
        assert 'worldbank' in str(caught.value), (
            'the message should name what is available')

    def test_the_command_line_offers_what_is_registered(self):
        """An enumerated list ages in silence, in both directions."""
        import datasets  # noqa: F401
        from core.dataset_config import list_datasets

        source = (_ROOT / 'pipeline.py').read_text()
        assert '_registered_datasets()' in source
        assert "choices=['worldbank', 'inep_censo']" not in source
        assert set(list_datasets()) == {'worldbank', 'inep_censo'}


class TestTheInternalSchemaIsNeutral:
    """The internal schema spoke the first dataset's vocabulary.

    Both datasets are adapted onto one shared table, and its columns were named
    for the World Bank: municipalities were stored in `country_code`, and INEP's
    target -- the complement of upper-secondary dropout -- in a column called
    `lower_secondary_completion_rate`. The second is the worse of the two: the
    name asserts the wrong education stage, and it does so inside an artifact a
    reviewer downloads.
    """

    RETIRED = ('country_code', 'country_name', 'country_stratum',
               'lower_secondary_completion_rate')
    REPLACEMENTS = ('entity_id', 'entity_name', 'entity_stratum',
                    'target_source_rate')

    @pytest.mark.parametrize('name', RETIRED)
    def test_the_dataset_specific_name_is_gone(self, name):
        import subprocess
        # Code and schema only. Prose may name a retired column -- this test
        # does, in the list above and in the docstring saying why, and the
        # README does, explaining what the rename was for.
        #
        # One use in code is legitimate and stays: translating a pre-rename
        # artifact forward, which reads `'country_code': 'entity_id'`. The v7
        # runs on disk were written before the rename and a diagnostic that
        # reads them has to name what it is renaming. Allowed only on a line
        # that also carries the new name, so the exemption cannot cover a line
        # that merely uses the old one.
        hits = subprocess.run(
            ['git', 'grep', '-n', '-w', name, '--',
             '*.py', '*.sql', ':!tests/test_dataset_config.py'],
            cwd=_ROOT, capture_output=True, text=True).stdout.splitlines()
        replacement = dict(zip(self.RETIRED, self.REPLACEMENTS))[name]
        offending = [h for h in hits if replacement not in h]
        assert not offending, f'{name} came back in: {offending}'

    def test_both_datasets_declare_the_neutral_columns(self):
        import datasets  # noqa: F401
        from core.dataset_config import get_dataset, list_datasets

        for name in list_datasets():
            config = get_dataset(name)
            assert config.entity_column == 'entity_id', name
            assert config.entity_name_column == 'entity_name', name
            assert config.target_source_column == 'target_source_rate', name

    def test_the_provenance_of_the_target_is_still_recorded(self):
        """Neutral in the schema must not mean unstated in the source."""
        collector = (_SRC / 'collection' / 'inep_collector.py').read_text()
        assert 'abandono_em' in collector, (
            'the collector no longer says what the INEP target is derived from')
