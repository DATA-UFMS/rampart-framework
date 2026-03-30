#!/usr/bin/env python3
"""
Testes de auto-descoberta do framework de paradigmas ML.

Valida o mecanismo __init_subclass__ de BaseArchitectureML:
- registro automático de subclasses concretas com PARADIGM_META
- rejeição de subclasses abstratas intermediárias
- TypeError para subclasses concretas sem PARADIGM_META
- descoberta dos três paradigmas reais após importação dos módulos
- presença das chaves obrigatórias em cada PARADIGM_META
"""

import pytest
from abc import abstractmethod
from core.base_architecture import BaseArchitectureML

# ---------------------------------------------------------------------------
# Helpers: stubs mínimos para criar subclasses de teste sem instanciar
# ---------------------------------------------------------------------------

_ABSTRACT_METHOD_NAMES = [
    'setup_environment',
    'load_data',
    'validate_data',
    'create_target_implementation',
    '_compute_target_statistics',
    '_validate_temporal_folds',
    'save_folds',
    'compute_feature_correlations',
    'apply_collinearity_filter',
    'get_numeric_features',
    'prepare_features',
]


def _make_concrete_class(name: str, paradigm_meta: dict) -> type:
    """Retorna uma subclasse completamente concreta de BaseArchitectureML com PARADIGM_META."""
    stubs = {m: (lambda self, *a, **kw: None) for m in _ABSTRACT_METHOD_NAMES}
    stubs['PARADIGM_META'] = paradigm_meta
    return type(name, (BaseArchitectureML,), stubs)


def _make_abstract_intermediate(name: str) -> type:
    """Retorna uma subclasse intermediária abstrata (mantém pelo menos um método abstrato)."""
    # Implementa todos exceto um método abstrato para que a classe ainda seja abstrata
    stubs = {m: (lambda self, *a, **kw: None) for m in _ABSTRACT_METHOD_NAMES[1:]}

    @abstractmethod
    def setup_environment(self) -> None:
        pass

    stubs['setup_environment'] = setup_environment
    return type(name, (BaseArchitectureML,), stubs)


# ---------------------------------------------------------------------------
# Teste 1 — _registry existe em BaseArchitectureML
# ---------------------------------------------------------------------------

class TestBaseClassHasRegistry:
    def test_base_class_has_registry(self):
        assert hasattr(BaseArchitectureML, '_registry')
        assert isinstance(BaseArchitectureML._registry, dict)

    def test_get_registered_paradigms_returns_dict(self):
        result = BaseArchitectureML.get_registered_paradigms()
        assert isinstance(result, dict)

    def test_get_registered_paradigms_is_copy(self):
        """Modificar o dicionário retornado não deve corromper o registro."""
        result = BaseArchitectureML.get_registered_paradigms()
        original_len = len(BaseArchitectureML._registry)
        result['__test_key__'] = object()
        assert len(BaseArchitectureML._registry) == original_len


# ---------------------------------------------------------------------------
# Teste 2 — subclasse concreta com PARADIGM_META registra automaticamente
# ---------------------------------------------------------------------------

class TestConcreteSubclassRegistersAutomatically:
    def test_concrete_subclass_registers_automatically(self):
        meta = {'name': '__test_paradigm_auto__', 'label': 'Auto Test'}
        cls = _make_concrete_class('_AutoRegisterTestClass', meta)
        assert '__test_paradigm_auto__' in BaseArchitectureML._registry
        assert BaseArchitectureML._registry['__test_paradigm_auto__'] is cls

    def teardown_method(self, method):
        BaseArchitectureML._registry.pop('__test_paradigm_auto__', None)


# ---------------------------------------------------------------------------
# Teste 3 — subclasse intermediária abstrata NÃO registra
# ---------------------------------------------------------------------------

class TestAbstractSubclassDoesNotRegister:
    def test_abstract_subclass_does_not_register(self):
        registry_before = set(BaseArchitectureML._registry.keys())
        _make_abstract_intermediate('_AbstractIntermediateTestClass')
        registry_after = set(BaseArchitectureML._registry.keys())
        assert registry_after == registry_before, (
            f"Registry grew unexpectedly: {registry_after - registry_before}"
        )


# ---------------------------------------------------------------------------
# Teste 4 — subclasse concreta sem PARADIGM_META é ignorada silenciosamente
# ---------------------------------------------------------------------------

class TestSubclassWithoutMetaIsSkipped:
    """Subclasse concreta SEM PARADIGM_META é ignorada silenciosamente (não é um paradigma)."""

    def test_concrete_without_meta_does_not_register(self):
        stubs = {m: (lambda self, *a, **kw: None) for m in _ABSTRACT_METHOD_NAMES}
        before = set(BaseArchitectureML._registry.keys())
        type('_NoMetaConcreteClass', (BaseArchitectureML,), stubs)
        after = set(BaseArchitectureML._registry.keys())
        assert before == after, "Class without PARADIGM_META should not register"

    def test_subclass_with_meta_missing_name_raises(self):
        stubs = {m: (lambda self, *a, **kw: None) for m in _ABSTRACT_METHOD_NAMES}
        stubs['PARADIGM_META'] = {'label': 'Missing name key'}
        with pytest.raises(TypeError, match='PARADIGM_META'):
            type('_NoNameMetaConcreteClass', (BaseArchitectureML,), stubs)

    def test_duplicate_name_raises(self):
        """Dois paradigmas com o mesmo nome devem lançar TypeError."""
        meta = {'name': '__dup_test__', 'label': 'First'}
        _make_concrete_class('_DupFirst', meta)
        try:
            with pytest.raises(TypeError, match='já está registrado'):
                _make_concrete_class('_DupSecond', {'name': '__dup_test__', 'label': 'Second'})
        finally:
            BaseArchitectureML._registry.pop('__dup_test__', None)


# ---------------------------------------------------------------------------
# Teste 5 — os três paradigmas reais são descobertos após importação
# ---------------------------------------------------------------------------

class TestAllThreeParadigmsDiscovered:
    def test_all_three_paradigms_discovered(self):
        # Importar os módulos aciona o registro via __init_subclass__
        import architectures_ml.data_lake.setup  # noqa: F401
        import architectures_ml.data_warehouse.setup  # noqa: F401
        import architectures_ml.polars_dataframe.setup  # noqa: F401

        registry = BaseArchitectureML.get_registered_paradigms()
        assert 'data_lake' in registry, "data_lake not in registry"
        assert 'data_warehouse' in registry, "data_warehouse not in registry"
        assert 'polars_dataframe' in registry, "polars_dataframe not in registry"

    def test_registry_values_are_classes(self):
        import architectures_ml.data_lake.setup  # noqa: F401
        import architectures_ml.data_warehouse.setup  # noqa: F401
        import architectures_ml.polars_dataframe.setup  # noqa: F401

        registry = BaseArchitectureML.get_registered_paradigms()
        for name, cls in registry.items():
            assert isinstance(cls, type), f"registry['{name}'] is not a class"
            assert issubclass(cls, BaseArchitectureML), (
                f"registry['{name}'] is not a subclass of BaseArchitectureML"
            )


# ---------------------------------------------------------------------------
# Teste 6 — cada PARADIGM_META possui as chaves obrigatórias
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    'name', 'label',
    'setup_script', 'processor_script', 'baseline_script', 'hierarchical_script',
    'processor_module', 'processor_class', 'processor_run_method',
    'baseline_module', 'baseline_class',
    'hierarchical_module', 'hierarchical_class',
}


class TestEachParadigmMetaHasRequiredKeys:
    @pytest.fixture(autouse=True)
    def _import_paradigms(self):
        import architectures_ml.data_lake.setup  # noqa: F401
        import architectures_ml.data_warehouse.setup  # noqa: F401
        import architectures_ml.polars_dataframe.setup  # noqa: F401

    def test_data_lake_meta_keys(self):
        from architectures_ml.data_lake.setup import DataLakeArchitectureML
        meta = DataLakeArchitectureML.PARADIGM_META
        missing = REQUIRED_KEYS - set(meta.keys())
        assert not missing, f"data_lake PARADIGM_META missing keys: {missing}"

    def test_data_warehouse_meta_keys(self):
        from architectures_ml.data_warehouse.setup import DataWarehouseArchitectureML
        meta = DataWarehouseArchitectureML.PARADIGM_META
        missing = REQUIRED_KEYS - set(meta.keys())
        assert not missing, f"data_warehouse PARADIGM_META missing keys: {missing}"

    def test_polars_dataframe_meta_keys(self):
        from architectures_ml.polars_dataframe.setup import PolarsDataFrameArchitectureML
        meta = PolarsDataFrameArchitectureML.PARADIGM_META
        missing = REQUIRED_KEYS - set(meta.keys())
        assert not missing, f"polars_dataframe PARADIGM_META missing keys: {missing}"

    def test_all_paradigms_in_registry_have_required_keys(self):
        registry = BaseArchitectureML.get_registered_paradigms()
        for paradigm_name, cls in registry.items():
            meta = getattr(cls, 'PARADIGM_META', {})
            missing = REQUIRED_KEYS - set(meta.keys())
            assert not missing, (
                f"PARADIGM_META for '{paradigm_name}' is missing keys: {missing}"
            )

    def test_meta_name_matches_registry_key(self):
        registry = BaseArchitectureML.get_registered_paradigms()
        for key, cls in registry.items():
            meta = getattr(cls, 'PARADIGM_META', {})
            assert meta.get('name') == key, (
                f"PARADIGM_META['name']='{meta.get('name')}' does not match "
                f"registry key '{key}' for class {cls.__name__}"
            )


class TestDiscoverParadigms:
    """Testa o módulo de descoberta automática."""

    def test_discover_returns_dict_with_all_paradigms(self):
        from core.paradigm_registry import discover_paradigms
        paradigms = discover_paradigms()
        assert isinstance(paradigms, dict)
        expected = {'data_lake', 'data_warehouse', 'polars_dataframe'}
        assert expected.issubset(set(paradigms.keys())), \
            f"Missing paradigms: {expected - set(paradigms.keys())}"
        # Detectar entradas de teste vazadas
        unexpected = set(paradigms.keys()) - expected
        assert not unexpected, f"Unexpected paradigms in registry (test leak?): {unexpected}"

    def test_discover_returns_script_paths(self):
        from core.paradigm_registry import discover_paradigms
        for name, meta in discover_paradigms().items():
            assert 'setup_script' in meta
            assert 'processor_script' in meta
            assert 'baseline_script' in meta
            assert 'hierarchical_script' in meta
            assert 'label' in meta

    def test_discover_is_idempotent(self):
        from core.paradigm_registry import discover_paradigms
        p1 = discover_paradigms()
        p2 = discover_paradigms()
        assert set(p1.keys()) == set(p2.keys())


class TestFrameworkContract:
    """Verifica o contrato completo: descoberta → registro → execução."""

    def test_each_setup_module_has_main(self):
        """Cada paradigma deve ter setup.py com main()."""
        import importlib
        from core.paradigm_registry import discover_paradigms
        for name, meta in discover_paradigms().items():
            # Usa o mesmo namespace de módulo que paradigm_registry (sem prefixo src.)
            mod_path = f'architectures_ml.{name}.setup'
            mod = importlib.import_module(mod_path)
            assert hasattr(mod, 'main'), f"{name} setup module missing main()"

    def test_each_baseline_class_has_run_complete_analysis(self):
        import importlib
        from core.paradigm_registry import discover_paradigms
        for name, meta in discover_paradigms().items():
            mod = importlib.import_module(meta['baseline_module'])
            cls = getattr(mod, meta['baseline_class'])
            assert hasattr(cls, 'run_complete_analysis'), \
                f"{name} baseline class missing run_complete_analysis()"

    def test_each_hierarchical_class_has_run_hierarchical_analysis(self):
        import importlib
        from core.paradigm_registry import discover_paradigms
        for name, meta in discover_paradigms().items():
            mod = importlib.import_module(meta['hierarchical_module'])
            cls = getattr(mod, meta['hierarchical_class'])
            assert hasattr(cls, 'run_hierarchical_analysis'), \
                f"{name} hierarchical class missing run_hierarchical_analysis()"

    def test_zero_paradigm_names_hardcoded_in_pipeline(self):
        """pipeline.py não deve conter nomes de paradigmas hardcoded."""
        import re, os
        pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'pipeline.py')
        with open(pipeline_path) as f:
            content = f.read()
        hardcoded = re.findall(
            r"['\"](?:data_lake|data_warehouse|polars_dataframe)['\"]",
            content
        )
        assert len(hardcoded) == 0, \
            f"pipeline.py still has hardcoded paradigm names: {hardcoded}"
