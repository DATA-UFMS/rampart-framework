#!/usr/bin/env python3
"""
Auto-discovery tests for the ML paradigm framework.

Validates BaseArchitectureML's __init_subclass__ mechanism:
- automatic registration of concrete subclasses with PARADIGM_META
- rejection of intermediate abstract subclasses
- TypeError for concrete subclasses without PARADIGM_META
- discovery of the three real paradigms after importing the modules
- presence of the mandatory keys in each PARADIGM_META
"""

import pytest
from abc import abstractmethod
from core.base_architecture import BaseArchitectureML

# ---------------------------------------------------------------------------
# Helpers: minimal stubs to create test subclasses without instantiating
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
    'discover_numeric_columns',
    'prepare_features',
]


def _make_concrete_class(name: str, paradigm_meta: dict) -> type:
    """Returns a fully concrete subclass of BaseArchitectureML with PARADIGM_META."""
    stubs = {m: (lambda self, *a, **kw: None) for m in _ABSTRACT_METHOD_NAMES}
    stubs['PARADIGM_META'] = paradigm_meta
    return type(name, (BaseArchitectureML,), stubs)


def _make_abstract_intermediate(name: str) -> type:
    """Returns an intermediate abstract subclass (keeps at least one abstract method)."""
    # Implements all but one abstract method so that the class is still abstract
    stubs = {m: (lambda self, *a, **kw: None) for m in _ABSTRACT_METHOD_NAMES[1:]}

    @abstractmethod
    def setup_environment(self) -> None:
        pass

    stubs['setup_environment'] = setup_environment
    return type(name, (BaseArchitectureML,), stubs)


# ---------------------------------------------------------------------------
# Test 1 — _registry exists on BaseArchitectureML
# ---------------------------------------------------------------------------

class TestBaseClassHasRegistry:
    def test_base_class_has_registry(self):
        assert hasattr(BaseArchitectureML, '_registry')
        assert isinstance(BaseArchitectureML._registry, dict)

    def test_get_registered_paradigms_returns_dict(self):
        result = BaseArchitectureML.get_registered_paradigms()
        assert isinstance(result, dict)

    def test_get_registered_paradigms_is_copy(self):
        """Modifying the returned dictionary must not corrupt the registry."""
        result = BaseArchitectureML.get_registered_paradigms()
        original_len = len(BaseArchitectureML._registry)
        result['__test_key__'] = object()
        assert len(BaseArchitectureML._registry) == original_len


# ---------------------------------------------------------------------------
# Test 2 — a concrete subclass with PARADIGM_META registers automatically
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
# Test 3 — an intermediate abstract subclass does NOT register
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
# Test 4 — a concrete subclass without PARADIGM_META is silently ignored
# ---------------------------------------------------------------------------

class TestSubclassWithoutMetaIsSkipped:
    """A concrete subclass WITHOUT PARADIGM_META is silently ignored (it is not a paradigm)."""

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
        """Two paradigms with the same name must raise TypeError."""
        meta = {'name': '__dup_test__', 'label': 'First'}
        _make_concrete_class('_DupFirst', meta)
        try:
            with pytest.raises(TypeError, match='is already registered by'):
                _make_concrete_class('_DupSecond', {'name': '__dup_test__', 'label': 'Second'})
        finally:
            BaseArchitectureML._registry.pop('__dup_test__', None)


# ---------------------------------------------------------------------------
# Test 5 — the three real paradigms are discovered after importing
# ---------------------------------------------------------------------------

class TestAllThreeParadigmsDiscovered:
    def test_all_three_paradigms_discovered(self):
        # Importing the modules triggers registration via __init_subclass__
        import architectures_ml.task_graph.setup  # noqa: F401
        import architectures_ml.sql_engine.setup  # noqa: F401
        import architectures_ml.dataframe_lib.setup  # noqa: F401

        registry = BaseArchitectureML.get_registered_paradigms()
        assert 'task_graph' in registry, "task_graph not in registry"
        assert 'sql_engine' in registry, "sql_engine not in registry"
        assert 'dataframe_lib' in registry, "dataframe_lib not in registry"

    def test_registry_values_are_classes(self):
        import architectures_ml.task_graph.setup  # noqa: F401
        import architectures_ml.sql_engine.setup  # noqa: F401
        import architectures_ml.dataframe_lib.setup  # noqa: F401

        registry = BaseArchitectureML.get_registered_paradigms()
        for name, cls in registry.items():
            assert isinstance(cls, type), f"registry['{name}'] is not a class"
            assert issubclass(cls, BaseArchitectureML), (
                f"registry['{name}'] is not a subclass of BaseArchitectureML"
            )


# ---------------------------------------------------------------------------
# Test 6 — each PARADIGM_META has the mandatory keys
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
        import architectures_ml.task_graph.setup  # noqa: F401
        import architectures_ml.sql_engine.setup  # noqa: F401
        import architectures_ml.dataframe_lib.setup  # noqa: F401

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
    """Tests the automatic discovery module."""

    def test_discover_returns_dict_with_all_paradigms(self):
        from core.paradigm_registry import discover_paradigms
        paradigms = discover_paradigms()
        assert isinstance(paradigms, dict)
        expected = {'task_graph', 'sql_engine', 'dataframe_lib'}
        assert expected.issubset(set(paradigms.keys())), \
            f"Missing paradigms: {expected - set(paradigms.keys())}"
        # Extra paradigms are legitimate; the registry exists to allow them.
        # A paradigm declared outside src/architectures_ml/ is not.
        for name, meta in paradigms.items():
            setup_script = str(meta.get('setup_script', ''))
            assert setup_script.startswith('src/architectures_ml/'), \
                (f"Paradigm '{name}' is not declared under src/architectures_ml/ "
                 f"(setup_script={setup_script!r}); possible test leak into the registry")

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
    """Checks the complete contract: discovery → registration → execution."""

    def test_each_setup_module_has_main(self):
        """Each paradigm must have a setup.py with main()."""
        import importlib
        from core.paradigm_registry import discover_paradigms
        for name, meta in discover_paradigms().items():
            # Uses the same module namespace as paradigm_registry (no src. prefix)
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
        """pipeline.py must not contain hardcoded paradigm names."""
        import re, os
        pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'pipeline.py')
        with open(pipeline_path) as f:
            content = f.read()
        hardcoded = re.findall(
            r"['\"](?:task_graph|sql_engine|dataframe_lib)['\"]",
            content
        )
        assert len(hardcoded) == 0, \
            f"pipeline.py still has hardcoded paradigm names: {hardcoded}"
