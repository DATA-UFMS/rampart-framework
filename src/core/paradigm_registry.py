#!/usr/bin/env python3
"""
Módulo de auto-descoberta de paradigmas.

Percorre src/architectures_ml/*/ em busca de módulos setup.py contendo
subclasses de BaseArchitectureML. A importação desses módulos dispara
o auto-registro via __init_subclass__.

Uso:
    from core.paradigm_registry import discover_paradigms
    paradigmas = discover_paradigms()  # {nome: dict PARADIGM_META, ...}
"""

import importlib
import os

_discovered = False


def discover_paradigms(*, force: bool = False, strict: bool = True) -> dict:
    """
    Percorre architectures_ml/*/ e importa cada módulo setup.
    Retorna dicionário mapeando nome do paradigma -> dict PARADIGM_META.

    Args:
        force: Se True, refaz a varredura mesmo se já descoberto (útil para testes).
    """
    global _discovered

    from core.base_architecture import BaseArchitectureML

    if force:
        _discovered = False

    if not _discovered:
        src_dir = os.path.join(os.path.dirname(__file__), '..')
        arch_dir = os.path.join(src_dir, 'architectures_ml')

        if os.path.isdir(arch_dir):
            for entry in sorted(os.listdir(arch_dir)):
                pkg_dir = os.path.join(arch_dir, entry)
                setup_file = os.path.join(pkg_dir, 'setup.py')
                if os.path.isdir(pkg_dir) and os.path.isfile(setup_file):
                    module_name = f'architectures_ml.{entry}.setup'
                    try:
                        importlib.import_module(module_name)
                    except Exception as e:
                        # A paradigm that fails to import would otherwise be
                        # absent from the comparison, silently reducing a
                        # three-way study to two.
                        if strict:
                            raise ImportError(
                                f"Paradigm module {module_name} failed to "
                                f"import: {e}"
                            ) from e
                        print(f"[WARN] Não foi possível importar {module_name}: {e}")
        _discovered = True

    return {
        name: dict(cls.PARADIGM_META)
        for name, cls in BaseArchitectureML.get_registered_paradigms().items()
    }
