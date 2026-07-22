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
from typing import Dict, List, Tuple

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


#: Phases the three paradigms each execute, and which a cross-paradigm table
#: may therefore compare. Collection is deliberately absent: it runs once,
#: upstream of the paradigms, and its row carries the sentinel architecture
#: below rather than a paradigm name. A table that includes it reports the
#: cost of fetching the data as though it were a property of one engine.
COMPARABLE_PHASES = ("processing", "setup", "baseline", "hierarchical")

#: Architecture recorded for work that is not attributable to any paradigm.
SHARED_ARCHITECTURE = "both"


def comparable_rows(frame):
    """Restrict a benchmark frame to comparable phases and real paradigms.

    Two things reached the published throughput table through the absence of
    this: the collection phase, and a row labelled with the sentinel above
    standing among the three paradigms as if it were a fourth.

    An unknown architecture raises rather than being dropped. Silently
    discarding rows is how a paradigm disappears from a table -- which is the
    defect this function exists to prevent, in the other direction.
    """
    known = set(discover_paradigms()) | {SHARED_ARCHITECTURE}
    unknown = sorted(set(frame["architecture"].unique()) - known)
    if unknown:
        raise ValueError(
            f"O CSV de benchmark traz arquiteturas que o registro não conhece: "
            f"{unknown}. Ou o registro está incompleto, ou o artefato veio de "
            f"outra configuração."
        )

    restricted = frame[
        frame["phase"].isin(COMPARABLE_PHASES)
        & (frame["architecture"] != SHARED_ARCHITECTURE)
    ].copy()
    if restricted.empty:
        raise ValueError(
            f"Nenhuma linha comparável no CSV de benchmark. Fases presentes: "
            f"{sorted(frame['phase'].unique())}; comparáveis: "
            f"{list(COMPARABLE_PHASES)}."
        )
    return restricted


def paradigm_pairs(*, force: bool = False) -> List[Tuple[str, str]]:
    """Ordered pairs of paradigms for pairwise comparison.

    Derived from the registry rather than enumerated. The paradigm names appeared
    in eleven files, so a fourth paradigm was compared only after each analysis
    module was edited by hand -- which is the opposite of what an extensible
    framework claims.

    Ordering is lexicographic, and it is load-bearing: the effect of a pair is
    measured as A minus B, so swapping a pair flips the sign of every estimate
    involving it. Lexicographic order is arbitrary but stable, whereas the
    previous hand-written order was arbitrary and undocumented. The 'advantage'
    field reported alongside each decision is invariant to it.

    Returns:
        Pairs (a, b) with a < b, covering every combination exactly once.
    """
    names = sorted(discover_paradigms(force=force))
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]


def baseline_results_paths(*, force: bool = False) -> Dict[str, str]:
    """Absolute path of each paradigm's baseline results, keyed by paradigm.

    The three paradigms write to different layouts, so the location is declared
    in PARADIGM_META instead of reconstructed by the reader.
    """
    from core.config import get_absolute_output_path

    paths = {}
    for name, meta in sorted(discover_paradigms(force=force).items()):
        relative = meta.get('baseline_results_json')
        if not relative:
            raise KeyError(
                f"Paradigm {name} does not declare 'baseline_results_json' in "
                f"PARADIGM_META, so its baseline results cannot be located."
            )
        paths[name] = get_absolute_output_path(relative)
    return paths
