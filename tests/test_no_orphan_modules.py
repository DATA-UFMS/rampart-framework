#!/usr/bin/env python3
"""Every module is reachable from an entry point.

Seven analysis scripts sat outside pipeline.py while producing artifacts that
were published -- effect sizes, the scorecard, latency and throughput
percentiles, resource usage, the operational panel, bootstrap sensitivity. All
seven ran correctly; nothing invoked them. Reproducing the published results
required knowing a sequence that existed nowhere.

A module that is neither reachable from the pipeline nor listed as a declared
tool is either dead code or a missing stage. This test forces that choice to be
made explicitly.
"""

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Entry points that are not the pipeline, each with the reason it stands alone.
DECLARED_TOOLS = {
    'scripts/validation/leakage_injection.py':
        'negative validation of the anti-leakage gate; invoked by CI',
}


def _module_paths():
    return sorted(p for p in _SRC.rglob('*.py') if p.name != '__init__.py')


def _local_modules():
    """Dotted name -> path, for every module under src/."""
    out = {}
    for path in _SRC.rglob('*.py'):
        parts = [p for p in path.relative_to(_SRC).with_suffix('').parts
                 if p != '__init__']
        out['.'.join(parts)] = path
    return out


def _imported_names(path):
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def _entry_points():
    from core.paradigm_registry import discover_paradigms

    entries = {_ROOT / 'pipeline.py'}
    for meta in discover_paradigms().values():
        for key in ('processor_script', 'setup_script', 'baseline_script',
                    'hierarchical_script'):
            if key in meta:
                entries.add(_ROOT / meta[key])
    # Scripts the pipeline invokes as subprocesses; the import graph cannot see
    # a subprocess call, so they are seeded from the pipeline's own source.
    source = (_ROOT / 'pipeline.py').read_text()
    for path in _module_paths():
        rel = str(path.relative_to(_ROOT))
        if rel in source:
            entries.add(path)
    for rel in DECLARED_TOOLS:
        entries.add(_ROOT / rel)
    return entries


def _reachable():
    local = _local_modules()
    seen, stack = set(), list(_entry_points())
    while stack:
        path = Path(stack.pop())
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for name in _imported_names(path):
            candidates = [name]
            if '.' in name:
                candidates.append(name.rsplit('.', 1)[0])
            for candidate in candidates:
                target = local.get(candidate)
                if target and target not in seen:
                    stack.append(target)
    return seen


def test_entry_points_were_found():
    entries = _entry_points()
    assert _ROOT / 'pipeline.py' in entries
    assert len(entries) > 5, f'suspiciously few entry points: {entries}'


def test_declared_tools_exist():
    """A stale entry here would silently exempt nothing."""
    for rel, reason in DECLARED_TOOLS.items():
        assert (_ROOT / rel).exists(), (
            f'{rel} is declared as a tool ({reason}) but does not exist'
        )


@pytest.mark.parametrize('path', _module_paths(),
                         ids=lambda p: str(p.relative_to(_SRC)))
def test_module_is_reachable(path):
    reachable = {p.resolve() for p in _reachable()}
    assert path.resolve() in reachable, (
        f"{path.relative_to(_ROOT)} is not reachable from pipeline.py nor "
        f"listed in DECLARED_TOOLS. Either wire it into the pipeline, or "
        f"declare it as a tool with the reason it stands alone, or delete it."
    )
