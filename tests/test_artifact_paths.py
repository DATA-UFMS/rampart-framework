#!/usr/bin/env python3
"""Every artifact lands under the dataset it belongs to.

The tree is segregated by dataset -- outputs/<dataset>/... -- because a World
Bank run and an INEP run write the same filenames. One script did not use the
resolver: the leakage-injection report went to the repository root's outputs/,
so the second dataset overwrote the first, and the file sat outside the tree
the reproduction package collects.

That report is the artifact demonstrating the anti-leakage gates actually fire
under injected violations. Losing it to whichever dataset ran last is losing
the demonstration for the other one.
"""

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: Files that write artifacts and must therefore resolve their paths.
WRITERS = sorted(
    path for path in list(_SRC.rglob('*.py'))
    + list((_ROOT / 'scripts').rglob('*.py'))
    if 'json.dump' in path.read_text() or 'to_csv(' in path.read_text()
)


class TestTheResolverIsUsed:

    #: The resolver reads DATASET_NAME and resolves to one dataset.
    #: derive_paper_tables crosses both panels -- comparing scales is the whole
    #: point of its table -- so it cannot use it, and appends the dataset
    #: itself. That is the only reason to build the path by hand.
    CROSS_DATASET = {'derive_paper_tables.py'}

    @pytest.mark.parametrize('path', WRITERS, ids=lambda p: p.name)
    def test_no_writer_builds_an_output_path_by_hand(self, path):
        if path.name in self.CROSS_DATASET:
            pytest.skip('reads every dataset; segregates by hand')
        """`<root> / 'outputs'` skips the per-dataset segment."""
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Div)):
                continue
            right = node.right
            if (isinstance(right, ast.Constant)
                    and right.value == 'outputs'):
                raise AssertionError(
                    f"{path.name}:{node.lineno} joins 'outputs' onto a root "
                    f"path; that skips the dataset segment and the two "
                    f"datasets overwrite each other"
                )

    def test_the_exemption_stays_small(self):
        """An exemption list is how the rule gets hollowed out."""
        assert len(self.CROSS_DATASET) == 1

    def test_the_exempt_writer_still_segregates(self):
        """By hand, but it must still do it."""
        import ast as ast_module
        source = (_ROOT / 'scripts' / 'derive_paper_tables.py').read_text()
        tree = ast_module.parse(source)
        function = next(node for node in ast_module.walk(tree)
                        if isinstance(node, ast_module.FunctionDef)
                        and node.name == '_dataset_root')
        body = ast_module.get_source_segment(source, function)
        assert "'outputs' / dataset" in body, body

    def test_the_exempt_writer_reads_more_than_one(self):
        """If it stopped crossing datasets, the exemption would expire."""
        source = (_ROOT / 'scripts' / 'derive_paper_tables.py').read_text()
        assert 'for dataset in datasets' in source

    def test_the_injection_script_uses_it(self):
        source = (_ROOT / 'scripts' / 'validation'
                  / 'leakage_injection.py').read_text()
        assert 'get_absolute_output_path' in source
        assert "_PROJECT_ROOT / 'outputs'" not in source


class TestTheSegregationHolds:

    @pytest.mark.parametrize('dataset', ['worldbank', 'inep_censo'])
    def test_the_dataset_appears_in_the_path(self, dataset, monkeypatch):
        import importlib

        import core.config as config
        monkeypatch.setenv('DATASET_NAME', dataset)
        importlib.reload(config)
        resolved = config.get_absolute_output_path('validation')
        assert f'/{dataset}/' in resolved, resolved

    def test_two_datasets_resolve_differently(self, monkeypatch):
        """Otherwise the segregation would be nominal."""
        import importlib

        import core.config as config
        resolved = {}
        for dataset in ('worldbank', 'inep_censo'):
            monkeypatch.setenv('DATASET_NAME', dataset)
            importlib.reload(config)
            resolved[dataset] = config.get_absolute_output_path('validation')
        assert len(set(resolved.values())) == 2, resolved

    def test_the_resolver_is_restored_afterwards(self):
        """The reload above must not leak into the rest of the suite."""
        import importlib

        import core.config as config
        importlib.reload(config)
        assert 'outputs' in config.get_absolute_output_path('validation')
