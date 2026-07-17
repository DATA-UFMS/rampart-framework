#!/usr/bin/env python3
"""The scorecard reads data, not the LaTeX rendering of data.

It recovered CPU and memory figures by parsing the table another script generates,
matching on lines like 'processing & task_graph' and splitting on ampersands. Two
consequences.

Any change to the table's format broke the extraction, and a bare `except: pass`
around each branch meant the value simply came back absent.

And one of the names was stale: it searched for 'processing & polars', which stopped
existing at the rename, so the third paradigm's resources were never extracted and
nothing said so. Both figures for it were reported as unavailable while the data
sat in the JSON.
"""

import ast
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import discover_paradigms

SCORECARD = _SRC / 'statistical_validation' / 'make_scorecard.py'


@pytest.fixture
def scorecard(tmp_path, monkeypatch):
    """A resource JSON holding every paradigm."""
    monkeypatch.setenv('DATASET_NAME', 'worldbank')
    import core.config as config
    monkeypatch.setattr(config, 'get_project_root', lambda: str(tmp_path))

    import importlib
    from statistical_validation import make_scorecard as module
    importlib.reload(module)

    stats = Path(config.get_absolute_output_path('outputs/statistics'))
    stats.mkdir(parents=True, exist_ok=True)
    per_phase = {
        paradigm: {'cpu_proc_mean': 50.0 + index * 10,
                   'rss_mb_mean': 300.0 + index * 100}
        for index, paradigm in enumerate(sorted(discover_paradigms()))
    }
    (stats / 'architectural_resource_usage.json').write_text(
        json.dumps({'per_phase': {'processing': per_phase}}))
    return module


class TestReadsJsonNotLatex:

    def test_no_ampersand_splitting_remains(self):
        source = SCORECARD.read_text()
        block = source[source.index('def get_resources_processing'):]
        block = block[:block.index('\ndef ', 1)]
        assert "split('&')" not in block, (
            'the figures are being recovered from a rendered table again'
        )

    def test_the_resource_json_is_the_source(self):
        source = SCORECARD.read_text()
        block = source[source.index('def get_resources_processing'):]
        block = block[:block.index('\ndef ', 1)]
        assert 'architectural_resource_usage.json' in block
        assert 'architectural_resource_usage.tex' not in block

    def test_no_silent_swallow_in_the_extraction(self):
        tree = ast.parse(SCORECARD.read_text())
        target = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == 'get_resources_processing')
        for handler in [n for n in ast.walk(target)
                        if isinstance(n, ast.ExceptHandler)]:
            caught = getattr(handler.type, 'id', None) if handler.type else None
            assert caught not in (None, 'Exception'), (
                'a swallowed failure returns the figure as absent, which is how '
                'a whole paradigm went missing without a signal'
            )


class TestEveryParadigmIsExtracted:

    def test_all_three_are_returned(self, scorecard):
        """The stale name lost one of them entirely."""
        resources = scorecard.get_resources_processing()
        assert set(resources) == set(discover_paradigms())

    def test_values_are_carried_through(self, scorecard):
        for values in scorecard.get_resources_processing().values():
            assert values['cpu_proc_mean'] is not None
            assert values['rss_mb_mean'] is not None

    def test_names_come_from_the_registry(self):
        """Checked over code literals, not the file text: the docstring that
        explains the removed pattern necessarily contains it."""
        tree = ast.parse(SCORECARD.read_text())
        docstrings = {id(node.value) for node in ast.walk(tree)
                      if isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Constant)}
        literals = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings]
        assert 'discover_paradigms' in SCORECARD.read_text()
        for literal in literals:
            assert 'polars' not in literal, (
                f'a paradigm name that stopped existing at the rename: {literal!r}'
            )

    def test_no_abbreviation_survives(self):
        """dl, dw and pl designated the pre-rename names."""
        source = SCORECARD.read_text()
        block = source[source.index('def get_resources_processing'):]
        block = block[:block.index('\ndef ', 1)]
        for token in ('cpu_dl', 'cpu_dw', 'cpu_pl', 'rss_dl', 'rss_dw', 'rss_pl'):
            assert token not in block, token

    def test_a_missing_paradigm_is_simply_absent(self, tmp_path, monkeypatch):
        """Absent from the input means absent from the output, not zero."""
        monkeypatch.setenv('DATASET_NAME', 'worldbank')
        import core.config as config
        monkeypatch.setattr(config, 'get_project_root', lambda: str(tmp_path))
        import importlib
        from statistical_validation import make_scorecard as module
        importlib.reload(module)

        stats = Path(config.get_absolute_output_path('outputs/statistics'))
        stats.mkdir(parents=True, exist_ok=True)
        only = sorted(discover_paradigms())[0]
        (stats / 'architectural_resource_usage.json').write_text(json.dumps(
            {'per_phase': {'processing': {
                only: {'cpu_proc_mean': 42.0, 'rss_mb_mean': 100.0}}}}))

        resources = module.get_resources_processing()
        assert set(resources) == {only}

    def test_absent_input_yields_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DATASET_NAME', 'worldbank')
        import core.config as config
        monkeypatch.setattr(config, 'get_project_root', lambda: str(tmp_path))
        import importlib
        from statistical_validation import make_scorecard as module
        importlib.reload(module)
        assert module.get_resources_processing() == {}

    def test_another_phase_can_be_requested(self, scorecard):
        """The phase was fixed in the function name and in the matched string."""
        assert scorecard.get_resources_processing('setup') == {}
        assert scorecard.get_resources_processing('processing')
