#!/usr/bin/env python3
"""Every generated .tex must compile, and a failed artifact must halt.

Ten modules write LaTeX. The escaping was decided independently in each, and
the ones that got it wrong produced files that do not compile -- an error that
surfaces to whoever assembles the paper, hours after the run.

Two ways it went wrong, both found by execution rather than reading:

  * hand-built rows interpolating text straight in. Every text column of the
    equivalence tables carries underscores: the pair key
    (dataframe_lib_vs_sql_engine), the phase (total_architectural), the
    decision (a_exceeds_b) and the advantage, which is a paradigm name.
  * pandas ``to_latex`` without ``escape=True``. It is not the default in
    pandas 2.x, and the frames are indexed by pair key.

And a class of silent loss: artifact writes wrapped in ``except: pass``. A
published artifact that is missing is indistinguishable from one that was
never asked for.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

WRITERS = sorted(
    path for path in list(_SRC.rglob('*.py')) + list((_ROOT / 'scripts').rglob('*.py'))
    if re.search(r"\.tex[\"']|OUT_TEX", path.read_text())
)


def _unescaped(text):
    return [line for line in text.splitlines()
            if not line.strip().startswith('%')
            and re.search(r'(?<!\\)[_%]', line)]


class TestTheWritersAreFound:

    def test_the_sweep_covers_them(self):
        assert len(WRITERS) >= 9, [p.name for p in WRITERS]


class TestPandasWritersEscape:
    """`escape=True` is not the default, and the frames carry pair keys."""

    @pytest.mark.parametrize('path', [p for p in WRITERS
                                      if 'to_latex(' in p.read_text()],
                             ids=lambda p: p.name)
    def test_to_latex_is_called_with_escape(self, path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, 'attr', '') == 'to_latex'):
                continue
            escapes = [kw for kw in node.keywords if kw.arg == 'escape']
            assert escapes, (
                f'{path.name}:{node.lineno} calls to_latex without escape=; '
                f'pandas 2.x writes the underscore raw'
            )
            assert escapes[0].value.value is True, f'{path.name}:{node.lineno}'

    def test_the_default_really_is_unsafe(self):
        """Pins the premise: if pandas changed, this test should say so."""
        import pandas as pd
        frame = pd.DataFrame({'pair': ['a_vs_b'], 'v': [1.0]})
        assert '_' in frame.to_latex(index=False).split('a')[1][:4] or \
            'a_vs_b' in frame.to_latex(index=False), (
            'pandas now escapes by default; the requirement above can be '
            'relaxed, but deliberately'
        )


class TestHandBuiltRowsEscape:

    def test_the_equivalence_tables_escape_every_text_cell(self):
        from statistical_validation.equivalence_estimation import _tex
        for value in ('dataframe_lib_vs_sql_engine', 'total_architectural',
                      'a_exceeds_b', 'sql_engine', '100%'):
            assert not _unescaped(_tex(value)), value

    def test_a_clean_value_is_left_alone(self):
        from statistical_validation.equivalence_estimation import _tex
        assert _tex('r2') == 'r2'

    @staticmethod
    def _rendered(tmp_path, monkeypatch):
        """Drive the writer and read what it actually put on disk."""
        import sys
        from core.paradigm_registry import paradigm_pairs
        from statistical_validation import equivalence_estimation as module

        monkeypatch.setattr(module, 'get_absolute_output_path',
                            lambda relative: str(tmp_path / relative))
        pair = '{}_vs_{}'.format(*paradigm_pairs()[0])
        payload = {
            'predictive': {pair: {'r2': {
                'n_pairs': 9, 'point_estimate': 0.001,
                'ci95': [-0.002, 0.004], 'delta': 0.02,
                'decision': 'a_exceeds_b',
                'advantage': paradigm_pairs()[0][0]}}},
            'latency': {'total_architectural': {pair: {
                'n_pairs': 9, 'point_estimate_lr': 0.01,
                'ci95_lr': [-0.02, 0.04], 'delta_pct': 0.15,
                'decision': 'insufficient_data',
                'advantage': None}}},
        }
        module._save_outputs(payload, write_tex=True)
        return (tmp_path / 'outputs' / 'statistics'
                / 'equivalence_estimation.tex').read_text()

    def test_the_written_file_has_no_raw_underscore(self, tmp_path,
                                                    monkeypatch):
        rendered = self._rendered(tmp_path, monkeypatch)
        assert not _unescaped(rendered), _unescaped(rendered)[:3]

    def test_both_tables_are_in_it(self, tmp_path, monkeypatch):
        """One escaped and the other not would pass a check on either alone."""
        rendered = self._rendered(tmp_path, monkeypatch)
        assert rendered.count('\\begin{tabular}') == 2, rendered.count(
            '\\begin{tabular}')

    def test_the_pair_key_reaches_the_file(self, tmp_path, monkeypatch):
        """Otherwise emitting nothing would satisfy the escaping check."""
        from core.paradigm_registry import paradigm_pairs
        rendered = self._rendered(tmp_path, monkeypatch)
        escaped = '{}_vs_{}'.format(*paradigm_pairs()[0]).replace('_', r'\_')
        assert escaped in rendered

    def test_the_decision_and_advantage_reach_it_escaped(self, tmp_path,
                                                         monkeypatch):
        rendered = self._rendered(tmp_path, monkeypatch)
        assert r'a\_exceeds\_b' in rendered
        assert r'insufficient\_data' in rendered


class TestNoArtifactWriteIsSwallowed:
    """`except: pass` around a write loses the artifact and the signal."""

    ARTIFACT_WRITERS = [
        'statistical_validation/significance_tests.py',
        'architectures_ml/sql_engine/models/hierarchical_model.py',
        'architectures_ml/task_graph/models/hierarchical_model.py',
        'architectures_ml/dataframe_lib/models/hierarchical_model.py',
    ]

    @pytest.mark.parametrize('relative', ARTIFACT_WRITERS)
    def test_no_bare_pass_handler(self, relative):
        tree = ast.parse((_SRC / relative).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = [n for n in node.body
                    if not (isinstance(n, ast.Expr)
                            and isinstance(n.value, ast.Constant))]
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                raise AssertionError(
                    f'{relative}:{node.lineno} swallows the failure; the '
                    f'artifact goes missing and the stage still exits 0'
                )

    def test_the_latency_profile_refuses_a_malformed_item(self):
        """It was dropped, and the run decided with a threshold nobody asked for."""
        from statistical_validation.equivalence_estimation import (
            _parse_latency_profile)
        with pytest.raises(ValueError, match='malformado'):
            _parse_latency_profile('setup:0.2,total-0.1', 0.15)

    def test_a_well_formed_profile_parses(self):
        from statistical_validation.equivalence_estimation import (
            _parse_latency_profile)
        parsed = _parse_latency_profile('setup:0.2, total:0.3', 0.15)
        assert parsed['setup'] == 0.2
        assert parsed['total'] == 0.3

    def test_an_empty_profile_keeps_the_defaults(self):
        from statistical_validation.equivalence_estimation import (
            _parse_latency_profile)
        assert _parse_latency_profile('', 0.15)['total'] == 0.15

    def test_a_none_profile_keeps_the_defaults(self):
        from statistical_validation.equivalence_estimation import (
            _parse_latency_profile)
        assert _parse_latency_profile(None, 0.15)['total'] == 0.15
