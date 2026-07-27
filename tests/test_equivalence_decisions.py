#!/usr/bin/env python3
"""Direction of the equivalence decision.

The decision used to be labelled 'superior' whenever the interval sat above
+delta. The effect is measured as A - B, or log(A/B) for latency, so a positive
effect favours A only when a larger value of the metric is the better outcome.
It is for R2; it is not for latency, MASE or WAPE.

Published consequence: on World Bank the baseline stage recorded log(dask/duckdb)
= +0.178 -- Dask slower -- as 'superior', and on INEP the same stage recorded
-0.715 -- Dask faster -- as 'inferior'. The artifact contradicted the paper,
which states that Dask leads the ML stages on INEP.
"""

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from statistical_validation.equivalence_estimation import (HIGHER_IS_BETTER,
                                                           _advantage,
                                                           _decision_equivalence)


class TestDecisionIsDirectional:

    def test_interval_above_delta(self):
        assert _decision_equivalence(0.10, 0.20, 0.05) == 'a_exceeds_b'

    def test_interval_below_minus_delta(self):
        assert _decision_equivalence(-0.20, -0.10, 0.05) == 'b_exceeds_a'

    def test_interval_within_delta(self):
        assert _decision_equivalence(-0.01, 0.01, 0.05) == 'equivalent'

    def test_interval_straddling_delta(self):
        assert _decision_equivalence(-0.01, 0.20, 0.05) == 'inconclusive'

    def test_missing_interval(self):
        assert _decision_equivalence(float('nan'), 0.1, 0.05) == \
            'insufficient_data'

    def test_no_label_implies_merit(self):
        """A merit label cannot be assigned without knowing the metric."""
        labels = {_decision_equivalence(lo, hi, 0.05)
                  for lo, hi in [(0.1, 0.2), (-0.2, -0.1), (-0.01, 0.01),
                                 (-0.01, 0.2)]}
        assert not labels & {'superior', 'inferior'}


class TestAdvantageKnowsTheDirection:

    def test_latency_favours_the_faster_side(self):
        """log(A/B) > 0 means A took longer, so B is the better side."""
        decision = _decision_equivalence(0.169, 0.187, 0.05)
        assert _advantage(decision, 'latency', 'task_graph', 'sql_engine') == \
            'sql_engine'

    def test_latency_favours_the_faster_side_when_negative(self):
        decision = _decision_equivalence(-0.722, -0.708, 0.05)
        assert _advantage(decision, 'latency', 'task_graph', 'sql_engine') == \
            'task_graph'

    def test_r2_favours_the_larger_side(self):
        decision = _decision_equivalence(0.08, 0.12, 0.01)
        assert _advantage(decision, 'r2', 'A', 'B') == 'A'

    @pytest.mark.parametrize('metric', ['mase', 'wape'])
    def test_error_measures_favour_the_smaller_side(self, metric):
        decision = _decision_equivalence(0.08, 0.12, 0.05)
        assert _advantage(decision, metric, 'A', 'B') == 'B'

    def test_the_two_directions_disagree_on_the_same_interval(self):
        """The bug in one line: same interval, opposite winner."""
        decision = _decision_equivalence(0.08, 0.12, 0.01)
        assert _advantage(decision, 'r2', 'A', 'B') != \
            _advantage(decision, 'mase', 'A', 'B')

    @pytest.mark.parametrize('decision', ['equivalent', 'inconclusive',
                                          'insufficient_data'])
    def test_no_winner_without_a_direction(self, decision):
        assert _advantage(decision, 'latency', 'A', 'B') is None


class TestAdvantageReachesTheArtifact:
    """A directional decision without the winner leaves the reader to infer it."""

    def test_every_dict_reporting_a_decision_also_reports_the_advantage(self):
        import ast

        source = (Path(__file__).resolve().parents[1] / 'src'
                  / 'statistical_validation'
                  / 'equivalence_estimation.py').read_text()
        tree = ast.parse(source)

        reporting = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if 'decision' not in keys:
                continue
            reporting += 1
            assert 'advantage' in keys, (
                f"line {node.lineno}: reports 'decision' without 'advantage', "
                f"so the direction is recorded but the favoured side is not"
            )

        assert reporting == 2, (
            f"expected the predictive and latency result dicts, "
            f"found {reporting}"
        )


class TestDirectionTableIsComplete:

    def test_every_reported_metric_has_a_direction(self):
        assert set(HIGHER_IS_BETTER) == {'r2', 'mase', 'wape', 'latency'}

    def test_only_r2_improves_upwards(self):
        assert HIGHER_IS_BETTER['r2'] is True
        assert not any(v for k, v in HIGHER_IS_BETTER.items() if k != 'r2')

    def test_an_unknown_metric_is_not_silently_guessed(self):
        with pytest.raises(KeyError):
            _advantage('a_exceeds_b', 'throughput', 'A', 'B')


class TestThePowerNoteNamesRealPairs:
    """The note described an analysis that was not run.

    Its pair list was a literal naming pre-rename abbreviations -- dl_vs_dw,
    dl_vs_pl, dw_vs_pl -- none of which exist in the artifacts it accompanies.
    A reader checking the note against the results finds three pairs that are
    not there and three results the note does not mention.
    """

    @staticmethod
    def _note():
        import sys
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / 'src'
                  / 'statistical_validation' / 'equivalence_estimation.py')
        return source.read_text()

    def test_no_pre_rename_pair_is_named(self):
        import ast as ast_module
        tree = ast_module.parse(self._note())
        docstrings = {id(n.value) for n in ast_module.walk(tree)
                      if isinstance(n, ast_module.Expr)
                      and isinstance(n.value, ast_module.Constant)}
        for node in ast_module.walk(tree):
            if not (isinstance(node, ast_module.Constant)
                    and isinstance(node.value, str)) or id(node) in docstrings:
                continue
            for stale in ('dl_vs_dw', 'dl_vs_pl', 'dw_vs_pl'):
                assert stale not in node.value, (
                    f'line {node.lineno} names {stale!r}, a pair that does not '
                    f'exist in the artifacts this note accompanies'
                )

    def test_the_pair_list_comes_from_the_registry(self):
        """The note takes its pairs as an argument; main derives them."""
        source = self._note()
        assert 'paradigm_pairs()' in source
        assert "'power_note': power_note(n_pairs, pairs)" in source

    def test_every_registry_pair_appears_in_the_rendered_note(self):
        import sys
        from pathlib import Path
        src = str(Path(__file__).resolve().parents[1] / 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        from core.paradigm_registry import paradigm_pairs
        from statistical_validation.equivalence_estimation import power_note

        pairs = [f'{a}_vs_{b}' for a, b in paradigm_pairs()]
        rendered = power_note(10, pairs)
        for pair in pairs:
            assert pair in rendered, f'{pair} missing from the note'
        assert 'n=10' in rendered

    def test_no_pairs_no_claim(self):
        import sys
        from pathlib import Path
        src = str(Path(__file__).resolve().parents[1] / 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        from statistical_validation.equivalence_estimation import power_note
        assert 'pairwise' not in power_note(0, ['a_vs_b'])
