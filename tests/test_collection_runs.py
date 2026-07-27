#!/usr/bin/env python3
"""The collection is EXECUTED, not inspected as text.

Reason for existing: the imputation was rewritten and was left with a dangling
reference to two variables that the rewrite removed from the scope. Any panel
with a missing cell raised NameError on the first column. The 34 tests in
test_imputation_scope.py passed green because they validate the function
entirely through COLLECTOR.read_text() and substring search -- they confirm that
the textual deletion happened, not that what was left runs.

None of the 715 tests executed apply_conservative_imputation. These do.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

TARGET = 'target_source_rate'


def _collector(tmp_path):
    """No-__init__ instance: it prints, reads the network, builds real paths."""
    import collection.raw_data_collector as module

    cls = next(getattr(module, name) for name in dir(module)
               if isinstance(getattr(module, name), type)
               and hasattr(getattr(module, name), 'apply_conservative_imputation'))
    instance = cls.__new__(cls)
    instance.output_dir = str(tmp_path)
    instance.indicator_to_category = {}
    instance.target_source_column = TARGET
    return instance


def _panel(n_years=6, gap_at=2002):
    """Panel with gaps: without them the imputation is not exercised."""
    rows = []
    for entity in ('AAA', 'BBB'):
        for year in range(2000, 2000 + n_years):
            rows.append({'entity_id': entity, 'year': year})
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(5)
    frame[TARGET] = rng.uniform(5.0, 25.0, len(frame))
    frame['gini_index'] = rng.uniform(30.0, 55.0, len(frame))
    frame['internet_users_percent'] = rng.uniform(10.0, 90.0, len(frame))
    # Gaps in two features, one of them in the middle of the series.
    frame.loc[frame['year'] == gap_at, 'gini_index'] = np.nan
    frame.loc[(frame['entity_id'] == 'AAA') & (frame['year'] == 2001),
              'internet_users_percent'] = np.nan
    return frame


class TestImputationExecutes:
    """Each one of these would have failed with the NameError."""

    def test_it_returns_a_frame(self, tmp_path):
        result = _collector(tmp_path).apply_conservative_imputation(_panel())
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_a_panel_with_gaps_does_not_raise(self, tmp_path):
        """The exact condition that broke: a column with a missing cell."""
        panel = _panel()
        assert panel['gini_index'].isna().any(), 'fixture without a gap'
        _collector(tmp_path).apply_conservative_imputation(panel)

    def test_a_panel_without_gaps_also_runs(self, tmp_path):
        """The path that skips the column, so as not to go only through the
        happy branch.
        """
        panel = _panel()
        panel['gini_index'] = 40.0
        panel['internet_users_percent'] = 50.0
        _collector(tmp_path).apply_conservative_imputation(panel)

    def test_the_log_is_written_and_readable(self, tmp_path):
        _collector(tmp_path).apply_conservative_imputation(_panel())
        log = json.loads(
            (tmp_path / 'scientific_imputation_log.json').read_text())
        assert log['imputation_log'], 'empty log'

    def test_the_log_records_a_single_mechanism(self, tmp_path):
        """The cross-sectional tiers are gone; the log must not suggest a
        choice.
        """
        _collector(tmp_path).apply_conservative_imputation(_panel())
        log = json.loads(
            (tmp_path / 'scientific_imputation_log.json').read_text())
        for column, entry in log['imputation_log'].items():
            assert entry['method_used'] == 'entity_forward_fill', column
            assert entry['geographic_count'] == 0, column
            assert entry['global_count'] == 0, column

    def test_coverage_is_written(self, tmp_path):
        _collector(tmp_path).apply_conservative_imputation(_panel())
        coverage = json.loads((tmp_path / 'target_coverage.json').read_text())
        assert coverage['target_source_column'] == TARGET
        assert coverage['rows_after'] <= coverage['rows_before']

    def test_the_target_is_not_imputed(self, tmp_path):
        """Executed, not verified by substring."""
        panel = _panel()
        panel.loc[panel['year'] == 2003, TARGET] = np.nan
        before = panel[TARGET].notna().sum()
        result = _collector(tmp_path).apply_conservative_imputation(panel)
        assert len(result) == before, (
            'rows without an observed target should be dropped, not filled in'
        )
        assert result[TARGET].notna().all()

    def test_forward_fill_uses_only_the_entity_past(self, tmp_path):
        """One entity must not receive a value from another."""
        panel = _panel()
        panel.loc[panel['entity_id'] == 'AAA', 'gini_index'] = np.nan
        panel.loc[panel['entity_id'] == 'BBB', 'gini_index'] = 99.0
        result = _collector(tmp_path).apply_conservative_imputation(panel)
        filled = result[result['entity_id'] == 'AAA']['gini_index'].dropna()
        assert (filled != 99.0).all(), (
            'a value from another entity leaked into AAA'
        )


class TestNoDanglingReference:
    """The class of defect, not just the instance."""

    def test_no_name_is_used_before_assignment_in_the_imputation(self):
        """Every name read inside the function has to be bound somewhere.

        It collects the targets as Name in Store context, which covers in one go
        tuple unpacking, for targets, comprehensions and `with ... as`; plus the
        names from `except ... as`, the arguments (including those of nested
        lambdas) and the module scope.
        """
        import ast
        import builtins

        source = (_SRC / 'collection' / 'raw_data_collector.py').read_text()
        tree = ast.parse(source)
        function = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)
                        and n.name == 'apply_conservative_imputation')

        bound = {n.id for n in ast.walk(function)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        bound |= {h.name for h in ast.walk(function)
                  if isinstance(h, ast.ExceptHandler) and h.name}
        for node in ast.walk(function):
            if isinstance(node, (ast.FunctionDef, ast.Lambda)):
                args = node.args
                bound |= {a.arg for a in
                          args.posonlyargs + args.args + args.kwonlyargs}
                for extra in (args.vararg, args.kwarg):
                    if extra:
                        bound.add(extra.arg)

        module_scope = {n.id for node in ast.walk(tree)
                        if isinstance(node, ast.Assign)
                        for n in ast.walk(node) if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Store)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_scope |= {(a.asname or a.name.split('.')[0])
                                 for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                module_scope |= {(a.asname or a.name) for a in node.names}
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                module_scope.add(node.name)

        used = {n.id for n in ast.walk(function)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        dangling = sorted(used - bound - module_scope - set(dir(builtins)))
        assert not dangling, (
            f'names read and never bound in apply_conservative_imputation: '
            f'{dangling} -- this is the class of defect that killed the '
            f'collection'
        )
