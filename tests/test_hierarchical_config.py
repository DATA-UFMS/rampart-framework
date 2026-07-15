#!/usr/bin/env python3
"""The hierarchical search space comes from the configuration.

The alpha grid, the residual shrinkage grid, the forest depth and leaf grids and
the fixed forest parameters were hard-coded in all three paradigms. They were
identical, so nothing was wrong yet; three copies of the search space can drift,
and paradigms searching different spaces are not fitting the same model, which is
the premise the equivalence check rests on.

They were also absent from the reproducibility snapshot, which records
SCIENTIFIC_CONFIG.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.scientific_config import SCIENTIFIC_CONFIG

MODELS = sorted((_SRC / 'architectures_ml').glob('*/models/hierarchical_model.py'))

# Literals that define the search space. Any of these appearing in a paradigm
# means the value exists in two places.
FORBIDDEN_LITERALS = {
    'ridge alpha grid': 'logspace(-1, 3, 20)',
    'shrinkage grid': '[0.6, 0.8, 1.0]',
    'forest depth grid': '[5, 6, 7]',
    'forest leaf grid': '[5, 8, 12]',
    'n_estimators': 'n_estimators=200',
    'min_samples_split': 'min_samples_split=15',
    "max_features": "max_features='sqrt'",
}


def test_models_were_found():
    assert len(MODELS) == 3, [str(m) for m in MODELS]


class TestNoParadigmCarriesItsOwnGrid:

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    @pytest.mark.parametrize('name,literal', sorted(FORBIDDEN_LITERALS.items()))
    def test_literal_absent(self, path, name, literal):
        source = path.read_text()
        assert literal not in source, (
            f"{path.parts[-3]}: the {name} appears as the literal {literal!r}. "
            f"Read it from SCIENTIFIC_CONFIG['hierarchical_model'] so the three "
            f"paradigms cannot diverge."
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_reads_the_configured_space(self, path):
        assert "SCIENTIFIC_CONFIG['hierarchical_model']" in path.read_text()


class TestConfigIsUsableAndComplete:

    def test_every_parameter_is_present(self):
        expected = {
            'ridge_alpha_log10_start', 'ridge_alpha_log10_stop',
            'ridge_alpha_count', 'residual_shrinkage_grid',
            'rf_max_depth_grid', 'rf_min_samples_leaf_grid',
            'rf_n_estimators', 'rf_min_samples_split', 'rf_max_features',
            'rf_n_jobs', 'ridge_cv_folds',
        }
        assert set(SCIENTIFIC_CONFIG['hierarchical_model']) == expected

    def test_alpha_grid_is_well_formed(self):
        hm = SCIENTIFIC_CONFIG['hierarchical_model']
        grid = np.logspace(hm['ridge_alpha_log10_start'],
                           hm['ridge_alpha_log10_stop'],
                           hm['ridge_alpha_count'])
        assert len(grid) == hm['ridge_alpha_count']
        assert grid[0] < grid[-1]
        assert (grid > 0).all(), 'a ridge penalty must be positive'

    def test_grids_are_non_empty(self):
        hm = SCIENTIFIC_CONFIG['hierarchical_model']
        for key in ('residual_shrinkage_grid', 'rf_max_depth_grid',
                    'rf_min_samples_leaf_grid'):
            assert len(hm[key]) >= 1, key

    def test_forest_is_single_threaded(self):
        """Parallel trees would make latency depend on core availability."""
        assert SCIENTIFIC_CONFIG['hierarchical_model']['rf_n_jobs'] == 1

    def test_search_space_reaches_the_snapshot(self):
        """The snapshot serialises SCIENTIFIC_CONFIG, so it must be JSON-safe."""
        import json
        payload = json.dumps({'scientific_config': SCIENTIFIC_CONFIG},
                             default=str)
        assert 'hierarchical_model' in payload
        assert 'rf_max_depth_grid' in payload


class TestScopeOfTheConfigLookup:
    """A lookup inside a conditional branch raises when the branch is skipped."""

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_lookup_precedes_every_use(self, path):
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)]:
            uses = [n.lineno for n in ast.walk(fn)
                    if isinstance(n, ast.Name) and n.id == '_hm'
                    and isinstance(n.ctx, ast.Load)]
            if not uses:
                continue
            # Assigned in the function body itself, not nested in a branch.
            direct = [s.lineno for s in fn.body if isinstance(s, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == '_hm'
                              for t in s.targets)]
            assert direct, (
                f"{path.parts[-3]}.{fn.name}: uses _hm at {uses} but assigns it "
                f"only inside a nested block, so a skipped branch raises "
                f"NameError"
            )
            assert min(direct) < min(uses)


class TestModelsRunWithTheConfiguredSpace:
    """Exercised in runtime: a scope error is invisible to a static check."""

    @staticmethod
    def _panel(n=60, seed=5):
        rng = np.random.default_rng(seed)
        entities = np.array(['BRA', 'ARG', 'CHL'] * (n // 3))
        X = pd.DataFrame({'gini': rng.normal(size=n),
                          'internet': rng.normal(size=n)})
        y = pd.Series(X['gini'] * 0.5 + rng.normal(0, 0.3, n))
        return X, y, pd.Series(entities)

    def _model(self, paradigm):
        import importlib
        module = importlib.import_module(
            f'architectures_ml.{paradigm}.models.hierarchical_model')
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and hasattr(obj, 'simple_hierarchical_model'):
                return obj
        pytest.skip(f'no hierarchical class found in {paradigm}')

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_simple_model_runs(self, paradigm):
        cls = self._model(paradigm)
        instance = cls.__new__(cls)  # bypass __init__, which loads data
        X, y, entities = self._panel()
        result = cls.simple_hierarchical_model(
            instance, X, y, X, y, entities, entities)
        assert 'r2' in result
        hm = SCIENTIFIC_CONFIG['hierarchical_model']
        # The payload describes the grid it searched, derived from the config.
        described = result['regularization_details']['alpha_selection']
        assert str(hm['ridge_alpha_count']) in described
        assert str(hm['ridge_alpha_log10_stop']) in described

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_forest_model_runs(self, paradigm):
        cls = self._model(paradigm)
        instance = cls.__new__(cls)
        X, y, entities = self._panel()
        result = cls.random_forest_hierarchical(
            instance, X, y, X, y, entities, entities)
        assert result['model_name'] == 'random_forest_hierarchical'

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_empty_residual_branch_still_returns(self, paradigm):
        """The branch that skips the ridge fit must not raise on _hm."""
        cls = self._model(paradigm)
        instance = cls.__new__(cls)
        # A single entity with one observation leaves no residual rows.
        X = pd.DataFrame({'gini': [0.1], 'internet': [0.2]})
        y = pd.Series([1.0])
        entities = pd.Series(['BRA'])
        result = cls.simple_hierarchical_model(
            instance, X, y, X, y, entities, entities)
        assert 'alpha_selection' in result['regularization_details']
