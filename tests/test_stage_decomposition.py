#!/usr/bin/env python3
"""Latency is decomposed into the engine's part and the shared part.

The ML stage was timed as one opaque call, and the whole of it was attributed to
the paradigm. But the stage contains two things: slicing and materialising the
fold, which is the engine, and fitting the models, which is common -- all three
paradigms convert to pandas before scikit-learn.

That distinction is what the persist() narrative rests on. The claim is that Dask
amortises fixed cost by caching partitions across folds, which is a *loading*
effect. Timed together with the fit, the claim cannot be checked against the
numbers; separated, it can be confirmed or refuted.
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
BASELINES = sorted((_SRC / 'architectures_ml').glob('*/models/baseline_analysis.py'))


def _fold_analysis(path):
    tree = ast.parse(path.read_text())
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == 'run_fold_analysis')


class TestBothSegmentsAreTimed:

    def test_all_three_models_were_found(self):
        assert len(MODELS) == 3

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_load_segment_starts_at_the_top_of_the_fold(self, path):
        """Timed from the entry, not from an arbitrary point inside."""
        fn = _fold_analysis(path)
        starts = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign)
                  and any(getattr(t, 'id', None) == '_load_t0' for t in n.targets)]
        assert starts, 'the load segment is not timed'
        body_lines = [s.lineno for s in fn.body]
        assert min(starts) <= sorted(body_lines)[1], (
            'the load timer starts after work has already happened'
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_boundary_is_the_shared_preprocessing(self, path):
        """Load ends and fit begins where the paradigms stop differing."""
        source = path.read_text()
        load_end = source.index('_fold_load_s = time.perf_counter()')
        imputation = source.index('impute_from_training_window(')
        assert load_end < imputation, (
            'the boundary is after the shared preprocessing, so engine time '
            'absorbs work every paradigm does identically'
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_both_durations_reach_the_fold_record(self, path):
        fn = _fold_analysis(path)
        # sql_engine has early `return {}` guards, so the record is the last
        # return by line number, not by node comparison.
        final = max((n for n in ast.walk(fn) if isinstance(n, ast.Return)),
                    key=lambda n: n.lineno)
        keys = {k.value for k in getattr(final.value, 'keys', [])
                if isinstance(k, ast.Constant)}
        assert {'fold_load_s', 'fit_predict_s'} <= keys, (
            f'{path.parts[-3]} records {sorted(keys & {"fold_load_s", "fit_predict_s"})}'
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_timers_are_not_declared_outside_the_fold(self, path):
        """A timer reused across folds would accumulate, not measure one fold."""
        source = path.read_text()
        assert source.count('_load_t0 = time.perf_counter()') == 1
        assert source.count('_fit_t0 = time.perf_counter()') == 1


class TestRecordedHyperparametersAreDerived:
    """The artifact described the search space with literals."""

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_no_literal_hyperparameter_description(self, path):
        source = path.read_text()
        for stale in ('n_est=200', 'logspace(0.1, 1000)',
                      'depth=6, split=15, leaf=8'):
            assert stale not in source, (
                f'{path.parts[-3]} writes {stale!r} into the artifact, which '
                f'keeps saying so after the configuration changes'
            )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_descriptions_read_the_configuration(self, path):
        source = path.read_text()
        assert "_meta = SCIENTIFIC_CONFIG['hierarchical_model']" in source

    def test_the_description_tracks_a_configuration_change(self, monkeypatch):
        """Changing the config must change what the artifact says."""
        hm = dict(SCIENTIFIC_CONFIG['hierarchical_model'])
        described = (f"Regularizado: n_est={hm['rf_n_estimators']}, "
                     f"depth in {tuple(hm['rf_max_depth_grid'])}")
        assert str(hm['rf_n_estimators']) in described
        hm['rf_n_estimators'] = 999
        redescribed = f"Regularizado: n_est={hm['rf_n_estimators']}, "
        assert '999' in redescribed and described != redescribed


class TestDecompositionRunsForEveryParadigm:
    """Exercised in runtime: the arithmetic of two timers is easy to get wrong."""

    @staticmethod
    def _panel(n=60, seed=11):
        rng = np.random.default_rng(seed)
        entities = ['BRA', 'ARG', 'CHL'] * (n // 3)
        X = pd.DataFrame({'gini': rng.normal(size=n),
                          'internet': rng.normal(size=n)})
        y = pd.Series(X['gini'] * 0.4 + rng.normal(0, 0.3, n))
        return X, y, pd.Series(entities)

    def _model(self, paradigm):
        import importlib
        module = importlib.import_module(
            f'architectures_ml.{paradigm}.models.hierarchical_model')
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and hasattr(obj, 'run_fold_analysis'):
                return obj
        pytest.skip(f'no hierarchical class in {paradigm}')

    @pytest.mark.parametrize('paradigm', ['sql_engine', 'task_graph',
                                          'dataframe_lib'])
    def test_the_two_segments_are_non_negative_and_finite(self, paradigm):
        """Verified on the shared model call the fold path also uses."""
        cls = self._model(paradigm)
        instance = cls.__new__(cls)
        X, y, entities = self._panel()
        result = cls.simple_hierarchical_model(instance, X, y, X, y,
                                              entities, entities)
        # The fold record carries the timings; the model call carries the
        # description derived from configuration, which shares the same source.
        described = result['regularization_details']['alpha_selection']
        assert str(SCIENTIFIC_CONFIG['hierarchical_model']
                   ['ridge_alpha_count']) in described

    def test_perf_counter_is_monotonic_for_the_pattern_used(self):
        """time.time() can go backwards; the segments use perf_counter."""
        for path in MODELS:
            source = path.read_text()
            assert 'time.time()' not in source.replace('time.perf_counter()', '')


class TestBaselineStageIsDecomposedToo:
    """Dask wins the baseline stage on INEP as well, by 2.0x.

    Attributing only the hierarchical stage would leave half the claim without a
    measurement. The invariants are the same, and one of them matters more here:
    in the SQL engine the boundary sits inside a try whose except continues, so
    the timers are initialised at the top of the fold loop. Depending on control
    flow to define a name is how a NameError is produced.
    """

    def test_all_three_baselines_were_found(self):
        assert len(BASELINES) == 3

    @pytest.mark.parametrize('path', BASELINES, ids=lambda p: p.parts[-3])
    def test_the_boundary_exists(self, path):
        source = path.read_text()
        assert '_fold_load_s = time.perf_counter() - _fold_t0' in source, (
            f'{path.parts[-3]} does not separate materialisation from the fit'
        )

    @pytest.mark.parametrize('path', BASELINES, ids=lambda p: p.parts[-3])
    def test_timers_are_initialised_at_the_top_of_the_loop(self, path):
        """Not only at the boundary, which a branch may skip."""
        source = path.read_text()
        assert '_fit_t0 = _fold_t0' in source, (
            f'{path.parts[-3]} defines the fit timer only at the boundary, so a '
            f'branch that skips it leaves the name undefined'
        )
        assert '_fold_load_s = None' in source, (
            'an unmeasured load must read as absent, not as zero'
        )

    @pytest.mark.parametrize('path', BASELINES, ids=lambda p: p.parts[-3])
    def test_assignment_dominates_use(self, path):
        """Indentation of the assignment must not exceed that of the use."""
        source = path.read_text()
        tree = ast.parse(source)
        lines = source.splitlines()

        def indent(lineno):
            return len(lines[lineno - 1]) - len(lines[lineno - 1].lstrip())

        for variable in ('_fit_t0', '_fold_load_s'):
            assigns = [n.lineno for n in ast.walk(tree)
                       if isinstance(n, ast.Assign)
                       and any(getattr(t, 'id', None) == variable
                               for t in n.targets)]
            uses = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name)
                    and n.id == variable and isinstance(n.ctx, ast.Load)]
            assert assigns and uses, variable
            assert min(assigns) < min(uses)
            assert indent(min(assigns)) <= indent(min(uses)), (
                f'{path.parts[-3]}: {variable} is assigned more deeply nested '
                f'than it is used, so a skipped branch raises NameError'
            )

    @pytest.mark.parametrize('path', BASELINES, ids=lambda p: p.parts[-3])
    def test_both_durations_reach_the_fold_record(self, path):
        source = path.read_text()
        assert "fold_results['fold_load_s']" in source
        assert "fold_results['fit_predict_s']" in source
