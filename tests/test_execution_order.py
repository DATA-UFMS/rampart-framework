#!/usr/bin/env python3
"""Results do not depend on the order the paradigms run in.

The benchmark permutes that order on every repetition, deliberately, so no
paradigm is systematically favoured by cache warmth. That only works if
nothing carries state across a phase boundary.

`setup_reproducibility` seeds the legacy global numpy generator, and the model
modules call it once at import rather than per run -- so across twelve
repetitions in shuffled order the global generator's state evolves freely and
is never reset. Nothing consumes it today: every estimator gets an explicit
random_state and every draw uses a local default_rng. That is the whole reason
the shuffle is safe, and it was resting on nothing but a grep.

Two paradigms also re-seeded the global generator inside their own
setup_environment and the third did not -- a difference in a comparison that
assumes they differ only in how they move data. Redundant, since the base
class already seeds, and asymmetric.
"""

import ast
import contextlib
import hashlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import discover_paradigms

#: Legacy global-generator entry points. Reading any of these makes the result
#: depend on how much randomness was drawn earlier in the process.
GLOBAL_CONSUMERS = {
    'rand', 'randn', 'randint', 'random_sample', 'random_integers',
    'normal', 'uniform', 'choice', 'shuffle', 'permutation', 'standard_normal',
    'binomial', 'poisson', 'exponential', 'beta', 'gamma', 'sample',
}

PRODUCTION = sorted(
    path for path in list(_SRC.rglob('*.py'))
    + list((_ROOT / 'scripts').rglob('*.py'))
    if '__pycache__' not in str(path)
)


def _global_random_calls(tree):
    """`np.random.<consumer>(...)`, excluding seed and default_rng."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in GLOBAL_CONSUMERS:
            continue
        owner = func.value
        if (isinstance(owner, ast.Attribute) and owner.attr == 'random'
                and isinstance(owner.value, ast.Name)
                and owner.value.id in ('np', 'numpy')):
            found.append((node.lineno, f'np.random.{func.attr}'))
    return found


class TestNothingConsumesTheGlobalGenerator:

    @pytest.mark.parametrize('path', PRODUCTION, ids=lambda p: p.name)
    def test_no_module_draws_from_it(self, path):
        found = _global_random_calls(ast.parse(path.read_text()))
        assert not found, (
            f'{path.name} draws from the global generator at {found}; the '
            f'result then depends on how much randomness was drawn before, '
            f'and the benchmark shuffles that order every repetition'
        )

    def test_the_detector_would_fire(self):
        """Otherwise the sweep above could be matching nothing."""
        tree = ast.parse('import numpy as np\nx = np.random.normal(size=3)\n')
        assert _global_random_calls(tree) == [(2, 'np.random.normal')]

    def test_seeding_is_not_flagged(self):
        """Seeding is a write; only reads make order matter."""
        tree = ast.parse('import numpy as np\nnp.random.seed(42)\n')
        assert _global_random_calls(tree) == []

    def test_a_local_generator_is_not_flagged(self):
        tree = ast.parse('import numpy as np\n'
                         'r = np.random.default_rng(42)\nx = r.normal(size=3)\n')
        assert _global_random_calls(tree) == []


class TestEveryEstimatorIsSeededExplicitly:
    """What makes the invariant above hold rather than being a coincidence."""

    SEEDED = {'RandomForestRegressor', 'RandomForestClassifier',
              'GradientBoostingRegressor', 'ExtraTreesRegressor'}

    @pytest.mark.parametrize('path', PRODUCTION, ids=lambda p: p.name)
    def test_stochastic_estimators_receive_a_random_state(self, path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, 'id', '') in self.SEEDED):
                continue
            keywords = {kw.arg for kw in node.keywords}
            assert 'random_state' in keywords, (
                f'{path.name}:{node.lineno} constructs '
                f'{node.func.id} without random_state, so it draws from the '
                f'global generator'
            )

    def test_at_least_one_such_estimator_exists(self):
        """Otherwise the sweep is vacuous."""
        total = sum(
            1 for path in PRODUCTION
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and getattr(node.func, 'id', '') in self.SEEDED
        )
        assert total >= 3, total


class TestNoParadigmSeedsGlobalStateItself:
    """The base class owns it; two of three were repeating it and one was not."""

    @pytest.mark.parametrize('paradigm', sorted(discover_paradigms()))
    def test_setup_does_not_reseed(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'setup.py').read_text()
        tree = ast.parse(source)
        setup = next((node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == 'setup_environment'), None)
        assert setup is not None, paradigm
        body = ast.get_source_segment(source, setup)
        assert 'np.random.seed' not in body, (
            f'{paradigm} re-seeds the global generator in setup_environment; '
            f'the base class already does it for all three'
        )

    def test_the_base_class_still_does_it(self):
        from core.scientific_config import setup_reproducibility
        import inspect
        assert 'np.random.seed' in inspect.getsource(setup_reproducibility)


class TestResultsSurviveAPerturbedGenerator:
    """The behavioural proof, rather than the structural argument."""

    @staticmethod
    def _fit(perturb):
        from core.models.hierarchical import simple_hierarchical_model

        if perturb:
            np.random.seed(999)
            np.random.rand(1000)

        rng = np.random.default_rng(3)
        size = 120
        entities = np.repeat([f'C{index}' for index in range(6)], size // 6)
        X = pd.DataFrame({'a': rng.normal(size=size),
                          'b': rng.normal(size=size)})
        y = pd.Series(0.7 * X['a'] - 0.4 * X['b']
                      + rng.normal(0, 0.3, size))
        with contextlib.redirect_stdout(io.StringIO()):
            warnings.filterwarnings('ignore')
            result = simple_hierarchical_model(
                X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:],
                pd.Series(entities[:90]), pd.Series(entities[90:]),
                architecture='sql_engine')
        vector = np.asarray(result.get('predictions',
                                       result.get('y_pred', [])), dtype=float)
        return hashlib.sha256(vector.tobytes()).hexdigest()

    def test_the_predictions_are_bitwise_identical(self):
        assert self._fit(False) == self._fit(True)

    def test_the_perturbation_actually_moves_the_generator(self):
        """Otherwise the test above compares two identical situations."""
        np.random.seed(0)
        before = np.random.get_state()[1][:4].copy()
        np.random.seed(999)
        np.random.rand(1000)
        assert not np.array_equal(before, np.random.get_state()[1][:4])

    def test_the_fit_produces_something(self):
        digest = self._fit(False)
        assert digest != hashlib.sha256(b'').hexdigest()


class TestTheInjectionPanelIsReproducible:
    """The report is the evidence that the gates fire; it must not drift.

    Its synthetic panel was drawn from the global generator, whose state
    depends on how much randomness the process drew earlier. Reproducibility
    is the entire point of that artifact, and it rested on module import
    order.
    """

    @staticmethod
    def _panel(perturb):
        import importlib
        sys.path.insert(0, str(_ROOT / 'scripts' / 'validation'))
        if perturb:
            np.random.seed(7)
            np.random.rand(5000)
        module = importlib.import_module('leakage_injection')
        importlib.reload(module)

        rng = np.random.default_rng(
            __import__('core.scientific_config', fromlist=['x'])
            .SCIENTIFIC_CONFIG['random_seed'])
        return hashlib.sha256(
            rng.uniform(15, 55, 32).tobytes()).hexdigest()

    def test_the_panel_does_not_depend_on_prior_draws(self):
        assert self._panel(False) == self._panel(True)

    def test_the_script_uses_a_local_generator(self):
        source = (_ROOT / 'scripts' / 'validation'
                  / 'leakage_injection.py').read_text()
        assert 'np.random.default_rng' in source

    def test_no_global_draw_remains(self):
        source = (_ROOT / 'scripts' / 'validation'
                  / 'leakage_injection.py').read_text()
        assert not _global_random_calls(ast.parse(source))

    def test_the_seed_comes_from_the_configuration(self):
        source = (_ROOT / 'scripts' / 'validation'
                  / 'leakage_injection.py').read_text()
        assert "default_rng(SCIENTIFIC_CONFIG['random_seed'])" in source
