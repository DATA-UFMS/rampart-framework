#!/usr/bin/env python3
"""The statistical modules must read their parameters from the configuration.

A module carrying its own default lets the executed protocol drift from the
reported one, without anything failing. That drift happened: the paper reports
10,000 bootstrap resamples while every code path defaulted to 3,000.
"""

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.scientific_config import SCIENTIFIC_CONFIG

_MODULES = sorted((_ROOT / 'src' / 'statistical_validation').glob('*.py'))

# Parameters whose value defines the protocol. Matched by name rather than by
# value, since a protocol value can coincide with an unrelated constant -- 0.05
# is both a SESOI and the conventional significance level.
_OWNED_PARAMETERS = ('n_boot', 'bootstrap_iters', 'iters', 'n_resamples',
                     'sesoi', 'sesoi_r2', 'sesoi_mase', 'sesoi_wape', 'seed')


def test_modules_were_found():
    assert _MODULES, 'no statistical modules discovered'


@pytest.mark.parametrize('path', _MODULES, ids=lambda p: p.name)
def test_no_module_defines_its_own_bootstrap_default(path):
    """A fallback default is a second source of truth for the resample count."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if 'DEFAULT_BOOTSTRAP_ITERS' not in names:
            continue
        # Must be derived from the configuration, not a literal.
        assert not isinstance(node.value, ast.Constant), (
            f"{path.name}: DEFAULT_BOOTSTRAP_ITERS is a literal "
            f"({getattr(node.value, 'value', '?')}), so this module can run "
            f"with a resample count other than the configured one"
        )


@pytest.mark.parametrize('path', _MODULES, ids=lambda p: p.name)
def test_protocol_parameters_do_not_default_to_literals(path):
    """A protocol parameter defaulting to a literal is a second source of truth."""
    tree = ast.parse(path.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        # Align each default with the argument it belongs to.
        positional = args.posonlyargs + args.args
        pairs = list(zip(positional[len(positional) - len(args.defaults):],
                         args.defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)
                  if d is not None]

        body = ast.unparse(node)
        for arg, default in pairs:
            if arg.arg not in _OWNED_PARAMETERS:
                continue
            if not isinstance(default, ast.Constant):
                continue
            # None is the sentinel for "read the configuration", and it is the
            # pattern the rest of the repository uses -- `x or default` treats
            # a legitimate zero as absent, so the explicit None check replaced
            # it in warmup_runs and repetitions. Allowing it here would hollow
            # the guard out unless the function demonstrably does read the
            # configuration, so that is what is required.
            if default.value is None:
                assert 'SCIENTIFIC_CONFIG' in body or 'BENCHMARK_CONFIG' in body, (
                    f"{path.name}:{node.lineno} {node.name}() takes "
                    f"{arg.arg}=None as the sentinel for the configured value "
                    f"and never reads a configuration"
                )
                continue
            raise AssertionError(
                f"{path.name}:{node.lineno} {node.name}() defaults "
                f"{arg.arg}={default.value!r} to a literal, so it can run "
                f"with a value other than the configured one"
            )


def test_bootstrap_count_meets_the_quantile_interval_requirement():
    """Every interval here reads quantiles of the bootstrap distribution.

    Hesterberg (2015, The American Statistician 69(4):371-386) derives
    r >= 15000 for Monte Carlo variability in percentile interval endpoints to
    stay within 10% of the exhaustive value.
    """
    assert SCIENTIFIC_CONFIG['bootstrap_iters'] >= 15000


def test_every_module_agrees_on_the_count():
    from statistical_validation import (effect_analysis,
                                        equivalence_estimation,
                                        significance_tests)
    expected = SCIENTIFIC_CONFIG['bootstrap_iters']
    assert significance_tests.DEFAULT_BOOTSTRAP_ITERS == expected
    assert equivalence_estimation.DEFAULT_BOOTSTRAP_ITERS == expected
    assert effect_analysis.DEFAULT_BOOTSTRAP_ITERS == expected


class TestEveryDeclaredParameterIsRead:
    """A declared parameter nothing reads is a decoy.

    Seven of them sat in the config. Four described cross-paradigm equivalence
    as agreement within a tolerance -- 85% feature overlap, MAE 0.001 on
    correlations -- while the framework verifies bitwise identity of the
    predicted vectors. They reached the published config snapshot, where a
    reader would reasonably take them for the operative criterion.

    A worse case: `collinearity_threshold` was declared here while each
    paradigm's filter kept its own default of the same value. Changing the
    config did nothing, and the agreement was a coincidence of the two numbers
    being equal.
    """

    #: Recorded for provenance rather than dispatched on. The transform is
    #: written out in each paradigm's own idiom, so there is nothing to switch;
    #: the declaration is checked against the three implementations in
    #: test_unit_core.
    RECORDED_ONLY = {'feature_transform'}

    @staticmethod
    def _declared():
        import re
        source = (_ROOT / 'src' / 'core' / 'scientific_config.py').read_text()
        return sorted(set(re.findall(r"^\s{4}'([a-z0-9_]+)':", source, re.M)))

    @staticmethod
    def _production_sources():
        roots = [_ROOT / 'src', _ROOT / 'scripts']
        files = [path for root in roots for path in root.rglob('*.py')
                 if path.name != 'scientific_config.py']
        files.append(_ROOT / 'pipeline.py')
        return '\n'.join(path.read_text() for path in files if path.exists())

    def test_no_declared_parameter_goes_unread(self):
        haystack = self._production_sources()
        unread = [name for name in self._declared()
                  if name not in self.RECORDED_ONLY and name not in haystack]
        assert not unread, (
            f'declared and never read by production: {unread}. Either wire '
            f'them or drop them: a value in the published snapshot that no '
            f'code consults describes a framework that does not exist.'
        )

    def test_the_collinearity_threshold_reaches_the_filter(self):
        source = (_ROOT / 'src' / 'core' / 'base_architecture.py').read_text()
        call = source[source.index('apply_collinearity_filter(\n'):]
        call = call[:call.index(')') + 1]
        assert 'collinearity_threshold' in call, (
            'the filter is called without the declared threshold, so each '
            'paradigm falls back to its own default'
        )

    def test_the_recorded_exemption_stays_small(self):
        """An exemption list is how the rule above gets hollowed out."""
        assert len(self.RECORDED_ONLY) <= 1
        for name in self.RECORDED_ONLY:
            assert name in self._declared()

    def test_no_weak_equivalence_tolerance_returns(self):
        """The claim is bitwise identity; a tolerance would contradict it."""
        source = (_ROOT / 'src' / 'core' / 'scientific_config.py').read_text()
        import ast as ast_module
        tree = ast_module.parse(source)
        for node in ast_module.walk(tree):
            if isinstance(node, ast_module.Constant) and isinstance(
                    node.value, str) and node.value in (
                    'target_stats_max_diff', 'features_overlap_min_pct',
                    'correlations_max_mae', 'fold_sizes_max_diff_pct'):
                raise AssertionError(
                    f'line {node.lineno}: {node.value!r} states equivalence as '
                    f'a tolerance, contradicting the bitwise claim'
                )
