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

        for arg, default in pairs:
            if arg.arg not in _OWNED_PARAMETERS:
                continue
            assert not isinstance(default, ast.Constant), (
                f"{path.name}:{node.lineno} {node.name}() defaults "
                f"{arg.arg}={getattr(default, 'value', '?')!r} to a literal, so "
                f"it can run with a value other than the configured one"
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
