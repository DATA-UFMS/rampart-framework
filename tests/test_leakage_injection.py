#!/usr/bin/env python3
"""Negative validation of the anti-leakage gate.

Reuses the injectors from scripts/validation/leakage_injection.py so the three
deliberate violations are exercised on every run of the suite:

  S1  insufficient gap between train and validation
  S2  temporal overlap (training years appear in the test window)
  S3  reversed ordering (test starts before training ends)
"""

import importlib.util
from pathlib import Path

import pytest

_INJECTOR = (
    Path(__file__).resolve().parents[1]
    / 'scripts' / 'validation' / 'leakage_injection.py'
)


def _load_injector():
    spec = importlib.util.spec_from_file_location('leakage_injection', _INJECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def injector():
    if not _INJECTOR.exists():
        pytest.skip(f"injector not found: {_INJECTOR}")
    return _load_injector()


@pytest.fixture(scope='module')
def validator(injector):
    return injector.TemporalValidator(min_gap_years=2)


def test_baseline_folds_pass_the_gate(injector, validator):
    """Valid walk-forward folds must not be rejected."""
    folds = injector.generate_valid_folds()
    assert folds, "fold generator returned an empty list"
    validator.enforce_walk_forward(folds)


@pytest.mark.parametrize('scenario', ['s1_zero_gap', 's2_temporal_overlap', 's3_reversed_order'])
def test_injected_violation_is_rejected(injector, validator, scenario):
    inject = getattr(injector, f'inject_{scenario}')
    contaminated = inject(injector.generate_valid_folds())

    with pytest.raises(ValueError) as exc:
        validator.enforce_walk_forward(contaminated)

    assert 'Anti-leakage violation' in str(exc.value), \
        f"error message lacks the expected diagnostic: {exc.value}"


def test_injection_does_not_mutate_the_valid_folds(injector):
    """Injectors must operate on copies of the reference folds."""
    original = injector.generate_valid_folds()
    reference = [dict(f) for f in original]

    for name in ('s1_zero_gap', 's2_temporal_overlap', 's3_reversed_order'):
        getattr(injector, f'inject_{name}')(original)

    assert original == reference, "an injector mutated the valid folds in place"
