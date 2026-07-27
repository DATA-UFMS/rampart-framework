#!/usr/bin/env python3
"""Which window the model evaluated on the test set is fitted on.

A recorded decision, and not a default: the model evaluated on the test set is
fitted on the training window alone, and validation serves exclusively to select
hyperparameters.

The alternative -- refitting on train+validation with the chosen hyperparameters
-- is standard practice and was verified as compatible with P2: the gap from
val_end to the test is exactly the 2 years required. It would use 25% more years
per entity and move the origin 4 years closer to the test.

It was not adopted because of an asymmetry. What it would buy is statistical
efficiency in a device whose predictive accuracy is not the object of study -- the
paper claims equivalence between paradigms and latency. What it would cost is
margin in the anti-leakage guarantee, which IS the object: the effective
separation between the last fitting datum and the first evaluation datum would
fall from 6 years to the declared minimum of 2. And it would require a second fit
of imputation and scaler inside the three run_fold_analysis, which have distinct
implementations per engine -- the configuration that produces divergence between
paradigms, when bitwise equivalence is the central claim.

These tests exist so that a change to that choice is deliberate.
"""

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.scientific_config import SCIENTIFIC_CONFIG

MODELS = sorted((_SRC / 'architectures_ml').glob('*/models/hierarchical_model.py'))
FINAL_FITS = ('simple_hierarchical_model', 'random_forest_hierarchical')


def _fold_analysis(path):
    tree = ast.parse(path.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name == 'run_fold_analysis')


def _calls_evaluating_test(path):
    """Model calls whose evaluation set is the test window."""
    found = []
    for call in ast.walk(_fold_analysis(path)):
        if not (isinstance(call, ast.Call)
                and getattr(call.func, 'attr', None) in FINAL_FITS):
            continue
        names = [getattr(a, 'id', None) for a in call.args]
        if any(n and 'test' in n for n in names):
            found.append((call, names))
    return found


class TestTheFinalModelFitsOnTrainOnly:

    def test_all_three_models_were_found(self):
        assert len(MODELS) == 3

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_both_models_are_evaluated_on_the_test_window(self, path):
        assert len(_calls_evaluating_test(path)) == len(FINAL_FITS), (
            f'{path.parts[-3]}: expected one final fit per model'
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_fit_arguments_are_the_training_window(self, path):
        """The first two arguments are the X and y of the fit."""
        for call, names in _calls_evaluating_test(path):
            assert names[0] == 'X_train_scaled', (
                f'{path.parts[-3]}:{call.lineno} fits on {names[0]}'
            )
            assert names[1] == 'y_train', (
                f'{path.parts[-3]}:{call.lineno} fits on {names[1]}'
            )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_validation_window_is_not_concatenated_into_the_fit(self, path):
        """A concat of train with validation is the change this guards against."""
        for call, _ in _calls_evaluating_test(path):
            for argument in call.args[:2]:
                assert not isinstance(argument, ast.Call), (
                    f'{path.parts[-3]}:{call.lineno} passes an expression as '
                    f'fitting data, and not the training window'
                )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_validation_window_is_used_for_selection(self, path):
        """Validation must serve some purpose, otherwise it is pure waste."""
        calls = []
        for call in ast.walk(_fold_analysis(path)):
            if isinstance(call, ast.Call) and \
                    getattr(call.func, 'attr', None) in FINAL_FITS:
                names = [getattr(a, 'id', None) for a in call.args]
                if any(n and 'val' in n for n in names):
                    calls.append(call)
        assert len(calls) == len(FINAL_FITS), (
            f'{path.parts[-3]}: validation is not evaluated in the selection'
        )


class TestTheEffectiveSeparationIsRecorded:

    @staticmethod
    def _fold(gap, min_train=8, val_len=2, test_len=2):
        start = SCIENTIFIC_CONFIG['temporal_range_start']
        train_end = start + min_train - 1
        val_start = train_end + gap + 1
        val_end = val_start + val_len - 1
        test_start = val_end + gap + 1
        return {'train_end': train_end, 'val_end': val_end,
                'test_start': test_start,
                'fit_to_test_gap': test_start - train_end - 1}

    def test_the_fold_record_carries_it(self):
        source = (_SRC / 'core' / 'base_architecture.py').read_text()
        assert "'fit_to_test_gap'" in source
        assert "'fit_window': 'train_only'" in source

    def test_the_separation_exceeds_the_declared_minimum(self):
        """This is why the choice buys something: 6 years against 2."""
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        fold = self._fold(gap)
        assert fold['fit_to_test_gap'] > gap, (
            'the effective separation does not exceed the declared minimum, so '
            'the choice not to refit would stop buying margin'
        )

    def test_refitting_would_reduce_it_to_the_minimum(self):
        """The check that supports the decision, not a loose claim."""
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        fold = self._fold(gap)
        would_be = fold['test_start'] - fold['val_end'] - 1
        assert would_be == gap, (
            f'refitting on train+validation would give separation {would_be}, '
            f'and the decision was taken assuming it would fall to the '
            f'minimum {gap}'
        )
        assert would_be < fold['fit_to_test_gap']

    def test_p2_would_still_hold_under_the_alternative(self):
        """The alternative was refused over margin, not for violating P2."""
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        fold = self._fold(gap)
        assert fold['test_start'] - fold['val_end'] - 1 >= gap
