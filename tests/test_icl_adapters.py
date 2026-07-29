#!/usr/bin/env python3
"""In-context adapters: what they pin, what they cap, and what they record.

Most of this file runs without either package installed, because the properties
that matter most are about absence: an optional dependency that is not there has
to fail at the point of use with a message naming what to install, and an import
of the framework must not drag a foundation model in behind it.

The two properties that need the package are marked to skip. They are the ones
a reviewer would ask about -- which weights, and how much context -- so they are
tested rather than asserted in prose.
"""

import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.models import icl  # noqa
from core.scientific_config import SCIENTIFIC_CONFIG  # noqa

_HAS_TABPFN = find_spec('tabpfn') is not None
_HAS_TABICL = find_spec('tabicl') is not None


def frame(rows=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(rows, 3)), columns=['a', 'b', 'c'])
    y = pd.Series(X['a'] * 2 + rng.normal(scale=.2, size=rows))
    entity = pd.Series([f'E{i % 4}' for i in range(rows)])
    years = pd.Series([2000 + i // 4 for i in range(rows)])
    return X, y, entity, years


class TestAbsenceIsHandledAtTheCallSite:

    def test_both_families_are_declared(self):
        assert set(icl.MODELS) == {'icl_tabpfn', 'icl_tabicl'}

    def test_a_missing_package_names_what_to_install(self, monkeypatch):
        """The error a user without the extra actually sees."""
        import builtins
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name.split('.')[0] in ('tabpfn', 'tabicl'):
                raise ImportError(f'No module named {name!r}')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', refuse)
        for family in icl.FAMILIES:
            with pytest.raises(icl.ICLUnavailable, match=r"rampart\[icl\]"):
                family.make()

    def test_unavailability_is_an_import_error(self):
        """So a caller that already handles ImportError keeps working."""
        assert issubclass(icl.ICLUnavailable, ImportError)


@pytest.mark.skipif(not _HAS_TABPFN, reason='tabpfn is an optional dependency')
class TestTabPFNIsPinned:

    def test_the_weights_are_the_ungated_v2(self):
        """v2.5, v2.6 and v3 need a browser license and a personal token.

        An artifact built on those does not reproduce for a reviewer who has
        neither, so the pin is part of the result and is checked, not assumed.
        """
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import resolve_model_version
        icl._tabpfn_regressor()
        assert resolve_model_version(None) == ModelVersion.V2

    def test_the_pin_survives_a_hostile_environment(self, monkeypatch):
        """Setting the variable is not enough: the settings object is built at
        import, so the pin has to be applied to the object itself."""
        monkeypatch.setenv('TABPFN_MODEL_VERSION', 'v3')
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import resolve_model_version
        icl._tabpfn_regressor()
        assert resolve_model_version(None) == ModelVersion.V2

    def test_the_provenance_reaches_the_result(self):
        X, y, entity, years = frame(rows=50)
        result = icl.fit_in_context(
            X, y, X, y, entity, entity,
            model=icl.MODELS['icl_tabpfn'], architecture='dataframe_lib',
            years_train=years)
        assert result['provenance']['package'] == 'tabpfn'
        assert result['provenance']['package_version'] != 'absent'
        assert result['context']['capped'] is False
        assert len(result['predictions']) == len(y)

    def test_predictions_do_not_depend_on_the_rest_of_the_batch(self):
        """The refuted premise, checked on the installed version rather than
        taken from a source reading of a different one.

        If the model were transductive, the evaluation rows would inform each
        other and contamination would have a route the classical comparators do
        not have. It is not, and the argument this study makes had to change
        because of it. The residual below is batch composition noise, and it is
        also where the tolerance for the ICL prediction stage comes from.
        """
        X, y, entity, years = frame(rows=80)
        query = X.iloc[:12].reset_index(drop=True)
        query_entities = entity.iloc[:12].reset_index(drop=True)
        model = icl._tabpfn_regressor()
        from core.models.ladder import entity_effect_frames
        train_augmented, query_augmented, _m, _g = entity_effect_frames(
            X, query, y, entity, query_entities)
        model.fit(train_augmented, y)

        together = np.asarray(model.predict(query_augmented), dtype=float)
        singly = np.array([float(model.predict(query_augmented.iloc[[i]])[0])
                           for i in range(len(query_augmented))])
        tolerance = (SCIENTIFIC_CONFIG['in_context_models']
                     ['determinism_tolerance_relative'] * float(np.std(y)))
        assert np.abs(together - singly).max() < tolerance


@pytest.mark.skipif(not _HAS_TABICL, reason='tabicl is an optional dependency')
class TestTabICLRuns:

    def test_the_second_family_returns_the_shared_contract(self):
        X, y, entity, years = frame(rows=50)
        result = icl.fit_in_context(
            X, y, X, y, entity, entity,
            model=icl.MODELS['icl_tabicl'], architecture='task_graph',
            years_train=years)
        assert result['model_name'] == 'icl_tabicl'
        assert result['architecture'] == 'task_graph'
        assert result['provenance']['package'] == 'tabicl'
        assert np.isfinite(result['r2'])


@pytest.mark.skipif(not _HAS_TABPFN, reason='tabpfn is an optional dependency')
class TestTheEnsembleSwitch:
    """One switch for the robustness reading, and the receipt says which is which.

    Averaging is how bagging drops a forest's absorption from 1.00 to 0.39, so
    "does the ensemble shrink it?" was a fair question to ask three times. It does
    not -- measured at a ratio of 1.010 -- because TabPFN averages over
    preprocessing permutations of the same context rather than over resamples, so
    the contaminated row sits in every member.
    """

    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv('RAMPART_ICL_ROBUSTNESS', raising=False)
        estimator = icl._tabpfn_regressor()
        assert estimator.n_estimators == (
            SCIENTIFIC_CONFIG['in_context_models']['tabpfn_n_estimators'])

    def test_the_switch_selects_the_configured_robustness_size(self, monkeypatch):
        monkeypatch.setenv('RAMPART_ICL_ROBUSTNESS', '1')
        estimator = icl._tabpfn_regressor()
        assert estimator.n_estimators == (
            SCIENTIFIC_CONFIG['in_context_models']
            ['tabpfn_n_estimators_robustness'])

    def test_the_receipt_records_which_reading_it_is(self, monkeypatch):
        X, y, entity, years = frame(rows=40)
        monkeypatch.setenv('RAMPART_ICL_ROBUSTNESS', '1')
        result = icl.fit_in_context(
            X, y, X, y, entity, entity, model=icl.MODELS['icl_tabpfn'],
            architecture='dataframe_lib', years_train=years)
        assert result['provenance']['ensemble_robustness'] is True




class TestTheCapCannotBeBypassed:
    """The cap lives in the estimator, and that is the whole design.

    It used to live in `fit_in_context`, so every probe that built a model and
    called `.fit()` itself walked past it. Then it lived in a helper the probes
    had to remember to call, and the absorption routine -- which appends rows to
    a frame and refits -- pushed a frame capped at exactly the limit twelve rows
    over it. Three cloud failures, one shape: a policy that depends on every
    caller remembering is a policy that will be forgotten.

    Now the factory returns a wrapper and the wrapper truncates whatever it is
    handed. There is nothing for a caller to remember, and these tests exist to
    keep it that way.
    """

    class Spy:
        """Records the largest frame the wrapped model was ever asked to fit."""

        def __init__(self):
            self.largest = 0

        def fit(self, X, y):
            self.largest = max(self.largest, len(X))
            return self

        def predict(self, X):
            return np.zeros(len(X))

    def _frames(self, rows=800):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(rows, 3)), columns=['a', 'b', 'c'])
        y = pd.Series(rng.normal(size=rows))
        return X, y

    def test_it_truncates_to_the_cap(self):
        X, y = self._frames()
        spy = self.Spy()
        icl.ContextCapped(spy, cap=200).fit(X, y)
        assert spy.largest == 200

    def test_a_frame_under_the_cap_is_untouched(self):
        X, y = self._frames(rows=50)
        spy = self.Spy()
        wrapped = icl.ContextCapped(spy, cap=200)
        wrapped.fit(X, y)
        assert spy.largest == 50 and wrapped.context['capped'] is False

    def test_appending_to_an_already_capped_frame_still_lands_on_the_cap(self):
        """The exact failure: capped to 10,000, absorption appended twelve, and
        the model refused 10,012. Under the wrapper there is nothing to reserve."""
        X, y = self._frames()
        spy = self.Spy()
        wrapped = icl.ContextCapped(spy, cap=200)
        wrapped.fit(X, y)
        widened_X = pd.concat([X.head(200), X.head(12)], ignore_index=True)
        widened_y = pd.concat([y.head(200), y.head(12)], ignore_index=True)
        icl.ContextCapped(spy, cap=200).fit(widened_X, widened_y)
        assert spy.largest == 200, (
            f'the model was handed {spy.largest} rows against a cap of 200')

    def test_recency_keeps_the_tail(self):
        """Which is the recency rule only because the harness sorts by year, and
        because rows an arm appends are evaluation-window rows -- newer than
        anything in training."""
        X = pd.DataFrame({'a': range(100)})
        y = pd.Series(range(100))
        spy = self.Spy()

        class Capture(self.Spy):
            def fit(self, X, y):
                self.seen = X['a'].tolist()
                return super().fit(X, y)

        capture = Capture()
        icl.ContextCapped(capture, cap=10).fit(X, y)
        assert capture.seen == list(range(90, 100))

    def test_the_sensitivity_rule_takes_from_across_the_window(self):
        X = pd.DataFrame({'a': range(100)})
        y = pd.Series(range(100))

        class Capture(self.Spy):
            def fit(self, X, y):
                self.seen = X['a'].tolist()
                return super().fit(X, y)

        capture = Capture()
        icl.ContextCapped(capture, cap=10, rule='random').fit(X, y)
        assert min(capture.seen) < 50, 'a random sample that only took the tail'
        assert len(capture.seen) == 10

    def test_the_receipt_says_which_rule_and_how_much_was_dropped(self):
        X, y = self._frames()
        wrapped = icl.ContextCapped(self.Spy(), cap=200)
        wrapped.fit(X, y)
        assert wrapped.context['rows_dropped'] == 600
        assert wrapped.context['offered_rows'] == 800
        assert 'most recent' in wrapped.context['rule']

    def test_nothing_outside_the_adapter_builds_a_bare_estimator(self):
        """The only route to an uncapped model is bypassing the factory.

        Read from the syntax tree so a comment mentioning the class name is not
        an offence, and so the next probe cannot quietly construct one.
        """
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        offending = []
        for path in list((root / 'scripts').rglob('*.py')) + \
                    list((root / 'src').rglob('*.py')):
            if path.name == 'icl.py':
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in ('TabPFNRegressor',
                                             'TabICLRegressor')):
                    offending.append(f'{path.relative_to(root)}:{node.lineno}')
        assert not offending, (
            f'an in-context estimator is built outside the adapter, which is '
            f'the one way to get one without a cap: {offending}')

    def test_below_the_cap_the_wrapper_is_the_identity(self):
        """Bit for bit, and this is what makes the refactor safe to land.

        On World Bank four hundred training rows never approach ten thousand, so
        the wrapper is a pass-through there and cannot have moved any published
        number. Worth an assertion rather than a forty-minute rerun on CPU to
        check something true by construction -- which is what a first attempt
        launched.
        """
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(400, 5)))
        y = pd.Series(rng.normal(size=400))
        query = pd.DataFrame(rng.normal(size=(64, 5)))

        bare = Ridge(alpha=1.0).fit(X, y).predict(query)
        wrapped = icl.ContextCapped(Ridge(alpha=1.0), cap=10_000)
        assert np.array_equal(bare, wrapped.fit(X, y).predict(query))
        assert wrapped.context['capped'] is False
