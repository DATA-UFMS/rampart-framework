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


class TestTheContextCap:

    def test_a_short_window_is_left_alone(self):
        X, y, entity, years = frame(rows=60)
        kept_X, kept_y, kept_e, kept_years, record = icl.cap_context(
            X, y, entity, years)
        assert record == {'capped': False, 'context_rows': 60,
                          'cap': SCIENTIFIC_CONFIG['in_context_models']
                                 ['context_cap_rows']}
        assert kept_X is X and kept_y is y and kept_e is entity
        assert kept_years is years

    def test_a_long_window_keeps_the_most_recent_rows(self, monkeypatch):
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_cap_rows', 20)
        X, y, entity, years = frame(rows=60)
        kept_X, kept_y, _e, kept_years, record = icl.cap_context(
            X, y, entity, years)
        assert len(kept_X) == 20 and record['capped'] is True
        assert record['rows_dropped'] == 40
        assert kept_years.min() > years.iloc[:40].max(), (
            'the cap dropped recent rows rather than old ones')

    def test_the_cap_is_recorded_when_it_bites(self, monkeypatch):
        """A truncated context reads as a full one unless the artifact says so."""
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_cap_rows', 20)
        X, y, entity, years = frame(rows=60)
        *_, record = icl.cap_context(X, y, entity, years)
        assert record['training_rows'] == 60
        assert record['context_rows'] == 20
        assert 'rule' in record

    def test_capping_without_years_is_refused(self, monkeypatch):
        """The failure this would have been: canonical_fold sorts by
        (entity, year), so the tail of the frame is the last entities in
        alphabetical order. Keeping it would silently narrow the context to a
        handful of entities and report itself as a recency rule."""
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_cap_rows', 20)
        X, y, entity, _years = frame(rows=60)
        with pytest.raises(ValueError, match='last entities alphabetically'):
            icl.cap_context(X, y, entity, None)

    def test_no_years_is_fine_when_the_cap_does_not_bite(self):
        X, y, entity, _years = frame(rows=10)
        *_, record = icl.cap_context(X, y, entity, None)
        assert record['capped'] is False

    def test_the_cap_does_not_depend_on_the_order_the_engine_produced(
            self, monkeypatch):
        """Three paradigms materialise the frame; the context must not differ."""
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_cap_rows', 20)
        X, y, entity, years = frame(rows=60)
        shuffled = np.random.default_rng(7).permutation(len(X))
        kept_a, *_ = icl.cap_context(X, y, entity, years)
        kept_b, *_ = icl.cap_context(
            X.iloc[shuffled].reset_index(drop=True),
            y.iloc[shuffled].reset_index(drop=True),
            entity.iloc[shuffled].reset_index(drop=True),
            years.iloc[shuffled].reset_index(drop=True))
        assert sorted(kept_a['a'].round(9)) == sorted(kept_b['a'].round(9))


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
