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


class TestTheCapDoesNotDegradeTheEntityEffect:
    """The order of capping and augmenting, which was wrong and only bites on the
    large panel.

    The entity effect is the mean of the outcome per entity and the strongest
    column in the design matrix. Capping first computes it from whatever survived:
    on INEP that is under two observations per entity against twelve for the
    classical models. The in-context arm would carry a noisier feature and score
    worse for a reason that is not about in-context learning at all.
    """

    def test_the_effect_is_the_same_whether_or_not_the_cap_bites(self, monkeypatch):
        from core.models.ladder import ENTITY_EFFECT_COLUMN, entity_effect_frames
        rows = 400
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.normal(size=(rows, 3)), columns=['a', 'b', 'c'])
        entity = pd.Series([f'E{i % 20}' for i in range(rows)])
        y = pd.Series([float(e[1:]) for e in entity] + rng.normal(scale=.1,
                                                                 size=rows))
        years = pd.Series([2000 + i // 20 for i in range(rows)])
        X_eval = X.head(20).reset_index(drop=True)
        entity_eval = entity.head(20).reset_index(drop=True)

        # What the whole training window says each entity's mean is.
        _f, uncapped_eval, _m, _g = entity_effect_frames(
            X, X_eval, y, entity, entity_eval)

        # Now make the cap bite hard, and take the same route fit_in_context takes.
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_cap_rows', 40)
        augmented, capped_eval, _m, _g = entity_effect_frames(
            X, X_eval, y, entity, entity_eval)
        kept, _y, _e, _yr, record = icl.cap_context(augmented, y, entity, years)

        assert record['capped'] is True and len(kept) == 40
        assert capped_eval[ENTITY_EFFECT_COLUMN].equals(
            uncapped_eval[ENTITY_EFFECT_COLUMN]), (
            'the evaluation entity effect changed when the cap bit, which means '
            'the statistic was fitted on the capped rows')

    def test_capping_after_augmenting_keeps_the_extra_column(self, monkeypatch):
        from core.models.ladder import ENTITY_EFFECT_COLUMN, entity_effect_frames
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_cap_rows', 15)
        X, y, entity, years = frame(rows=60)
        augmented, _eval, _m, _g = entity_effect_frames(X, X, y, entity, entity)
        kept, *_ = icl.cap_context(augmented, y, entity, years)
        assert ENTITY_EFFECT_COLUMN in kept.columns
        assert len(kept) == 15


class TestTheRegisteredSensitivityArm:
    """Recency is the pre-registered rule; a random sample is the arm that shows
    the conclusion does not depend on it. Both record which they were."""

    def _capped(self, rule, monkeypatch):
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_cap_rows', 20)
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_rule', rule)
        X, y, entity, years = frame(rows=80)
        return icl.cap_context(X, y, entity, years)

    def test_random_takes_rows_from_across_the_window(self, monkeypatch):
        _X, _y, _e, years, record = self._capped('random', monkeypatch)
        assert 'random' in record['rule']
        assert years.min() < 2000 + 80 // 4 // 2, (
            'a random sample that only took recent years is not random')

    def test_recency_takes_only_the_end(self, monkeypatch):
        _X, _y, _e, years, record = self._capped('recent', monkeypatch)
        assert 'most recent' in record['rule']
        assert years.min() >= 2000 + (80 - 20) // 4

    def test_the_two_rules_disagree_about_which_rows(self, monkeypatch):
        recent = self._capped('recent', monkeypatch)[3].tolist()
        random_ = self._capped('random', monkeypatch)[3].tolist()
        assert recent != random_, 'the sensitivity arm is not a different arm'

    def test_random_is_reproducible(self, monkeypatch):
        assert (self._capped('random', monkeypatch)[3].tolist()
                == self._capped('random', monkeypatch)[3].tolist())


    def test_the_environment_overrides_the_configuration(self, monkeypatch):
        """One flag on a job, not a rewritten config.

        The first attempt at the sensitivity arm exec'd the probe inside
        `python3 -c`, which leaves __file__ undefined; the job died in two
        minutes on a GPU node.
        """
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_cap_rows', 20)
        monkeypatch.setitem(SCIENTIFIC_CONFIG['in_context_models'],
                            'context_rule', 'recent')
        monkeypatch.setenv('RAMPART_CONTEXT_RULE', 'random')
        X, y, entity, years = frame(rows=80)
        *_, record = icl.cap_context(X, y, entity, years)
        assert 'random' in record['rule']
