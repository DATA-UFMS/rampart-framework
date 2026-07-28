#!/usr/bin/env python3
"""An experimental arm gets its own outputs root, and evaluation rows stay out
of training.

Two pieces of plumbing the factorial needs, and neither existed. Both were found
by asking what the experiment would do rather than by reading the code:

  - Every arm wrote to `outputs/<dataset>`. Running the clean arm and then an
    injected one left one set of artifacts, not two, and nothing to compare --
    which is precisely what the per-dataset split already exists to prevent one
    level up.

  - Pasting rows from the test window into the training frame passed every
    check. Verified before the gate below existed: `canonical_fold` accepts it,
    because its duplicate test is within a split; the temporal gate accepts it,
    because it reads the fold configuration rather than the data. That is
    memorisation leakage, the class where Roth measures severity scaling with
    model capacity, and this framework did not see it.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.validation import AntiLeakageViolation, assert_splits_disjoint  # noqa


class TestTheArmSeparatesArtifacts:

    def _root(self, monkeypatch, arm=None, dataset='worldbank'):
        import importlib
        import core.config as config
        monkeypatch.setenv('DATASET_NAME', dataset)
        if arm is None:
            monkeypatch.delenv('RAMPART_ARM', raising=False)
        else:
            monkeypatch.setenv('RAMPART_ARM', arm)
        importlib.reload(config)
        return config.get_outputs_root()

    def test_without_an_arm_the_path_is_unchanged(self, monkeypatch):
        """Production is not asked to know that experiments exist."""
        assert self._root(monkeypatch).endswith(os.path.join('outputs', 'worldbank'))

    def test_each_arm_gets_its_own_root(self, monkeypatch):
        clean = self._root(monkeypatch, arm='c0_clean')
        leaked = self._root(monkeypatch, arm='c3_dose030')
        assert clean != leaked
        assert clean.endswith(os.path.join('worldbank', 'c0_clean'))

    def test_the_arm_sits_under_the_dataset_not_beside_it(self, monkeypatch):
        """Otherwise two datasets under one arm name would collide."""
        wb = self._root(monkeypatch, arm='probe', dataset='worldbank')
        inep = self._root(monkeypatch, arm='probe', dataset='inep_censo')
        assert wb != inep

    @pytest.mark.parametrize('bad', ['../escape', 'a/b', '.hidden'])
    def test_an_arm_that_escapes_the_directory_is_refused(self, monkeypatch,
                                                          bad):
        """A separator would put artifacts where no reader of this dataset looks."""
        with pytest.raises(ValueError, match='single directory name'):
            self._root(monkeypatch, arm=bad)


class TestEvaluationRowsStayOutOfTraining:

    TRAIN = (['ARG', 'ARG', 'BRA', 'BRA'], [2000, 2001, 2000, 2001])
    TEST = (['ARG', 'BRA'], [2014, 2014])

    def test_a_clean_fold_passes(self):
        assert_splits_disjoint({'train': self.TRAIN, 'test': self.TEST},
                               paradigm='probe') is None

    def test_a_test_row_in_training_halts(self):
        """The injection the framework used to accept in silence."""
        contaminated = (list(self.TRAIN[0]) + ['ARG'],
                        list(self.TRAIN[1]) + [2014])
        with pytest.raises(AntiLeakageViolation, match='evaluation independence'):
            assert_splits_disjoint({'train': contaminated, 'test': self.TEST},
                                   paradigm='probe')

    def test_a_validation_row_in_training_halts_too(self):
        """Validation is an evaluation split; tuning on fitted rows is the same
        defect wearing another name."""
        val = (['ARG'], [2010])
        contaminated = (list(self.TRAIN[0]) + ['ARG'],
                        list(self.TRAIN[1]) + [2010])
        with pytest.raises(AntiLeakageViolation, match="'val'"):
            assert_splits_disjoint({'train': contaminated, 'val': val},
                                   paradigm='probe')

    def test_the_same_entity_in_both_splits_is_not_a_violation(self):
        """The panel's whole structure. Keying on entity alone would fail every
        legitimate fold, which is why the key is (entity, year)."""
        assert_splits_disjoint({'train': self.TRAIN, 'test': self.TEST},
                               paradigm='probe')
        assert set(self.TRAIN[0]) == set(self.TEST[0]), 'fixture lost its point'

    def test_the_message_names_the_paradigm_and_the_rows(self):
        contaminated = (list(self.TRAIN[0]) + ['ARG'],
                        list(self.TRAIN[1]) + [2014])
        with pytest.raises(AntiLeakageViolation) as caught:
            assert_splits_disjoint({'train': contaminated, 'test': self.TEST},
                                   paradigm='task_graph')
        message = str(caught.value)
        assert 'task_graph' in message
        assert "('ARG', 2014)" in message, 'the offending row is not named'

    def test_a_missing_training_split_is_an_error_not_a_pass(self):
        """Silently passing when handed the wrong shape is how a gate stops
        gating."""
        with pytest.raises(ValueError, match='training split'):
            assert_splits_disjoint({'test': self.TEST}, paradigm='probe')


class TestEveryParadigmCallsIt:
    """A gate one paradigm skips is a gate the comparison does not have."""

    @pytest.mark.parametrize('paradigm',
                             ['sql_engine', 'task_graph', 'dataframe_lib'])
    def test_the_paradigm_checks_disjointness(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        assert 'assert_splits_disjoint(' in source, paradigm
        assert 'return_years=True' in source, (
            f'{paradigm} cannot key rows without the years')

    @pytest.mark.parametrize('paradigm',
                             ['sql_engine', 'task_graph', 'dataframe_lib'])
    def test_all_three_splits_are_checked(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        # From the call to the end of its statement, found by matching the
        # parentheses rather than by looking for a closing string -- the call
        # gained an argument once already, and a literal terminator would have
        # silently started slicing the wrong region.
        start = source.index('assert_splits_disjoint(')
        depth, end = 0, start
        for index in range(start, len(source)):
            if source[index] == '(':
                depth += 1
            elif source[index] == ')':
                depth -= 1
                if depth == 0:
                    end = index
                    break
        block = source[start:end]
        for split in ("'train'", "'val'", "'test'"):
            assert split in block, f'{paradigm} does not check {split}'


class TestTheInjectionSpec:
    """Declared before the run, refused when malformed, silent when absent."""

    def test_no_variable_means_no_injection(self, monkeypatch):
        from core.injection import active
        monkeypatch.delenv('RAMPART_INJECTION', raising=False)
        assert active() is None

    def test_unparseable_stops_the_run(self, monkeypatch):
        """Treating a broken spec as absent would label a clean arm experimental."""
        from core.injection import active
        monkeypatch.setenv('RAMPART_INJECTION', '{not json')
        with pytest.raises(ValueError, match='not valid JSON'):
            active()

    @pytest.mark.parametrize('payload,message', [
        ({'class': 'C9', 'dose': 0.1}, 'unknown injection class'),
        ({'class': 'C3', 'dose': 0.0}, 'dose must be in'),
        ({'class': 'C3', 'dose': 1.5}, 'dose must be in'),
        ({'class': 'C3', 'dose': 0.1, 'waived_gates': ['L9.9']}, 'unknown gate'),
    ])
    def test_a_malformed_spec_is_refused(self, monkeypatch, payload, message):
        import json as _json
        from core.injection import active
        monkeypatch.setenv('RAMPART_INJECTION', _json.dumps(payload))
        with pytest.raises(ValueError, match=message):
            active()

    def test_a_typo_in_a_gate_name_is_not_a_silent_non_waiver(self, monkeypatch):
        """Otherwise the arm aborts hours in, over the violation it was built
        to commit."""
        import json as _json
        from core.injection import active
        monkeypatch.setenv('RAMPART_INJECTION',
                           _json.dumps({'class': 'C3', 'dose': 0.1,
                                        'waived_gates': ['L11']}))
        with pytest.raises(ValueError, match='unknown gate'):
            active()

    def test_one_switch_per_spec(self):
        """Two at once and the inflation is attributable to neither."""
        import inspect
        from core.injection import InjectionSpec
        klass = inspect.signature(InjectionSpec).parameters['klass']
        assert klass.annotation in (str, 'str'), (
            'the class became a collection; a spec must carry exactly one')


class TestTheWaiverIsVisible:
    """An arm that waived a gate must not look like a clean run."""

    TRAIN = (['A', 'A', 'B'], [2000, 2001, 2000])
    TEST = (['A', 'B'], [2014, 2014])
    DIRTY = (['A', 'A', 'B', 'A'], [2000, 2001, 2000, 2014])

    def test_a_clean_run_records_no_waiver(self):
        record = assert_splits_disjoint({'train': self.TRAIN, 'test': self.TEST},
                                        paradigm='probe')
        assert record['waived'] == []

    def test_the_waiver_needs_the_spec_at_the_call_site(self):
        """Not read from the environment: a gate that softens itself behind the
        caller is a gate nobody can audit by reading the call."""
        import os
        os.environ['RAMPART_INJECTION'] = (
            '{"class": "C3", "dose": 0.5, "waived_gates": ["L1.1"]}')
        try:
            with pytest.raises(AntiLeakageViolation):
                assert_splits_disjoint({'train': self.DIRTY, 'test': self.TEST},
                                       paradigm='probe')
        finally:
            del os.environ['RAMPART_INJECTION']

    def test_a_waived_overlap_is_recorded_with_what_declared_it(self):
        from core.injection import InjectionSpec

        spec = InjectionSpec(klass='C3', dose=0.5, waived=('L1.1',))
        record = assert_splits_disjoint({'train': self.DIRTY, 'test': self.TEST},
                                        paradigm='probe', injection=spec)
        assert len(record['waived']) == 1
        waived = record['waived'][0]
        assert waived['overlapping_rows'] == 1
        assert waived['declared_by']['class'] == 'C3'
        assert waived['declared_by']['dose'] == 0.5

    def test_a_spec_that_does_not_name_the_gate_does_not_waive_it(self):
        from core.injection import InjectionSpec

        spec = InjectionSpec(klass='C1', dose=0.5)
        with pytest.raises(AntiLeakageViolation):
            assert_splits_disjoint({'train': self.DIRTY, 'test': self.TEST},
                                   paradigm='probe', injection=spec)


class TestTheInjectionIsReproducible:

    @staticmethod
    def _splits(rows=20):
        X = pd.DataFrame({'f': [float(i) for i in range(rows)]})
        y = pd.Series([float(i) for i in range(rows)])
        entities = pd.Series(['A' if i % 2 else 'B' for i in range(rows)])
        years = pd.Series([2000 + i for i in range(rows)])
        return X, y, entities, years

    def _run(self, seed, fold_id, dose=0.30):
        from core.injection import InjectionSpec, duplicate_evaluation_rows

        Xtr, ytr, etr, yrtr = self._splits(20)
        Xte, yte, ete, yrte = self._splits(10)
        spec = InjectionSpec(klass='C3', dose=dose, waived=('L1.1',), seed=seed)
        _, record = duplicate_evaluation_rows(Xtr, ytr, etr, yrtr,
                                              Xte, yte, ete, yrte,
                                              spec=spec, fold_id=fold_id)
        return record

    def test_the_same_seed_and_fold_move_the_same_rows(self):
        assert self._run(42, 0)['keys_moved'] == self._run(42, 0)['keys_moved']

    def test_different_folds_move_different_rows(self):
        """One generator for the whole run would make a fold's sample depend on
        how many folds preceded it, and a single-fold rerun would not replay."""
        moved = [tuple(map(tuple, self._run(42, f)['keys_moved']))
                 for f in range(6)]
        assert len(set(moved)) > 1, 'every fold drew the same rows'

    def test_the_dose_sets_how_many_rows_move(self):
        assert self._run(42, 0, dose=0.30)['rows_moved'] == 3
        assert self._run(42, 0, dose=0.10)['rows_moved'] == 1

    def test_the_moved_rows_are_named_not_just_counted(self):
        """A count says an arm was contaminated; the keys say how, and let the
        contamination be reconstructed from the artifact alone."""
        record = self._run(42, 0)
        assert len(record['keys_moved']) == record['rows_moved']
        assert all(len(k) == 2 for k in record['keys_moved'])


class TestEveryParadigmAppliesBothClasses:
    """A class one paradigm skips is an arm whose label is wrong.

    Structural, because running a fold needs a live engine, and the level that
    matters here is whether the wiring exists at all. The arithmetic of each
    class is tested behaviourally in core; what these assert is that the three
    paradigms reach it. Caught by mutation: disabling C1 in one paradigm left
    the whole suite green, so a run labelled C1 would have produced the clean
    result under an experimental name -- the worst failure this design can have,
    because nothing downstream would contradict it.
    """

    PARADIGMS = ('sql_engine', 'task_graph', 'dataframe_lib')

    @staticmethod
    def _source(paradigm):
        return (_SRC / 'architectures_ml' / paradigm / 'models'
                / 'hierarchical_model.py').read_text()

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_it_reads_the_spec_once(self, paradigm):
        source = self._source(paradigm)
        assert 'injection_active()' in source, paradigm
        assert source.count('injection_active()') == 1, (
            f'{paradigm} reads the spec more than once; two reads can disagree '
            f'within a fold')

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_it_applies_c3(self, paradigm):
        source = self._source(paradigm)
        assert "klass == 'C3'" in source, paradigm
        assert 'duplicate_evaluation_rows(' in source, paradigm

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_it_applies_c1_to_both_statistics(self, paradigm):
        """Both, because on these panels only one of them does any work.

        The World Bank collection has zero missing cells, so contaminating the
        imputer alone injects nothing and C1 would measure zero for a reason
        that has nothing to do with the hypothesis. The scaler touches every
        row.
        """
        source = self._source(paradigm)
        assert "klass == 'C1'" in source, paradigm
        assert 'contaminated_fit_frame(' in source, paradigm
        assert source.count('fit_on=_fit_on') == 2, (
            f'{paradigm} passes the contaminated frame to '
            f'{source.count("fit_on=_fit_on")} statistic(s); C1 has to reach '
            f'both the imputation and the scaler')

    @pytest.mark.parametrize('paradigm', PARADIGMS)
    def test_the_contamination_is_recorded(self, paradigm):
        """An arm has to be readable from its artifacts, not from its label."""
        source = self._source(paradigm)
        assert '_injection_records.append(' in source, paradigm
        assert source.count('_injection_records.append(') == 2, (
            f'{paradigm} records {source.count("_injection_records.append(")} '
            f'of the two classes')
