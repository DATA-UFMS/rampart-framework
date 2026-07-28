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
        block = source[source.index('assert_splits_disjoint('):]
        block = block[:block.index('paradigm=PARADIGM)')]
        for split in ("'train'", "'val'", "'test'"):
            assert split in block, f'{paradigm} does not check {split}'
