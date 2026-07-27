#!/usr/bin/env python3
"""The design matrix has the rank its feature count implies, or says it does not.

Approval, failure and abandonment rates sum to one hundred by construction, so
any two determine the third. That is not leakage -- Ridge absorbs it through
regularisation and a forest never notices -- but the reported dimensionality
overstates the information, and no coefficient reading survives it.

Pairwise collinearity filtering does not have to catch it. Measured on the
INEP shape with comparable variances, two of the three pairs correlate at 0.13
and -0.69, both under the 0.8 ceiling, so all three survive the filter while
one exact dependency remains among them.

Two measures, because they answer different things. The rank counts how many
independent directions exist; the redundant list names which columns the
others determine. Neither follows from the other: three rates summing to a
constant have one dependency -- rank three counting the intercept, deficiency
one -- and yet each of the three is individually determined by the other two.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import audit_panel

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import discover_paradigms
from core.scientific_config import SCIENTIFIC_CONFIG
from core.validation import audit_feature_set, redundant_features

TOLERANCE = SCIENTIFIC_CONFIG['target_reproduction_tolerance']


def _rendimento_panel(n=300, seed=3):
    """The INEP shape: three rates summing to a hundred."""
    rng = np.random.default_rng(seed)
    approval = rng.uniform(70, 90, n)
    failure = rng.uniform(5, 20, n)
    abandonment = 100 - approval - failure
    return pd.DataFrame({
        'aprov_ef': approval, 'reprov_ef': failure,
        'abandono_ef': abandonment,
        'target': 0.4 * abandonment + rng.normal(0, 3, n)})


def _independent_panel(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n),
                         'c': rng.normal(size=n),
                         'target': rng.normal(size=n)})


class TestThePairwiseFilterNeedNotCatchIt:
    """Why a rank check earns its place beside the collinearity filter."""

    def test_two_of_the_three_pairs_sit_under_the_ceiling(self):
        panel = _rendimento_panel()
        ceiling = SCIENTIFIC_CONFIG['collinearity_threshold']
        pairs = [('aprov_ef', 'reprov_ef'), ('aprov_ef', 'abandono_ef'),
                 ('reprov_ef', 'abandono_ef')]
        under = [pair for pair in pairs
                 if abs(panel[pair[0]].corr(panel[pair[1]])) < ceiling]
        assert len(under) >= 2, (
            'this panel is caught pairwise, so it does not demonstrate the gap'
        )

    def test_the_dependency_is_nonetheless_exact(self):
        panel = _rendimento_panel()
        total = panel['aprov_ef'] + panel['reprov_ef'] + panel['abandono_ef']
        assert np.allclose(total, 100.0)


class TestRedundantFeatures:

    def test_each_rate_is_determined_by_the_others(self):
        panel = _rendimento_panel()
        found = redundant_features(
            panel, ['aprov_ef', 'reprov_ef', 'abandono_ef'], 'target',
            TOLERANCE)
        assert sorted(found) == ['abandono_ef', 'aprov_ef', 'reprov_ef']

    def test_independent_features_are_not_flagged(self):
        found = redundant_features(_independent_panel(), ['a', 'b', 'c'],
                                   'target', TOLERANCE)
        assert found == {}

    def test_a_single_feature_cannot_be_redundant(self):
        assert redundant_features(_independent_panel(), ['a'], 'target',
                                  TOLERANCE) == {}

    def test_a_near_dependency_is_not_flagged(self):
        """Only exact reproduction counts; correlation is the filter's job."""
        rng = np.random.default_rng(9)
        size = 300
        first = rng.normal(size=size)
        panel = pd.DataFrame({
            'a': first, 'b': 0.95 * first + 0.05 * rng.normal(size=size),
            'target': rng.normal(size=size)})
        assert redundant_features(panel, ['a', 'b'], 'target',
                                  TOLERANCE) == {}


class TestTheAuditReportsRank:

    def test_the_rendimento_shape_is_rank_deficient(self):
        report = audit_panel(_rendimento_panel(), ['aprov_ef', 'reprov_ef', 'abandono_ef'], 'target')
        assert report['design_rank'] == 3
        assert report['rank_deficiency'] == 1

    def test_independent_features_are_full_rank(self):
        report = audit_panel(_independent_panel(), ['a', 'b', 'c'], 'target')
        assert report['design_rank'] == 4
        assert report['rank_deficiency'] == 0

    def test_the_rank_is_not_the_feature_count_minus_the_redundant(self):
        """The arithmetic that looks right and is not.

        Three features with one dependency leave rank two among the columns --
        three counting the intercept -- while all three are individually
        redundant. Subtracting would report zero.
        """
        report = audit_panel(_rendimento_panel(), ['aprov_ef', 'reprov_ef', 'abandono_ef'], 'target')
        naive = len(report['features_audited']) - len(
            report['redundant_features'])
        assert naive == 0
        assert report['design_rank'] != naive

    def test_it_reports_rather_than_halting(self):
        """Rank deficiency is not leakage; aborting would kill a valid run."""
        report = audit_panel(_rendimento_panel(), ['aprov_ef', 'reprov_ef', 'abandono_ef'], 'target')
        assert report['rank_deficiency'] == 1

    def test_too_few_rows_yields_no_verdict(self):
        panel = _independent_panel(n=2)
        report = audit_panel(panel, ['a', 'b', 'c'], 'target')
        assert report['design_rank'] is None


class TestTheAuditIsPersisted:
    """It was assigned to an attribute nothing read."""

    @pytest.mark.parametrize('paradigm', sorted(discover_paradigms()))
    def test_the_paradigm_writes_it(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        assert 'shared_write_feature_audit(' in source, paradigm
        assert 'self._feature_audits = []' in source, paradigm
        assert 'self._feature_audits.append(' in source, paradigm

    def test_the_writer_lands_beside_the_fold_artifacts(self, tmp_path,
                                                        monkeypatch):
        import json

        import core.config as config
        from core.models.hierarchical import write_feature_audit

        monkeypatch.setattr(config, 'get_absolute_output_path',
                            lambda relative: str(tmp_path / relative))
        report = audit_panel(_rendimento_panel(),
                             ['aprov_ef', 'reprov_ef', 'abandono_ef'], 'target')
        path = write_feature_audit([(0, report)], architecture='sql_engine')
        payload = json.loads(Path(path).read_text())
        assert payload['architecture'] == 'sql_engine'
        # Per fold, shaped like the imputation receipt beside it.
        assert payload['folds']['0']['rank_deficiency'] == 1
        assert 'redundant_features' in payload['folds']['0']
        assert payload['checks_across_folds']['joint_reconstruction'] == 'ran'
