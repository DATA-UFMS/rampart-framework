#!/usr/bin/env python3
"""P4: feature selection sees only the training window of the first fold.

Choosing features by their agreement with future values of the target is
look-ahead bias (Kapoor & Narayanan, 2023): the feature enters the model because
it works in the period over which the model will be evaluated, and the reported
performance measures the choice, not predictive capability.

Nothing tested this. Four mutations in `run_feature_selection` -- passing the
whole panel to the correlations, pushing `_first_fold_train_end` a hundred years
forward, making `_filter_by_year` inert, and assigning `data_train_only = data`
-- all survived with the suite green. The four have the same observable effect,
and it is that effect the tests here detect: a feature that only correlates with
the target *after* the training window cannot be selected.

The panel is built to discriminate, and the test itself verifies that before
concluding anything: if the correlation inside the window were not negligible
and the correlation over the whole panel did not clear the selection floor,
passing would mean nothing.

The second half of the file covers the P3 gates that run after selection --
proxy over the whole panel, joint reconstruction fitted on the window, excluded
column in the final selection. They were unreachable in the existing tests,
whose probe returns empty correlations: with no feature selected there is
nothing to audit.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import audit_panel

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.base_architecture import BaseArchitectureML
from core.validation import AntiLeakageViolation

YEARS = list(range(2000, 2016))
ENTITIES = ['BRA', 'ARG', 'CHL', 'URY']
TARGET = 'dropout_rate_sql_engine'

# Derived from the config in _first_fold_train_end: start=2000, min_train=8,
# val_len=2, gap=2 give test_start=2014, val=[2010,2011], train_end=2007.
# The test below checks against the formula instead of trusting this number.
TRAIN_END = 2007


class _Config:
    year_column = 'year'
    entity_column = 'country_code'
    entity_name_column = 'country_name'
    stratification_column = None
    target_source_column = 'source_rate'
    feature_columns = ['honest', 'future_only']
    excluded_columns = ['year', 'country_code', 'source_rate']
    temporal_range = (2000, 2015)
    walk_forward_config = {'min_train': 8, 'val_len': 2}


def _panel():
    """Panel where `future_only` only couples to the target after TRAIN_END.

    Inside the window the correlation is zeroed by construction, not by luck of
    the seed: with 32 rows independent noise still gives |r| near 0.27, enough
    for the feature to enter under the relaxed criterion and for the test to
    measure the seed instead of P4. Outside the window the coupling is strong,
    but stays below the proxy ceiling over the whole panel -- otherwise the P3
    audit would abort earlier and mask what P4 does.
    """
    rng = np.random.default_rng(20260726)
    rows = []
    for entity in ENTITIES:
        target = rng.normal(size=len(YEARS))
        honest = 0.6 * target + 0.8 * rng.normal(size=len(YEARS))
        future = np.where(np.array(YEARS) > TRAIN_END,
                          0.95 * target + 0.3 * rng.normal(size=len(YEARS)),
                          rng.normal(size=len(YEARS)))
        for index, year in enumerate(YEARS):
            rows.append({'year': year, 'country_code': entity,
                         'country_name': entity, 'source_rate': 0.0,
                         TARGET: target[index], 'honest': honest[index],
                         'future_only': future[index]})

    panel = pd.DataFrame(rows)
    window = panel['year'] <= TRAIN_END
    x = panel.loc[window, TARGET].to_numpy()
    y = panel.loc[window, 'future_only'].to_numpy()
    beta = np.cov(x, y, bias=True)[0, 1] / x.var()
    panel.loc[window, 'future_only'] = y - beta * x
    return panel


def _probe():
    class Probe(BaseArchitectureML):
        """Real correlations: a probe returning {} passes through vacuously."""

        def setup_environment(self): pass
        def load_data(self): pass
        def validate_data(self, data): pass
        def create_target_implementation(self, data): return data
        def _compute_target_statistics(self, data): pass
        def _validate_temporal_folds(self, data, folds): pass
        def save_folds(self, data, folds): pass
        def prepare_features(self, data, features): return data

        def discover_numeric_columns(self, data):
            return ['honest', 'future_only']

        def compute_feature_correlations(self, data, features):
            frame = self._materialise_pandas(data, list(features) + [TARGET])
            return {feat: float(frame[feat].corr(frame[TARGET]))
                    for feat in features}

        def apply_collinearity_filter(self, data, features, threshold=0.8):
            return list(features)

    return Probe


@pytest.fixture
def architecture(tmp_path):
    return _probe()('sql_engine', str(tmp_path), dataset_config=_Config())


@pytest.fixture
def panel():
    return _panel()


class TestThePanelDiscriminates:
    """Without this, passing does not tell P4 apart from a panel with no
    signal at all."""

    def test_the_late_feature_is_inert_inside_the_window(self, panel):
        window = panel[panel['year'] <= TRAIN_END]
        corr = abs(window['future_only'].corr(window[TARGET]))
        assert corr < 0.10, (
            f'|corr| = {corr:.3f} inside the window: the relaxed selection '
            f'floor is 0.1005, so the feature would enter even under P4 and '
            f'the test would not be measuring P4'
        )

    def test_the_late_feature_clears_the_floor_on_the_full_panel(self, panel):
        corr = panel['future_only'].corr(panel[TARGET])
        assert corr >= 0.15, (
            f'corr = {corr:.3f} over the whole panel: below the selection '
            f'floor, so not even without P4 would the feature be chosen'
        )

    def test_the_late_feature_stays_below_the_proxy_ceiling(self, panel):
        """Otherwise the P3 audit would abort and mask what P4 does."""
        corr = abs(panel['future_only'].corr(panel[TARGET]))
        assert corr < 0.80, corr

    def test_the_honest_feature_is_visible_inside_the_window(self, panel):
        window = panel[panel['year'] <= TRAIN_END]
        assert window['honest'].corr(window[TARGET]) >= 0.15


class TestTheWindowIsTheFirstFoldTrainWindow:

    def test_it_matches_the_configured_folds(self, architecture):
        """End of training = start of validation minus the gap, minus one."""
        cfg = architecture.config
        wf = _Config.walk_forward_config
        gap = int(cfg['temporal_gap_years'])
        test_start = (_Config.temporal_range[0] + wf['min_train']
                      + wf['val_len'] + 2 * gap)
        val_end = test_start - gap - 1
        val_start = val_end - wf['val_len'] + 1
        assert architecture._first_fold_train_end() == val_start - gap - 1

    def test_the_gap_separates_it_from_validation(self, architecture):
        """P2: between training and validation there are years nobody reads."""
        cfg = architecture.config
        gap = int(cfg['temporal_gap_years'])
        train_end = architecture._first_fold_train_end()
        wf = _Config.walk_forward_config
        test_start = (_Config.temporal_range[0] + wf['min_train']
                      + wf['val_len'] + 2 * gap)
        val_start = test_start - gap - 1 - wf['val_len'] + 1
        assert val_start - train_end - 1 == gap

    def test_it_leaves_evaluation_years_outside(self, architecture, panel):
        train_end = architecture._first_fold_train_end()
        assert train_end < max(YEARS), (
            'the window covers the whole panel, so filtering by it restricts '
            'nothing'
        )
        kept = architecture._filter_by_year(panel, max_year=train_end)
        assert len(kept) < len(panel)
        assert kept['year'].max() <= train_end


class TestSelectionIsRestrictedToTheWindow:
    """The invariant. Each of the four rejected mutations fails here."""

    def test_a_feature_that_only_works_later_is_not_selected(
            self, architecture, panel):
        stats = architecture.run_feature_selection(panel)
        assert 'future_only' not in stats['selected_features'], (
            'feature chosen by its agreement with the target in years the '
            'model has yet to predict -- it is the selection P4 exists to block'
        )

    def test_a_feature_that_works_inside_the_window_is_selected(
            self, architecture, panel):
        """Otherwise passing would just be the selection choosing nothing."""
        stats = architecture.run_feature_selection(panel)
        assert 'honest' in stats['selected_features']

    def test_the_correlations_recorded_come_from_the_window(
            self, architecture, panel):
        stats = architecture.run_feature_selection(panel)
        window = panel[panel['year'] <= TRAIN_END]
        for feat, recorded in stats['target_correlations'].items():
            expected = float(window[feat].corr(window[TARGET]))
            assert recorded == pytest.approx(expected, abs=1e-9), (
                f'{feat}: recorded {recorded:.4f}, window {expected:.4f}, '
                f'panel {panel[feat].corr(panel[TARGET]):.4f}'
            )

    def test_the_recorded_scope_names_the_window(self, architecture, panel):
        stats = architecture.run_feature_selection(panel)
        assert str(architecture._first_fold_train_end()) in \
            stats['temporal_scope']


class TestSelectionReadsOnlyWindowRows:
    """Watches the calls: covers the mutation that passes the panel and
    reorders."""

    @staticmethod
    def _record(architecture):
        seen = []
        original = architecture.compute_feature_correlations

        def spy(data, features):
            seen.append(architecture._materialise_pandas(data, ['year']))
            return original(data, features)

        architecture.compute_feature_correlations = spy
        return seen

    def test_the_selection_call_sees_no_evaluation_year(
            self, architecture, panel):
        seen = self._record(architecture)
        architecture.run_feature_selection(panel)
        assert seen, 'the selection never computed a single correlation'
        train_end = architecture._first_fold_train_end()
        assert seen[0]['year'].max() <= train_end, (
            f"the selection read up to {seen[0]['year'].max()}, beyond "
            f"{train_end}"
        )

    def test_the_collinearity_filter_sees_the_same_rows(
            self, architecture, panel):
        seen = []
        original = architecture.apply_collinearity_filter

        def spy(data, features, threshold=0.8):
            seen.append(architecture._materialise_pandas(data, ['year']))
            return original(data, features, threshold)

        architecture.apply_collinearity_filter = spy
        architecture.run_feature_selection(panel)
        assert seen
        assert seen[0]['year'].max() <= architecture._first_fold_train_end()

    def test_the_proxy_audit_reads_the_full_panel(self, architecture, panel):
        """The audit is the opposite of P4, and for a reason declared in the
        code.

        A feature whose correlation only clears the ceiling outside the
        training window is still a proxy; auditing inside the window is what
        let one through.
        """
        seen = self._record(architecture)
        architecture.run_feature_selection(panel)
        assert len(seen) >= 2, 'there was no audit after the selection'
        assert seen[-1]['year'].max() == max(YEARS), (
            'the proxy audit also ended up restricted to the window'
        )


class TestTheLeakageGatesFire:
    """P3 gates that run after selection, and never fired in a test.

    `TestPoolGate` covers the pool gate, which runs before any correlation. The
    three that follow -- excluded column in the final selection, proxy over the
    whole panel, and joint reconstruction of the target -- were unreachable
    because that probe returns empty correlations: with no feature selected,
    there is nothing to audit. Here the correlations are real.
    """

    @staticmethod
    def _run(tmp_path, build, config=None, **overrides):
        panel = _panel()
        columns = build(panel)
        Probe = _probe()

        class Gated(Probe):
            def discover_numeric_columns(self, data):
                return sorted(columns)

        for name, function in overrides.items():
            setattr(Gated, name, function)

        class Config(_Config):
            feature_columns = sorted(columns)

        architecture = Gated('sql_engine', str(tmp_path),
                             dataset_config=Config())
        architecture.config = {**architecture.config, **(config or {})}
        return architecture.run_feature_selection(panel)

    def test_a_proxy_visible_in_the_window_never_reaches_the_audit(
            self, tmp_path):
        """Defence in depth, and the first line is the selection ceiling.

        |r| above the ceiling in the training window makes the feature be
        refused before any audit. Before, it was *selected* -- the comparison
        was signed and the relaxation brought the ceiling down -- and only the
        proxy audit, downstream, blocked it.
        """
        def build(panel):
            panel['proxy'] = 0.98 * panel[TARGET] + 0.02 * panel['honest']
            return ['honest', 'proxy']

        stats = self._run(tmp_path, build)
        assert 'proxy' not in stats['selected_features']
        assert 'honest' in stats['selected_features']

    def test_the_audit_remains_the_second_line(self, tmp_path):
        """The ceiling applies over the window; the audit, over the whole
        panel.

        The lags enter the set after selection and never go through the
        ceiling, so the audit remains the only thing between them and the
        model.
        """
        import numpy as np
        from core.scientific_config import SCIENTIFIC_CONFIG
        from core.validation import AntiLeakageViolation, audit_feature_set

        panel = _panel()
        panel['dropout_rate_lag_0'] = panel[TARGET]
        with pytest.raises(AntiLeakageViolation, match='target reproduction'):
            audit_panel(panel, ['honest', 'dropout_rate_lag_0'], TARGET)

    def test_a_proxy_only_outside_the_window_halts(self, tmp_path):
        """Auditing inside the P4 window is what let one through.

        Inside the window the correlation is moderate, so the feature is
        chosen; over the whole panel it clears the ceiling. An audit restricted
        to the window would see nothing.
        """
        def build(panel):
            late = panel['year'] > TRAIN_END
            panel['proxy'] = np.where(late, panel[TARGET],
                                      0.35 * panel[TARGET]
                                      + 0.94 * panel['honest'])
            return ['honest', 'proxy']

        panel = _panel()
        build(panel)
        window = panel[panel['year'] <= TRAIN_END]
        assert abs(window['proxy'].corr(window[TARGET])) < 0.80, (
            'inside the window it already clears the ceiling, so the test does '
            'not distinguish a whole-panel audit from a window audit'
        )
        assert abs(panel['proxy'].corr(panel[TARGET])) > 0.80

        with pytest.raises(AntiLeakageViolation, match='P3 proxy detection'):
            self._run(tmp_path, build)

    def test_a_negative_proxy_halts(self, tmp_path):
        """Moderate and positive in the window, strong and negative outside it.

        Selection admits it -- |r| in the window is inside the band -- and only
        then does the audit see it. Without the absolute value in the gate, it
        passes.

        The ceiling is lowered to 0.50 in this configuration because the
        correlation over the whole panel saturates near -0.66: the rows in the
        window have a positive relation with the target and pull the joint
        coefficient. The same parameter governs the selection ceiling, so the
        window has to stay below it -- which this panel guarantees and the test
        checks.
        """
        def build(panel):
            rng = np.random.default_rng(5)
            late = (panel['year'] > TRAIN_END).to_numpy()
            panel['proxy'] = np.where(
                late, -5.0 * panel[TARGET],
                0.20 * panel[TARGET] + 1.0 * rng.normal(size=len(panel)))
            return ['proxy']

        threshold = 0.50
        panel = _panel()
        build(panel)
        window = panel[panel['year'] <= TRAIN_END]
        in_window = window['proxy'].corr(window[TARGET])
        assert 0.15 <= abs(in_window) <= threshold, (
            f'|r| = {abs(in_window):.3f} in the window: outside the band the '
            f'selection refuses it and the test does not reach the audit'
        )
        full = panel['proxy'].corr(panel[TARGET])
        assert full < -threshold, (
            f'corr = {full:.3f}: without a negative correlation beyond the '
            f'ceiling, the absolute value in the gate makes no difference and '
            f'the test does not exercise it'
        )
        assert full <= threshold, 'with the sign, the gate would not fire'

        with pytest.raises(AntiLeakageViolation,
                           match='P3 proxy detection') as exc:
            self._run(tmp_path, build,
                      config={'proxy_correlation_threshold': threshold})
        assert 'proxy' in str(exc.value)

    def test_features_that_reconstruct_the_target_halt(self, tmp_path):
        """Additive identity: each part correlates weakly, together they close.

        Pairwise correlation does not see that -- which is exactly why the
        reconstruction gate exists.
        """
        def build(panel):
            rng = np.random.default_rng(7)
            target = panel[TARGET].to_numpy()
            noise = 1.5 * rng.normal(size=len(panel))
            # Orthogonalised: with a non-zero sample covariance the noise
            # enters one part and leaves the other, and the two stop having the
            # same correlation -- one of them falls below the selection floor.
            noise -= (np.cov(target, noise, bias=True)[0, 1]
                      / target.var()) * target
            panel['half_a'] = 0.5 * target + noise
            panel['half_b'] = 0.5 * target - noise
            return ['half_a', 'half_b']

        panel = _panel()
        build(panel)
        for part in ('half_a', 'half_b'):
            corr = abs(panel[part].corr(panel[TARGET]))
            assert 0.15 <= corr < 0.80, (
                f'{part}: |corr| = {corr:.3f}. Outside that range the proxy '
                f'gate fires first and the test does not reach the '
                f'reconstruction'
            )

        with pytest.raises(AntiLeakageViolation,
                           match='P3 joint reconstruction'):
            self._run(tmp_path, build)

    def test_the_reconstruction_is_fitted_on_the_window(self, tmp_path):
        """An exact identity is detected without consulting the evaluation
        years.

        The parts sum to the target exactly inside the window and stop summing
        after it. Fitting over the whole panel dilutes the R2 below the ceiling
        and the identity passes -- which is the opposite of what the gate
        promises.
        """
        def build(panel):
            rng = np.random.default_rng(7)
            target = panel[TARGET].to_numpy()
            noise = 1.5 * rng.normal(size=len(panel))
            noise -= (np.cov(target, noise, bias=True)[0, 1]
                      / target.var()) * target
            late = (panel['year'] > TRAIN_END).to_numpy()
            panel['half_a'] = 0.5 * target + noise
            panel['half_b'] = np.where(
                late, rng.normal(size=len(panel)), 0.5 * target - noise)
            return ['half_a', 'half_b']

        panel = _panel()
        build(panel)
        window = panel[panel['year'] <= TRAIN_END]
        assert abs(window['half_a'] + window['half_b']
                   - window[TARGET]).max() < 1e-9
        assert abs(panel['half_a'] + panel['half_b']
                   - panel[TARGET]).max() > 1.0, (
            'the identity also holds outside the window, so the test does not '
            'distinguish where the fit is done'
        )
        for part in ('half_a', 'half_b'):
            corr = window[part].corr(window[TARGET])
            assert corr >= 0.1005, (
                f'{part}: corr {corr:.3f} in the window, it would not be '
                f'selected'
            )

        with pytest.raises(AntiLeakageViolation,
                           match='P3 joint reconstruction'):
            self._run(tmp_path, build)

    def test_an_excluded_column_in_the_final_selection_halts(self, tmp_path):
        """The collinearity filter returns the list; nothing guarantees it is a
        subset of what went in."""
        def build(panel):
            return ['honest']

        def smuggle(self, data, features, threshold=0.8):
            return list(features) + [self.target_column]

        with pytest.raises(AntiLeakageViolation, match='P3 data separation'):
            self._run(tmp_path, build, apply_collinearity_filter=smuggle)

    def test_a_clean_panel_reaches_the_end(self, tmp_path):
        """Baseline: without this, each test above could be failing for another
        reason."""
        stats = self._run(tmp_path, lambda panel: ['honest'])
        assert stats['selected_features'] == ['honest']
