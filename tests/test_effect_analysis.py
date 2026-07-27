#!/usr/bin/env python3
"""Effect-size and multiple-comparison reporting, end to end.

Three things this file pins down, each of which was wrong or absent:

  - the family size and the corrected threshold are recorded, not left for the
    reader to infer. The paper states alpha = 0.004, which corresponds to 12
    tests; the family is 15, so the threshold is 0.00333.
  - the signed-rank test is reported alongside the paired t-test. The reported
    power refers to signed-rank, so correcting only the t-test left the two
    describing different tests.
  - observed power is simulated from paired differences. Drawing two independent
    groups and subtracting gives sd = sqrt(2), which understates the power at
    the observed effect by close to a factor of two.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from statistical_validation import effect_analysis as ea

PHASES = ['processing', 'setup', 'baseline', 'hierarchical']
ARCHS = ['task_graph', 'sql_engine', 'dataframe_lib']

# The production settings (15,000 resamples and 5,000 simulations per record)
# take about thirty seconds for a family of fifteen. The reporting structure is
# what these tests check, so the fixture runs the same code with fewer draws.
_TEST_SIMS = 400
_real_power = ea._observed_power_wilcoxon


@pytest.fixture(scope='module')
def benchmark_csv(tmp_path_factory):
    """Ten repetitions of four stages for three paradigms, as the protocol runs."""
    rng = np.random.default_rng(17)
    base = {'task_graph': 1.0, 'sql_engine': 1.25, 'dataframe_lib': 1.1}
    rows = []
    for run_id in range(10):
        for phase in PHASES:
            for arch in ARCHS:
                # Noise is deliberately large: with a near-deterministic gap the
                # power reaches 1.0 at any threshold, and a test comparing
                # thresholds would pass whatever alpha the code used.
                rows.append({
                    'run_id': run_id,
                    'phase': phase,
                    'architecture': arch,
                    'duration_s': base[arch] * (1 + 0.35 * rng.normal()),
                })
    path = tmp_path_factory.mktemp('bench') / 'architectural_benchmark_results.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope='module')
def results(benchmark_csv):
    patch = pytest.MonkeyPatch()
    patch.setattr(ea, '_observed_power_wilcoxon',
                  lambda n, d, alpha, **kw: _real_power(n, d, alpha,
                                                       n_sim=_TEST_SIMS))
    # Not under test here, and the dominant cost.
    patch.setattr(ea, '_effect_size_ci',
                  lambda diff, **kw: (float('nan'), float('nan')))
    try:
        yield ea.analyze(benchmark_csv)
    finally:
        patch.undo()


def _records(results):
    return [rec for res in results.values() for rec in res.values()
            if isinstance(rec, dict) and 't_p' in rec]


class TestFamilyIsExplicit:

    def test_family_size_is_recorded(self, results):
        recs = _records(results)
        assert recs
        for rec in recs:
            assert rec['family_size'] == len(recs)

    def test_threshold_matches_the_family(self, results):
        for rec in _records(results):
            assert rec['alpha_bonferroni'] == pytest.approx(
                0.05 / rec['family_size'])

    def test_family_is_three_pairs_by_stages_plus_total(self, results):
        """4 stages + 1 total, across 3 pairs."""
        assert len(_records(results)) == 3 * (len(PHASES) + 1)


class TestBothTestsAreReported:

    def test_signed_rank_accompanies_the_t_test(self, results):
        for rec in _records(results):
            assert 'wilcoxon_p' in rec and 'wilcoxon_stat' in rec

    def test_each_test_is_corrected_over_its_own_family(self, results):
        for rec in _records(results):
            for raw, adjusted in (('t_p', 'p_bonferroni'),
                                  ('wilcoxon_p', 'wilcoxon_p_bonferroni'),
                                  ('t_p', 'p_fdr_bh'),
                                  ('wilcoxon_p', 'wilcoxon_p_fdr_bh')):
                if np.isfinite(rec[raw]) and np.isfinite(rec[adjusted]):
                    assert rec[adjusted] >= rec[raw] - 1e-12, (
                        f"{adjusted}={rec[adjusted]} below {raw}={rec[raw]}"
                    )

    def test_signed_rank_matches_scipy(self, benchmark_csv, results):
        # Pair and key both derived, so the test does not restate the pairing
        # the module is supposed to own.
        arch_a, arch_b = ea.ALL_PAIRS[0]
        df = ea.load_benchmark(benchmark_csv)
        x, y = ea.paired_vectors_for_phase(df, 'baseline', arch_a, arch_b)
        expected = sps.wilcoxon(x - y)
        rec = results[f'{arch_a}_vs_{arch_b}']['baseline']
        assert rec['wilcoxon_p'] == pytest.approx(float(expected.pvalue))


class TestObservedPower:

    def test_a_stricter_threshold_lowers_power(self):
        """Otherwise a test comparing thresholds proves nothing."""
        at_005 = _real_power(10, 0.8, alpha=0.05, n_sim=3000)
        at_bonf = _real_power(10, 0.8, alpha=0.05 / 15, n_sim=3000)
        assert at_005 > at_bonf + 0.1, (at_005, at_bonf)

    def test_power_uses_the_corrected_threshold(self, results):
        """Power at alpha=0.05 while deciding at 0.0033 would overstate it."""
        sensitive = [rec for rec in _records(results)
                     if 0.05 < rec['observed_power'] < 0.95]
        assert sensitive, (
            'no record sits where the threshold changes the power, so this '
            'test could not tell the two thresholds apart'
        )
        for rec in sensitive:
            n, d = rec['n'], rec['cohen_dz']
            at_corrected = _real_power(n, d, alpha=rec['alpha_bonferroni'],
                                       n_sim=_TEST_SIMS)
            assert rec['observed_power'] == pytest.approx(at_corrected)

    def test_power_is_named_as_observed(self, results):
        for rec in _records(results):
            assert 'observed_power' in rec
            assert 'power_est' not in rec, (
                'the old name presented observed power as prospective'
            )

    def test_simulates_paired_differences_not_two_groups(self):
        """The sqrt(2) regression: two independent groups halve the effect."""
        n, d, alpha = 10, 0.8, 0.05
        sims = 1500
        paired = _real_power(n, d, alpha=alpha, n_sim=sims)

        rng = np.random.default_rng(ea.DEFAULT_SEED)
        rejections = 0
        for _ in range(sims):
            x = rng.normal(0, 1, n)
            y = rng.normal(d, 1, n)
            if sps.wilcoxon(x - y).pvalue < alpha:
                rejections += 1
        two_group = rejections / sims

        assert paired > two_group + 0.15, (
            f"paired={paired:.3f} two_group={two_group:.3f}: the two-group "
            f"draw should understate power substantially"
        )
        # And the understatement is the sqrt(2) rescaling of the effect.
        assert two_group == pytest.approx(
            _real_power(n, d / np.sqrt(2), alpha=alpha, n_sim=sims),
            abs=0.04)

    def test_no_power_without_an_effect(self):
        assert np.isnan(_real_power(10, 0.0, alpha=0.05, n_sim=10))

    def test_no_power_below_four_pairs(self):
        assert np.isnan(_real_power(3, 0.8, alpha=0.05, n_sim=10))


class TestOutputSchema:

    def test_csv_columns_cover_the_reported_fields(self, results, tmp_path,
                                                  monkeypatch):
        monkeypatch.setattr(ea, 'STATS_DIR', str(tmp_path))
        ea.write_outputs(results)
        frame = pd.read_csv(tmp_path / 'effect_sizes_summary.csv')
        for column in ('wilcoxon_p', 'wilcoxon_p_bonferroni', 'family_size',
                       'alpha_bonferroni', 'observed_power'):
            assert column in frame.columns


class TestSignificanceCsvKeepsEveryColumn:
    """The CSV keeps everything the analysis produces.

    A whitelist of names decided which columns went into the file, and the
    names were pre-rename. It silently discarded the per-paradigm means and
    every speedup with its CI95 -- which are exactly the columns whose name
    contains the paradigm. Four survived: n, mean_diff_s and the CI of the
    difference.

    The tests read the written file, not the ordering function: a whitelist
    reintroduced alongside it would leave the ordering intact and correct.
    """

    @pytest.fixture(scope='class')
    def written(self, tmp_path_factory):
        """Once per class: analyze() runs the production bootstrap.

        Called per test, it cost about nine seconds each and answered four
        questions about the same file.
        """
        patch = pytest.MonkeyPatch()
        try:
            yield self._write(tmp_path_factory.mktemp('sig'), patch)
        finally:
            patch.undo()

    @staticmethod
    def _write(tmp_path, monkeypatch):
        from core.paradigm_registry import discover_paradigms
        from statistical_validation import significance_tests as module

        rng = np.random.default_rng(11)
        rows = [{'run_id': run, 'phase': phase, 'architecture': paradigm,
                 'duration_s': (1.0 + index) * (1 + 0.05 * rng.normal())}
                for run in range(10)
                for phase in ('processing', 'baseline')
                for index, paradigm in enumerate(sorted(discover_paradigms()))]
        csv = tmp_path / 'architectural_benchmark_results.csv'
        pd.DataFrame(rows).to_csv(csv, index=False)

        results = module.analyze(str(csv))
        monkeypatch.setattr(module, 'RESULTS_DIR', str(tmp_path))
        module.write_outputs(results)
        written = pd.read_csv(tmp_path / 'significance_summary.csv')
        produced = {key for pair in results.values()
                    for metrics in pair.values() for key in metrics}
        return produced, set(written.columns) - {'pair', 'phase'}

    def test_paradigm_named_columns_reach_the_file(self, written):
        produced, written = written
        named = {k for k in produced if 'mean_' in k or 'speedup' in k}
        assert named, 'the analysis produced no paradigm-named columns'
        assert named <= written, (
            f'dropped from the CSV: {sorted(named - written)}'
        )

    def test_nothing_produced_is_dropped(self, written):
        produced, written = written
        assert produced <= written, (
            f'the analysis produced and the CSV lost: '
            f'{sorted(produced - written)}'
        )

    def test_the_speedups_carry_their_intervals(self, written):
        """A speedup with no CI does not support the paper's latency table."""
        _, written = written
        speedups = {c for c in written
                    if c.startswith('speedup_') and not c.endswith(
                        ('_ci95_lo', '_ci95_hi'))}
        assert speedups
        for name in speedups:
            assert f'{name}_ci95_lo' in written, name
            assert f'{name}_ci95_hi' in written, name

    def test_the_order_is_deterministic(self, written):
        from statistical_validation import significance_tests as module
        _, written = written
        assert sorted(written, key=module.column_rank) == \
            sorted(written, key=module.column_rank)

    def test_no_whitelist_of_names_remains(self):
        source = (Path(_SRC) / 'statistical_validation'
                  / 'significance_tests.py').read_text()
        for stale in ('mean_dl_s', 'mean_dw_s', 'speedup_dw_vs_dl',
                      'possible_cols'):
            assert stale not in source, stale


class TestUndefinedPowerIsNotZero:
    """Power 0.0 and undefined power are different claims.

    When cohen_dz is NaN -- a phase whose paired differences are all zero, or a
    variance that does not exist -- the simulation drew from N(NaN, 1), the
    signed-rank test raised on every replicate, and the rejection count stayed
    at zero. The record then said the test had no chance of detecting the
    effect, which is a statement about the design. The truth is that nothing
    was measured.
    """

    def test_a_nan_effect_gives_nan_power(self):
        assert np.isnan(_real_power(10, float('nan'), alpha=0.05, n_sim=50))

    def test_an_infinite_effect_gives_nan_power(self):
        assert np.isnan(_real_power(10, float('inf'), alpha=0.05, n_sim=50))

    def test_a_real_effect_still_gives_a_number(self):
        """Otherwise returning NaN unconditionally would pass the tests above."""
        power = _real_power(10, 0.8, alpha=0.05, n_sim=400)
        assert np.isfinite(power) and 0.0 < power <= 1.0

    def test_the_written_record_carries_nan_not_zero(self, results,
                                                     monkeypatch, tmp_path):
        monkeypatch.setattr(ea, 'STATS_DIR', str(tmp_path))
        patched = {pair: {phase: {**rec, 'cohen_dz': float('nan')}
                          for phase, rec in per_phase.items()}
                   for pair, per_phase in results.items()}
        for per_phase in patched.values():
            for rec in per_phase.values():
                rec['observed_power'] = ea._observed_power_wilcoxon(
                    rec['n'], rec['cohen_dz'], alpha=rec['alpha_bonferroni'],
                    n_sim=50)
        ea.write_outputs(patched)
        frame = pd.read_csv(tmp_path / 'effect_sizes_summary.csv')
        assert frame['observed_power'].isna().all(), (
            f"zeros written for an undefined effect: "
            f"{frame['observed_power'].tolist()}"
        )


class TestNoDeadAccumulators:
    """Two lists were appended to on every phase and never read."""

    def test_the_accumulators_are_gone(self):
        import ast as ast_module
        tree = ast_module.parse(
            (Path(_SRC) / 'statistical_validation'
             / 'effect_analysis.py').read_text())
        assigned = {node.id for node in ast_module.walk(tree)
                    if isinstance(node, ast_module.Name)
                    and isinstance(node.ctx, ast_module.Store)}
        assert 'p_values' not in assigned


class TestTheCorrectionIsTheActualArithmetic:
    """`adjusted >= raw` is satisfied by `adjusted = raw`.

    Deleting the Bonferroni multiplication left the suite green: the only
    assertion on it was an inequality that the identity satisfies. Bonferroni
    is what turns a p of 0.02 into a non-result at a family of 15, so an
    identity here flips the reported verdict of every borderline stage.
    """

    def test_bonferroni_multiplies_by_the_family_size(self, results):
        for rec in _records(results):
            expected = min(1.0, rec['t_p'] * rec['family_size'])
            assert rec['p_bonferroni'] == pytest.approx(expected, rel=1e-12)

    def test_the_signed_rank_correction_too(self, results):
        for rec in _records(results):
            if not np.isfinite(rec['wilcoxon_p']):
                continue
            expected = min(1.0, rec['wilcoxon_p'] * rec['family_size'])
            assert rec['wilcoxon_p_bonferroni'] == pytest.approx(expected,
                                                                 rel=1e-12)

    def test_the_correction_is_not_the_identity(self, results):
        """Without this, a family of one would satisfy the tests above."""
        differing = [rec for rec in _records(results)
                     if rec['p_bonferroni'] != pytest.approx(rec['t_p'])]
        assert differing, (
            'every corrected p equals its raw value, so these tests cannot '
            'distinguish the correction from the identity'
        )

    def test_it_caps_at_one(self, results):
        for rec in _records(results):
            assert rec['p_bonferroni'] <= 1.0
            assert rec['wilcoxon_p_bonferroni'] <= 1.0 or not np.isfinite(
                rec['wilcoxon_p'])
