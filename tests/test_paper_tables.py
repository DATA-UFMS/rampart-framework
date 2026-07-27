#!/usr/bin/env python3
"""The latency table is derived from the artifacts, not transcribed.

Transcribing cell by cell is the mechanism by which a published table stops
matching the data without anything flagging it. The generator reads the
benchmark, the significance and the provenance of each panel, computes the winner
per stage and carries in the caption the commit, the instant and the core budget
-- every latency is conditional on them.

Two decisions that transcription hid are pinned down here:

  * The reported p is the **largest** among the pairs. A stage's claim is "the
    paradigms differ here", and it requires that all pairs differ.
  * A p above the Bonferroni threshold is marked (n.s.) instead of omitted, and
    the cell's bold face does not come to suggest what the p does not support.
"""

import json
import re
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for path in (_ROOT / 'src', _ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.paradigm_registry import discover_paradigms

STAGES = ['processing', 'setup', 'baseline', 'hierarchical']


@pytest.fixture
def tables(tmp_path, monkeypatch):
    """Two panels, with the winner flipping between them."""
    import derive_paper_tables as module
    monkeypatch.setattr(module, '_ROOT', tmp_path)

    paradigms = sorted(discover_paradigms())
    fast_small, fast_large = paradigms[0], paradigms[-1]
    layouts = {
        # In the small panel the first paradigm wins everything.
        'worldbank': {s: {p: (0.1 if p == fast_small else 1.0 + i)
                          for i, p in enumerate(paradigms)} for s in STAGES},
        # In the large one the last wins in the ML stages: that is the crossover.
        'inep_censo': {
            s: {p: ((0.1 if p == fast_small else 1.0 + i)
                    if s in ('processing', 'setup')
                    else (0.5 if p == fast_large else 2.0 + i))
                for i, p in enumerate(paradigms)} for s in STAGES},
    }
    for dataset, layout in layouts.items():
        root = tmp_path / 'outputs' / dataset
        (root / 'benchmarks').mkdir(parents=True)
        (root / 'statistics').mkdir(parents=True)
        rows = [{'run_id': run, 'phase': stage, 'architecture': paradigm,
                 'duration_s': layout[stage][paradigm], 'records': 100}
                for run in range(10) for stage in STAGES
                for paradigm in paradigms]
        pd.DataFrame(rows).to_csv(
            root / 'benchmarks' / 'architectural_benchmark_results.csv',
            index=False)
        pd.DataFrame([
            # n_nonzero_diffs, not n: the signed-rank discards ties and it is
            # that n which fixes the floor. Equal here, no tie in the synthetic
            # panel.
            {'pair': f'{a}_vs_{b}', 'phase': stage, 'n': 10,
             'n_nonzero_diffs': 10, 'wilcoxon_p': 0.00195}
            for stage in STAGES
            for a, b in [(paradigms[0], paradigms[1]),
                         (paradigms[0], paradigms[2]),
                         (paradigms[1], paradigms[2])]
        ]).to_csv(root / 'statistics' / 'significance_summary.csv', index=False)
        (root / 'scientific_config_snapshot.json').write_text(json.dumps({
            'git_commit': 'abc1234567def', 'timestamp': '2026-07-26T18:00:00Z',
            'scientific_config': {'engine_threads': 8, 'blas_threads': 1}}))
    return module, fast_small, fast_large


class TestDerivedFromArtifacts:

    def test_both_panels_are_read(self, tables):
        module, *_ = tables
        report = module.build(['worldbank', 'inep_censo'])
        assert set(report['datasets']) == {'worldbank', 'inep_censo'}

    def test_the_winner_is_computed(self, tables):
        module, fast_small, fast_large = tables
        report = module.build(['worldbank', 'inep_censo'])
        small = {r['stage']: r['winner']
                 for r in report['datasets']['worldbank']['stages']}
        large = {r['stage']: r['winner']
                 for r in report['datasets']['inep_censo']['stages']}
        assert set(small.values()) == {fast_small}
        assert large['baseline'] == fast_large, 'the crossover does not show up'
        assert large['processing'] == fast_small

    def test_every_stage_has_a_cell_per_paradigm(self, tables):
        module, *_ = tables
        report = module.build(['worldbank'])
        for row in report['datasets']['worldbank']['stages']:
            assert set(row['cells']) == set(discover_paradigms())

    def test_a_panel_without_provenance_is_refused(self, tables, tmp_path):
        """A latency without the commit and the budget is comparable to nothing."""
        module, *_ = tables
        (tmp_path / 'outputs' / 'worldbank'
         / 'scientific_config_snapshot.json').unlink()
        assert 'worldbank' not in module.build(['worldbank'])['datasets']

    @pytest.mark.parametrize('field', ['git_commit', 'timestamp'])
    def test_an_incomplete_snapshot_is_refused(self, tables, tmp_path, field):
        """Present but without the field: a caption with commit '?' is worse
        than nothing.
        """
        module, *_ = tables
        snapshot = (tmp_path / 'outputs' / 'worldbank'
                    / 'scientific_config_snapshot.json')
        payload = json.loads(snapshot.read_text())
        payload.pop(field)
        snapshot.write_text(json.dumps(payload))
        with pytest.raises(KeyError, match=field):
            module.build(['worldbank'])

    @pytest.mark.parametrize('field', ['engine_threads', 'blas_threads'])
    def test_a_snapshot_without_the_budget_is_refused(self, tables, tmp_path,
                                                     field):
        module, *_ = tables
        snapshot = (tmp_path / 'outputs' / 'worldbank'
                    / 'scientific_config_snapshot.json')
        payload = json.loads(snapshot.read_text())
        payload['scientific_config'].pop(field)
        snapshot.write_text(json.dumps(payload))
        with pytest.raises(KeyError, match=field):
            module.build(['worldbank'])

    def test_mismatched_core_budgets_are_refused(self, tables, tmp_path):
        """Panels measured with different budgets do not go in the same table."""
        module, *_ = tables
        snapshot = (tmp_path / 'outputs' / 'inep_censo'
                    / 'scientific_config_snapshot.json')
        payload = json.loads(snapshot.read_text())
        payload['scientific_config']['engine_threads'] = 4
        snapshot.write_text(json.dumps(payload))
        report = module.build(['worldbank', 'inep_censo'])
        with pytest.raises(ValueError,
                           match='measured with different core budgets'):
            module.to_latex(report)


class TestSignificanceReporting:

    def test_the_reported_p_is_the_worst_pair(self, tables, tmp_path):
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        frame = pd.read_csv(path)
        frame.loc[frame['phase'] == 'baseline', 'wilcoxon_p'] = [0.001, 0.002,
                                                                0.04]
        frame.to_csv(path, index=False)
        row = next(r for r in module.build(['worldbank'])['datasets']
                   ['worldbank']['stages'] if r['stage'] == 'baseline')
        assert row['worst_pair_p'] == pytest.approx(0.04), (
            'reporting the smallest p would describe the most favourable pair'
        )

    def test_a_p_above_the_threshold_is_marked_not_significant(self, tables,
                                                              tmp_path):
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        frame = pd.read_csv(path)
        frame.loc[frame['phase'] == 'hierarchical', 'wilcoxon_p'] = 0.037
        frame.to_csv(path, index=False)
        report = module.build(['worldbank'])
        row = next(r for r in report['datasets']['worldbank']['stages']
                   if r['stage'] == 'hierarchical')
        assert not row['significant']
        line = next(l for l in module.to_latex(report).splitlines()
                    if 'hierarchical' in l)
        assert '(n.s.)' in line

    def test_the_threshold_follows_the_family_size(self, tables):
        module, *_ = tables
        report = module.build(['worldbank'])
        row = report['datasets']['worldbank']['stages'][0]
        assert row['family_size'] == len(STAGES) * 3
        assert row['threshold'] == pytest.approx(0.05 / row['family_size'])

    def test_the_wilcoxon_floor_is_reported_as_such(self, tables):
        """0.00195 with n=10 is the test's floor, not a measure of precision."""
        module, *_ = tables
        report = module.build(['worldbank'])
        row = report['datasets']['worldbank']['stages'][0]
        assert row['wilcoxon_floor'] == pytest.approx(2 / 2 ** 10)
        assert '(floor, $n$=' in module.to_latex(report)


class TestTheDesignMustBeAbleToResolve:
    """The test's floor against the corrected threshold.

    The two are independent: the floor comes from the repetitions (2/2^n for the
    two-sided Wilcoxon), the threshold comes from the family size, which grows
    with the number of paradigms. With a fourth paradigm the family goes from 15
    to 30 and the threshold falls to 0.00167, below the floor of 0.00195 -- no
    stage can be significant, whatever the data. The table came out normally,
    with every stage marked (n.s.).
    """

    def test_the_current_design_resolves(self, tables):
        """Baseline: 15 tests give 0.00333, above the floor of 0.00195."""
        module, *_ = tables
        row = module.build(['worldbank'])['datasets']['worldbank']['stages'][0]
        assert row['threshold'] > row['wilcoxon_floor']

    @staticmethod
    def _inflate_family(path):
        """Smallest family that does not resolve at n=10: alpha/m < 2/2^10 => m >= 26.

        A fourth paradigm takes 3 pairs to 6, and the family of 4 stages plus
        the total goes from 15 to 30 -- above that limit.
        """
        frame = pd.read_csv(path)
        floor = 2 / 2 ** 10
        copies = [frame]
        while sum(len(c) for c in copies) * 1.0 and \
                0.05 / sum(len(c) for c in copies) >= floor:
            index = len(copies)
            copies.append(frame.assign(pair=frame['pair'] + f'_x{index}'))
        inflated = pd.concat(copies, ignore_index=True)
        assert 0.05 / len(inflated) < floor
        inflated.to_csv(path, index=False)
        return len(inflated)

    def test_a_family_below_the_floor_halts(self, tables, tmp_path):
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        self._inflate_family(path)
        with pytest.raises(ValueError,
                           match='floor of the two-sided Wilcoxon'):
            module.build(['worldbank'])

    def test_the_message_says_how_many_repetitions_would_do(self, tables,
                                                            tmp_path):
        """Without it the operator knows that it stopped, not what to change."""
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        size = self._inflate_family(path)
        with pytest.raises(ValueError) as exc:
            module.build(['worldbank'])
        message = str(exc.value)
        required = math.ceil(math.log2(2.0 / (0.05 / size)))
        assert str(required) in message, message
        assert 2 / 2 ** required <= 0.05 / size, (
            'the suggested number would not actually resolve'
        )


class TestLatexIsCompilable:

    def test_no_unescaped_underscore_outside_comments(self, tables):
        module, *_ = tables
        body = [line for line in
                module.to_latex(module.build(['worldbank', 'inep_censo'])
                                ).splitlines() if not line.startswith('%')]
        for line in body:
            assert not re.search(r'(?<!\\)_', line), line

    def test_the_caption_carries_provenance(self, tables):
        module, *_ = tables
        table = module.to_latex(module.build(['worldbank', 'inep_censo']))
        caption = next(l for l in table.splitlines() if '\\caption' in l)
        assert 'abc1234567' in caption
        assert '2026-07-26' in caption
        assert 'cores per engine' in caption

    def test_the_column_count_matches_the_paradigms(self, tables):
        module, *_ = tables
        table = module.to_latex(module.build(['worldbank']))
        line = next(l for l in table.splitlines() if 'begin{tabular}' in l)
        # Only the specification between braces: the word "tabular" has an r.
        spec = re.search(r'begin\{tabular\}\{([^}]*)\}', line).group(1)
        assert spec.count('r') == len(discover_paradigms()), spec
        assert spec.startswith('ll'), 'panel and stage'
        assert spec.endswith('l'), 'p column'

    def test_the_winner_is_the_only_bold_cell_per_row(self, tables):
        module, *_ = tables
        table = module.to_latex(module.build(['worldbank']))
        for line in table.splitlines():
            if '&' in line and 'tiny' in line:
                assert line.count('\\textbf') == 1, line

    def test_it_says_it_was_generated(self, tables):
        module, *_ = tables
        assert module.to_latex(module.build(['worldbank'])).startswith('%')


class TestProvenanceIsRequired:
    """A latency table without the commit that produced it is not comparable.

    write_environment_snapshot records 'unavailable' when it cannot resolve the
    commit -- running outside a git clone, or from a tarball. The caption
    truncated the commit to ten characters, so the published artifact read "em
    unavailabl": a meaningless string in the exact place a reader looks for
    provenance.
    """

    def test_an_unavailable_commit_halts(self, tables, tmp_path):
        import json
        module, *_ = tables
        path = tmp_path / 'outputs' / 'worldbank' / 'scientific_config_snapshot.json'
        payload = json.loads(path.read_text())
        payload['git_commit'] = 'unavailable'
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match='does not record the commit'):
            module.build(['worldbank'])

    def test_an_empty_commit_halts(self, tables, tmp_path):
        import json
        module, *_ = tables
        path = tmp_path / 'outputs' / 'worldbank' / 'scientific_config_snapshot.json'
        payload = json.loads(path.read_text())
        payload['git_commit'] = ''
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match='does not record the commit'):
            module.build(['worldbank'])

    def test_a_real_commit_passes(self, tables):
        """Otherwise raising unconditionally would satisfy the tests above."""
        module, *_ = tables
        report = module.build(['worldbank'])
        assert report['datasets']['worldbank']['commit']

    def test_the_message_says_what_to_do(self, tables, tmp_path):
        import json
        module, *_ = tables
        path = tmp_path / 'outputs' / 'worldbank' / 'scientific_config_snapshot.json'
        payload = json.loads(path.read_text())
        payload['git_commit'] = 'unavailable'
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError) as exc:
            module.build(['worldbank'])
        assert 'git clone with a clean tree' in str(exc.value)


class TestTheReadmeMatchesTheBudgetCheck:

    def test_no_machine_smaller_than_the_budget_is_documented(self):
        """The README described a VM on which the documented command refuses."""
        import re
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        if str(root / 'src') not in sys.path:
            sys.path.insert(0, str(root / 'src'))
        from core.scientific_config import SCIENTIFIC_CONFIG

        minimum = (SCIENTIFIC_CONFIG['engine_threads']
                   + SCIENTIFIC_CONFIG['blas_threads'] - 1)
        readme = (root / 'README.md').read_text()
        for count in re.findall(r'(\d+)\s*vCPU', readme):
            assert int(count) >= minimum, (
                f'README documents a {count}-vCPU machine while pipeline.py '
                f'requires {minimum} and refuses to start below it'
            )

    def test_the_minimum_is_stated(self):
        from pathlib import Path
        readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
        assert 'eight' in readme or 'at minimum' in readme


class TestTheFloorUsesTheTestsOwnN:
    """The signed-rank drops tied pairs, so its n is not the number of pairs.

    The floor was computed from the pair count. With three ties in ten, the
    effective n is seven and the smallest attainable p is 2/2^7 = 0.0156 --
    eight times the 0.00195 the table reported, and above the corrected
    threshold of 0.0033. The resolution guard read the same wrong n, so it did
    not fire either.
    """

    @staticmethod
    def _with_ties(path, nonzero):
        frame = pd.read_csv(path)
        frame['n_nonzero_diffs'] = nonzero
        frame.to_csv(path, index=False)

    def test_the_floor_follows_the_nonzero_count(self, tables, tmp_path):
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        # 9, not less: below that the floor passes the corrected threshold and
        # the resolution guard halts before the row is assembled.
        self._with_ties(path, 9)
        row = module.build(['worldbank'])['datasets']['worldbank']['stages'][0]
        assert row['wilcoxon_floor'] == pytest.approx(2 / 2 ** 9)
        assert row['n_observations'] == 9

    def test_ties_raise_the_floor(self, tables, tmp_path):
        """The direction matters: dropping pairs makes the floor coarser."""
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        self._with_ties(path, 10)
        loose = module.build(['worldbank'])['datasets']['worldbank']['stages'][0]
        self._with_ties(path, 9)
        tight = module.build(['worldbank'])['datasets']['worldbank']['stages'][0]
        assert tight['wilcoxon_floor'] == pytest.approx(
            2 * loose['wilcoxon_floor'])

    def test_enough_ties_trip_the_resolution_guard(self, tables, tmp_path):
        """The guard read the pair count and stayed silent."""
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        threshold = 0.05 / len(pd.read_csv(path))
        nonzero = 6
        assert 2 / 2 ** nonzero > threshold, (
            'this many ties still resolves, so the test proves nothing'
        )
        self._with_ties(path, nonzero)
        with pytest.raises(ValueError,
                           match='floor of the two-sided Wilcoxon'):
            module.build(['worldbank'])

    def test_an_artifact_without_the_column_halts(self, tables, tmp_path):
        """Falling back to the pair count is what understated the floor."""
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        frame = pd.read_csv(path).drop(columns=['n_nonzero_diffs'])
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match='n_nonzero_diffs'):
            module.build(['worldbank'])

    def test_the_analysis_records_the_column(self):
        """So the guard above is not merely unsatisfiable."""
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root / 'src') not in sys.path:
            sys.path.insert(0, str(root / 'src'))
        source = (root / 'src' / 'statistical_validation'
                  / 'significance_tests.py').read_text()
        assert 'n_nonzero_diffs=n_nonzero' in source
        assert 'np.count_nonzero' in source


class TestAnUntestedPairIsNotHidden:
    """`.max()` skips NaN, so a pair whose test could not run vanished.

    The stage's claim is that the paradigms differ there, and that requires
    every pair to differ. Dropping the pair without a test reported the stage
    against a smaller family than the one it claims.
    """

    @staticmethod
    def _blank_one_pair(path):
        frame = pd.read_csv(path)
        mask = (frame['phase'] == 'baseline')
        first = frame[mask].index[0]
        frame.loc[first, 'wilcoxon_p'] = float('nan')
        frame.to_csv(path, index=False)
        return frame.loc[first, 'pair']

    def test_the_stage_is_not_significant(self, tables, tmp_path):
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        self._blank_one_pair(path)
        row = next(r for r in module.build(['worldbank'])['datasets']
                   ['worldbank']['stages'] if r['stage'] == 'baseline')
        assert not row['significant']
        assert row['pairs_untested'] == 1

    def test_the_other_stages_are_unaffected(self, tables, tmp_path):
        """Otherwise the check could be failing everything indiscriminately."""
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        self._blank_one_pair(path)
        rows = module.build(['worldbank'])['datasets']['worldbank']['stages']
        others = [r for r in rows if r['stage'] != 'baseline']
        assert others
        assert all(r['pairs_untested'] == 0 for r in others)
        assert any(r['significant'] for r in others)

    def test_the_untested_count_is_reported(self, tables):
        module, *_ = tables
        for row in module.build(['worldbank'])['datasets']['worldbank']['stages']:
            assert row['pairs_untested'] == 0

    def test_the_worst_p_does_not_silently_drop_it(self, tables, tmp_path):
        module, *_ = tables
        path = (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')
        self._blank_one_pair(path)
        row = next(r for r in module.build(['worldbank'])['datasets']
                   ['worldbank']['stages'] if r['stage'] == 'baseline')
        assert not np.isfinite(row['worst_pair_p']), (
            'a finite worst p here means the untested pair was skipped'
        )


class TestTheArtifactsComeFromTheSameRun:
    """The table read two files and never checked they agree.

    mean() and std() skip missing values, so a paradigm measured fewer times
    produces a cell that looks like the others: a mean over a smaller n,
    printed beside means over a larger one, with nothing in the table saying
    so. And the significance summary carries its own n, which is where a stale
    artifact shows up.
    """

    @staticmethod
    def _benchmark(tmp_path):
        return (tmp_path / 'outputs' / 'worldbank' / 'benchmarks'
                / 'architectural_benchmark_results.csv')

    @staticmethod
    def _significance(tmp_path):
        return (tmp_path / 'outputs' / 'worldbank' / 'statistics'
                / 'significance_summary.csv')

    def test_a_balanced_run_passes(self, tables):
        module, *_ = tables
        rows = module.build(['worldbank'])['datasets']['worldbank']['stages']
        counts = {row['repetitions'] for row in rows}
        assert len(counts) == 1

    def test_an_unbalanced_stage_halts(self, tables, tmp_path):
        module, *_ = tables
        path = self._benchmark(tmp_path)
        frame = pd.read_csv(path)
        paradigms = sorted(discover_paradigms())
        drop = frame[(frame['phase'] == 'baseline')
                     & (frame['architecture'] == paradigms[0])].index[:1]
        frame.drop(index=drop).to_csv(path, index=False)
        with pytest.raises(ValueError, match='different repetition counts'):
            module.build(['worldbank'])

    def test_a_missing_duration_counts_as_a_missing_repetition(self, tables,
                                                               tmp_path):
        """NaN is skipped by mean(); it must not be skipped by the count."""
        module, *_ = tables
        path = self._benchmark(tmp_path)
        frame = pd.read_csv(path)
        paradigms = sorted(discover_paradigms())
        target = frame[(frame['phase'] == 'baseline')
                       & (frame['architecture'] == paradigms[0])].index[0]
        frame.loc[target, 'duration_s'] = float('nan')
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match='different repetition counts'):
            module.build(['worldbank'])

    def test_artifacts_from_different_runs_halt(self, tables, tmp_path):
        module, *_ = tables
        path = self._significance(tmp_path)
        frame = pd.read_csv(path)
        frame['n'] = frame['n'] + 1
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError,
                           match='do not come from the same run'):
            module.build(['worldbank'])

    def test_the_repetition_count_reaches_the_row(self, tables, tmp_path):
        module, *_ = tables
        frame = pd.read_csv(self._benchmark(tmp_path))
        expected = int(frame['run_id'].nunique())
        for row in module.build(['worldbank'])['datasets']['worldbank']['stages']:
            assert row['repetitions'] == expected
