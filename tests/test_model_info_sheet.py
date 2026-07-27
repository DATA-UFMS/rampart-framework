#!/usr/bin/env python3
"""The info sheet's answers are derived, pending, or declared unanswerable.

Kapoor & Narayanan propose the sheet as the instrument for detecting leakage,
and name its limitation: its claims cannot be verified without computational
reproducibility. This framework is that apparatus, so the answers it can reach
come out of artifacts rather than out of the author.

What the sheet must never do is manufacture the rest. L2 admits no automated
verdict -- K&N decline to subdivide it because the judgment "requires domain
knowledge" -- and L3.2 and L3.3 ask the researcher to reason about dependence
and selection. Those are marked, not filled.
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'scripts'))

import derive_model_info_sheet as sheet
from core.paradigm_registry import discover_paradigms

PARADIGMS = sorted(discover_paradigms())


def _write_artifacts(root, dataset='worldbank'):
    """The artifacts a completed run leaves behind."""
    base = root / 'outputs' / dataset
    for paradigm in PARADIGMS:
        prep = base / 'ml_pipeline' / 'architectures' / paradigm / 'prep'
        prep.mkdir(parents=True, exist_ok=True)
        (prep / f'temporal_folds_{paradigm}.json').write_text(json.dumps({
            'folds': [{'fold_id': 0, 'train_start': 2000, 'train_end': 2007,
                       'val_start': 2010, 'val_end': 2011,
                       'test_start': 2014, 'test_end': 2015,
                       'fit_to_test_gap': 6, 'information_horizon_years': 2},
                      {'fold_id': 1, 'train_start': 2000, 'train_end': 2008,
                       'val_start': 2011, 'val_end': 2012,
                       'test_start': 2015, 'test_end': 2016,
                       'fit_to_test_gap': 6, 'information_horizon_years': 2}]}))
        (prep / f'fold_imputation_{paradigm}.json').write_text(json.dumps({
            'across_folds': {'train': {'rows': 100, 'total': 12,
                                       'fraction': 0.12},
                             'apply_0': {'rows': 40, 'total': 8,
                                         'fraction': 0.2}}}))
        (prep / f'feature_selection_{paradigm}.json').write_text(json.dumps({
            'temporal_scope': 'train_only (<=2007)',
            'features_selected': 5, 'total_features_analyzed': 20,
            'selection_bounds': {'abs_correlation_floor': 0.15,
                                 'abs_correlation_ceiling': 0.8,
                                 'floor_was_relaxed': False},
            'target_correlations': {'gini_index': 0.42,
                                    'gdp_per_capita': -0.51}}))
        # The re-audit of the set the models train on, lags included. Written
        # by the models rather than by setup, and it is what carries the proxy
        # and identity verdicts -- the selection artifact above holds neither.
        (prep / f'feature_audit_{paradigm}.json').write_text(json.dumps({
            'creation_timestamp': '2026-07-27T10:30:00',
            'features_audited': ['gini_index', 'gdp_per_capita',
                                 'dropout_rate_lag_2', 'dropout_rate_lag_3'],
            'proxy_correlation_threshold': 0.8,
            'identity_r2_threshold': 0.95,
            'joint_reconstruction_r2': 0.37,
            'full_set_reconstruction_r2': 0.71,
            'autoregressive_exemptions': {'dropout_rate_lag_2': 0.86,
                                          'dropout_rate_lag_3': 0.79}}))
    (base / 'collection' / 'raw_data').mkdir(parents=True, exist_ok=True)
    (base / 'collection' / 'raw_data' / 'target_coverage.json').write_text(
        json.dumps({'rows_before': 768, 'rows_after': 500,
                    'rows_removed_missing_target': 268,
                    'observed_fraction': {'gini_index': 0.55,
                                          'unemployment_total': 0.47}}))
    (base / 'scientific_config_snapshot.json').write_text(json.dumps({
        'git_commit': 'abc1234def', 'timestamp': '2026-07-27T10:00:00',
        'scientific_config': {'engine_threads': 8}}))


@pytest.fixture
def filled(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.setattr(sheet, '_ROOT', tmp_path)
    monkeypatch.setattr(sheet, '_dataset_root',
                        lambda dataset: tmp_path / 'outputs' / dataset)
    return sheet.build(['worldbank'])['datasets']['worldbank']


@pytest.fixture
def empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sheet, '_ROOT', tmp_path)
    monkeypatch.setattr(sheet, '_dataset_root',
                        lambda dataset: tmp_path / 'outputs' / dataset)
    return sheet.build(['worldbank'])['datasets']['worldbank']


class TestTheDerivableAnswersAreDerived:

    @pytest.mark.parametrize('key', ['L1.1', 'L1.2', 'L1.3', 'L1.4'])
    def test_l1_is_answered_from_artifacts(self, filled, key):
        assert filled['L1'][key]['kind'] == sheet.DERIVED

    def test_l3_1_is_answered_from_artifacts(self, filled):
        assert filled['L3']['L3.1']['kind'] == sheet.DERIVED

    def test_the_fold_count_comes_from_the_artifact(self, filled):
        assert '2 folds' in filled['L1']['L1.1']['text']

    def test_the_imputed_fraction_reaches_the_sheet(self, filled):
        text = filled['L1']['L1.2']['text']
        assert '12.0%' in text and '20.0%' in text

    def test_the_two_temporal_separations_are_distinguished(self, filled):
        """Six years of fitting separation, two of information horizon."""
        text = filled['L3']['L3.1']['text']
        assert '6 anos' in text and '2 anos' in text

    def test_the_removed_rows_reach_the_sheet(self, filled):
        text = filled['L3']['L3.3.derived']['text']
        assert '268' in text and '34.9%' in text

    def test_every_derived_answer_names_its_source(self, filled):
        for argument in ('L1', 'L2', 'L3'):
            for answer in filled[argument].values():
                if answer['kind'] == sheet.DERIVED:
                    assert answer['source'], answer


class TestTheUnanswerableAreDeclared:

    def test_l2_judgment_is_never_derived(self, filled):
        assert filled['L2']['L2.argument']['kind'] == sheet.ARGUMENT

    def test_l3_2_is_never_derived(self, filled):
        assert filled['L3']['L3.2']['kind'] == sheet.ARGUMENT

    def test_l3_3_keeps_an_argument_alongside_the_numbers(self, filled):
        assert filled['L3']['L3.3.derived']['kind'] == sheet.DERIVED
        assert filled['L3']['L3.3.argument']['kind'] == sheet.ARGUMENT

    def test_the_argument_answers_survive_a_complete_run(self, filled, empty):
        """Artifacts must never convert a judgment into a derived answer."""
        for section, key in (('L2', 'L2.argument'), ('L3', 'L3.2'),
                             ('L3', 'L3.3.argument')):
            assert filled[section][key]['kind'] == empty[section][key]['kind']

    def test_l2_says_why_it_cannot_be_automated(self, filled):
        text = filled['L2']['L2.argument']['text']
        assert 'conhecimento de domínio' in text


class TestWithoutArtifacts:

    def test_everything_derivable_is_pending(self, empty):
        for key in ('L1.1', 'L1.2', 'L1.3'):
            assert empty['L1'][key]['kind'] == sheet.PENDING

    def test_a_pending_answer_names_the_artifact_it_waits_for(self, empty):
        for key in ('L1.1', 'L1.2', 'L1.3'):
            assert empty['L1'][key]['source']

    def test_the_enforced_check_does_not_wait_for_a_run(self, empty):
        """L1.4 halts the run; that is true before any artifact exists."""
        assert empty['L1']['L1.4']['kind'] == sheet.DERIVED

    def test_nothing_is_silently_blank(self, empty):
        for argument in ('L1', 'L2', 'L3'):
            for key, answer in empty[argument].items():
                assert answer['text'].strip(), key


class TestTheRendering:

    def test_every_answer_is_labelled(self, filled, tmp_path, monkeypatch):
        monkeypatch.setattr(sheet, '_dataset_root',
                            lambda dataset: tmp_path / 'outputs' / dataset)
        _write_artifacts(tmp_path)
        text = sheet.to_markdown(sheet.build(['worldbank']))
        for marker in ('derivado', 'exige argumento do autor'):
            assert marker in text

    def test_the_invariance_section_is_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sheet, '_dataset_root',
                            lambda dataset: tmp_path / 'outputs' / dataset)
        text = sheet.to_markdown(sheet.build(['worldbank']))
        assert 'Invariância entre implementações' in text
        assert 'pressupõe uma implementação' in text

    def test_reproducibility_is_reported(self, tmp_path, monkeypatch):
        _write_artifacts(tmp_path)
        monkeypatch.setattr(sheet, '_dataset_root',
                            lambda dataset: tmp_path / 'outputs' / dataset)
        report = sheet.build(['worldbank'])
        entry = report['datasets']['worldbank']['reproducibility']
        assert entry['kind'] == sheet.DERIVED
        assert 'abc1234def' in entry['text']


class TestTheSheetSpeaksForEveryParadigm:
    """It quoted whichever paradigm happened to have the file, and said so.

    The loop took the first artifact it found and stopped. With three
    paradigms claimed to run the same protocol over the same data that reads
    as harmless -- until one of them diverges, or one leaves no artifact at
    all, and the sheet describes a single paradigm while presenting itself as
    the study. That is the unverifiable assertion the derived answers exist to
    replace, reintroduced by the derivation itself.
    """

    def _root(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sheet, '_ROOT', tmp_path)
        monkeypatch.setattr(sheet, '_dataset_root',
                            lambda dataset: tmp_path / 'outputs' / dataset)
        return tmp_path / 'outputs' / 'worldbank'

    @staticmethod
    def _prep(base, paradigm):
        return base / 'ml_pipeline' / 'architectures' / paradigm / 'prep'

    def test_divergent_paradigms_do_not_yield_a_derived_answer(
            self, tmp_path, monkeypatch):
        base = self._root(tmp_path, monkeypatch)
        _write_artifacts(tmp_path)
        path = (self._prep(base, PARADIGMS[-1])
                / f'temporal_folds_{PARADIGMS[-1]}.json')
        payload = json.loads(path.read_text())
        payload['folds'][0]['test_end'] = 2099
        path.write_text(json.dumps(payload))

        answers = sheet.build(['worldbank'])['datasets']['worldbank']
        assert answers['L1']['L1.1']['kind'] == sheet.PENDING
        assert 'divergem' in answers['L1']['L1.1']['text']

    def test_divergence_is_named_rather_than_averaged(self, tmp_path,
                                                      monkeypatch):
        base = self._root(tmp_path, monkeypatch)
        _write_artifacts(tmp_path)
        path = (self._prep(base, PARADIGMS[0])
                / f'fold_imputation_{PARADIGMS[0]}.json')
        payload = json.loads(path.read_text())
        payload['across_folds']['train']['fraction'] = 0.99
        path.write_text(json.dumps(payload))

        answers = sheet.build(['worldbank'])['datasets']['worldbank']
        assert answers['L1']['L1.2']['kind'] == sheet.PENDING
        assert '99' not in answers['L1']['L1.2']['text'], (
            'a divergent value reached the sheet as though it were the study')

    def test_a_paradigm_without_the_artifact_is_disclosed(self, tmp_path,
                                                          monkeypatch):
        """Agreement among two of three is agreement, but the reader is told."""
        base = self._root(tmp_path, monkeypatch)
        _write_artifacts(tmp_path)
        missing = PARADIGMS[-1]
        (self._prep(base, missing)
         / f'temporal_folds_{missing}.json').unlink()

        answers = sheet.build(['worldbank'])['datasets']['worldbank']
        assert answers['L1']['L1.1']['kind'] == sheet.DERIVED
        assert missing in answers['L1']['L1.1']['text']

    def test_float_noise_is_not_reported_as_divergence(self, tmp_path,
                                                       monkeypatch):
        """Three engines, one protocol: the last bits are summation order."""
        base = self._root(tmp_path, monkeypatch)
        _write_artifacts(tmp_path)
        path = (self._prep(base, PARADIGMS[0])
                / f'fold_imputation_{PARADIGMS[0]}.json')
        payload = json.loads(path.read_text())
        payload['across_folds']['train']['fraction'] = 0.12 + 1e-15
        path.write_text(json.dumps(payload))

        answers = sheet.build(['worldbank'])['datasets']['worldbank']
        assert answers['L1']['L1.2']['kind'] == sheet.DERIVED


class TestTheProxyVerdictComesFromTheAudit:
    """L2 asserted P3's result from an artifact that does not hold it.

    The sentence "none passed the proxy ceiling, and the set does not
    reconstruct the target above the identity threshold" was emitted whenever
    feature_selection_<paradigm>.json existed. That file is written during
    setup, unconditionally, and contains neither measurement: the ceiling and
    the identity R2 are computed later, by the re-audit, over the set the
    models actually train on -- lags included, which selection never saw.

    So the sheet stated the outcome of a check from a file that would look
    exactly the same had the check never run. On the instrument whose whole
    purpose is that its claims are verifiable, that was the worst place for it.
    """

    def test_the_verdict_is_absent_without_the_audit(self, tmp_path,
                                                     monkeypatch):
        monkeypatch.setattr(sheet, '_ROOT', tmp_path)
        monkeypatch.setattr(sheet, '_dataset_root',
                            lambda dataset: tmp_path / 'outputs' / dataset)
        _write_artifacts(tmp_path)
        base = tmp_path / 'outputs' / 'worldbank'
        for paradigm in PARADIGMS:
            (base / 'ml_pipeline' / 'architectures' / paradigm / 'prep'
             / f'feature_audit_{paradigm}.json').unlink()

        answer = sheet.build(['worldbank'])['datasets']['worldbank']['L2']['L2.screen']
        assert answer['kind'] == sheet.PENDING
        # Naming the ceiling as pending is fine; clearing it is the defect.
        assert 'passou do teto' not in answer['text'], (
            'the sheet still states the verdict the audit was to establish')

    def test_the_measured_thresholds_reach_the_sheet(self, filled):
        answer = filled['L2']['L2.screen']
        assert answer['kind'] == sheet.DERIVED
        assert '0.8' in answer['text'], 'the proxy ceiling is not quoted'
        assert '0.95' in answer['text'], 'the identity threshold is not quoted'
        assert '0.37' in answer['text'], 'the measured R2 is not quoted'

    def test_the_autoregressive_exemptions_are_disclosed(self, filled):
        """An exemption granted without its correlation is an exemption hidden."""
        text = filled['L2']['L2.screen']['text']
        assert 'dropout_rate_lag_2' in text
        assert '+0.860' in text

    def test_the_answer_names_both_artifacts(self, filled):
        source = filled['L2']['L2.screen']['source']
        assert 'feature_audit' in source
        assert 'feature_selection' in source

    def test_the_marginal_screen_survives_without_the_audit(self, tmp_path,
                                                            monkeypatch):
        """Losing the verdict must not cost the evidence that does exist."""
        monkeypatch.setattr(sheet, '_ROOT', tmp_path)
        monkeypatch.setattr(sheet, '_dataset_root',
                            lambda dataset: tmp_path / 'outputs' / dataset)
        _write_artifacts(tmp_path)
        base = tmp_path / 'outputs' / 'worldbank'
        for paradigm in PARADIGMS:
            (base / 'ml_pipeline' / 'architectures' / paradigm / 'prep'
             / f'feature_audit_{paradigm}.json').unlink()

        text = sheet.build(['worldbank'])['datasets']['worldbank'][
            'L2']['L2.screen']['text']
        assert 'gini_index' in text
