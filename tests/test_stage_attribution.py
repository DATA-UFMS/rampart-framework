#!/usr/bin/env python3
"""Where the measured difference between paradigms actually sits.

Recording fold_load_s and fit_predict_s is only half of it: data nobody reads is
the orphan problem again. This aggregates them into a table whose purpose is
discriminating -- if a paradigm's advantage in the total does not appear in
fold_load, the partition-caching explanation does not hold in the numbers.

The fit segment is common to all three by construction, since each materialises to
pandas before scikit-learn, so a ratio near 1.0 there is the expected result and a
ratio far from it means the segments are not measuring what they claim.
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import discover_paradigms


@pytest.fixture
def attribution(tmp_path, monkeypatch):
    """Three paradigms differing only in load time."""
    monkeypatch.setenv('DATASET_NAME', 'worldbank')
    import core.config as config
    monkeypatch.setattr(config, 'get_project_root', lambda: str(tmp_path))

    import importlib
    from benchmarking import derive_stage_attribution as module
    importlib.reload(module)

    loads = {'sql_engine': 1.0, 'task_graph': 0.4, 'dataframe_lib': 0.8}
    for paradigm in sorted(discover_paradigms()):
        directory = Path(config.get_absolute_output_path(
            f'outputs/ml_pipeline/architectures/{paradigm}/models'))
        directory.mkdir(parents=True, exist_ok=True)
        folds = [{'fold_id': i,
                  'fold_load_s': loads[paradigm],
                  'fit_predict_s': 4.0} for i in range(3)]
        (directory / f'hierarchical_analysis_{paradigm}_results.json').write_text(
            json.dumps({'folds': folds}))
        # O estágio de baselines grava um dicionário com chaves fold_<n>, e não
        # uma lista: o leitor tem de aceitar os dois layouts.
        (directory / f'baseline_analysis_{paradigm}_results.json').write_text(
            json.dumps({'baseline_model_results': {
                f'fold_{i}': {'fold_load_s': loads[paradigm] / 2,
                              'fit_predict_s': 1.0} for i in range(3)}}))
    return module


class TestAttribution:

    def test_every_paradigm_is_covered(self, attribution):
        report = attribution.attribute()
        for stage, entry in report['stages'].items():
            assert set(entry['paradigms']) == set(discover_paradigms()), stage

    def test_segments_sum_to_the_total(self, attribution):
        for entry in attribution.attribute()['stages'].values():
            for values in entry['paradigms'].values():
                assert values['fold_load_s'] + values['fit_predict_s'] == \
                    pytest.approx(values['total_s'])

    def test_engine_share_is_a_fraction(self, attribution):
        for entry in attribution.attribute()['stages'].values():
            for values in entry['paradigms'].values():
                assert 0.0 <= values['engine_share'] <= 1.0

    def test_the_fastest_total_is_identified(self, attribution):
        for entry in attribution.attribute()['stages'].values():
            assert entry['fastest_total'] == 'task_graph'

    def test_the_fit_segment_ratio_is_one_when_fits_are_equal(self, attribution):
        """The discriminating check: equal fits, so the gap is all in loading."""
        ratios = attribution.attribute()['stages']['hierarchical']['ratios_against_fastest']
        for paradigm, values in ratios.items():
            assert values['fit_predict_s'] == pytest.approx(1.0), (
                f'{paradigm} differs in the shared fit, which every paradigm '
                f'performs identically after converting to pandas'
            )

    def test_a_load_advantage_shows_up_in_the_load_ratio(self, attribution):
        ratios = attribution.attribute()['stages']['hierarchical']['ratios_against_fastest']
        assert ratios['sql_engine']['fold_load_s'] == pytest.approx(2.5)
        assert ratios['sql_engine']['total_s'] < \
            ratios['sql_engine']['fold_load_s'], (
            'the total dilutes the engine difference, which is why the total '
            'alone cannot support an engine-level explanation'
        )

    def test_the_baseline_layout_is_read(self, attribution):
        """Baselines key folds in a dict; the hierarchical stage uses a list."""
        entry = attribution.attribute()['stages']['baseline']
        assert set(entry['paradigms']) == set(discover_paradigms())
        for values in entry['paradigms'].values():
            assert values['folds'] == 3

    def test_both_ml_stages_are_attributed(self, attribution):
        """Dask wins both on INEP, so attributing one leaves half unmeasured."""
        assert set(attribution.attribute()['stages']) == {'baseline',
                                                          'hierarchical'}

    def test_folds_are_counted(self, attribution):
        for entry in attribution.attribute()['stages'].values():
            for values in entry['paradigms'].values():
                assert values['folds'] == 3
                assert len(values['per_fold']) == 3


class TestMissingDecompositionIsReported:

    def test_results_without_the_fields_are_skipped_not_zeroed(
            self, tmp_path, monkeypatch, capsys):
        """A pre-decomposition result must not enter the sums as zero."""
        monkeypatch.setenv('DATASET_NAME', 'worldbank')
        import core.config as config
        monkeypatch.setattr(config, 'get_project_root', lambda: str(tmp_path))
        import importlib
        from benchmarking import derive_stage_attribution as module
        importlib.reload(module)

        paradigm = sorted(discover_paradigms())[0]
        directory = Path(config.get_absolute_output_path(
            f'outputs/ml_pipeline/architectures/{paradigm}/models'))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'hierarchical_analysis_{paradigm}_results.json').write_text(
            json.dumps({'folds': [{'fold_id': 0, 'r2': 0.5}]}))

        report = module.attribute()
        assert paradigm not in report['stages']['hierarchical']['paradigms']
        assert 'não registra a decomposição' in capsys.readouterr().out

    def test_absent_results_are_reported(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv('DATASET_NAME', 'worldbank')
        import core.config as config
        monkeypatch.setattr(config, 'get_project_root', lambda: str(tmp_path))
        import importlib
        from benchmarking import derive_stage_attribution as module
        importlib.reload(module)

        assert all(not e['paradigms'] for e in module.attribute()['stages'].values())
        assert 'ausentes' in capsys.readouterr().out

    def test_main_succeeds_with_nothing_to_attribute(self, tmp_path,
                                                     monkeypatch):
        monkeypatch.setenv('DATASET_NAME', 'worldbank')
        import core.config as config
        monkeypatch.setattr(config, 'get_project_root', lambda: str(tmp_path))
        import importlib
        from benchmarking import derive_stage_attribution as module
        importlib.reload(module)
        assert module.main() == 0


class TestOutputs:

    def test_json_and_latex_are_written(self, attribution, tmp_path):
        assert attribution.main() == 0
        stats = tmp_path / 'outputs' / 'worldbank' / 'statistics'
        assert (stats / 'stage_attribution.json').exists()
        assert (stats / 'stage_attribution.tex').exists()

    def test_latex_names_both_segments(self, attribution):
        table = attribution._latex(attribution.attribute())
        assert 'Carregamento' in table and 'Ajuste' in table
        assert 'engine' in table.lower()

    def test_latex_has_a_row_per_paradigm(self, attribution):
        table = attribution._latex(attribution.attribute())
        for paradigm in discover_paradigms():
            assert paradigm.replace('_', r'\_') in table

    def test_latex_escapes_every_underscore(self, attribution):
        """Every paradigm name carries one, and the file would not compile.

        The error surfaces to whoever assembles the paper, not to whoever ran
        the pipeline, and by then the run is hours old.
        """
        import re
        table = attribution._latex(attribution.attribute())
        body = [line for line in table.splitlines()
                if '&' in line and not line.startswith('%')]
        assert len(body) >= len(discover_paradigms()), (
            'no data rows: the test would pass on an empty table'
        )
        for line in body:
            assert not re.search(r'(?<!\\)_', line), line

    def test_a_fold_count_mismatch_halts(self, attribution, monkeypatch):
        """Ratios over different fold counts are not attributable to the engine.

        Nine folds of work against eight is a 12% difference that the table
        presents as an engine difference.
        """
        paradigms = sorted(discover_paradigms())
        real = attribution._fold_segments

        def uneven(path):
            segments = real(path)
            if paradigms[0] in str(path):
                return segments[:-1]
            return segments

        monkeypatch.setattr(attribution, '_fold_segments', uneven)
        with pytest.raises(ValueError, match='números de fold'):
            attribution.attribute()
