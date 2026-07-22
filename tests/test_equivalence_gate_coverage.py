#!/usr/bin/env python3
"""O gate de equivalência exige TODOS os paradigmas registrados.

A afirmação é que os três predizem o mesmo. Violações nasciam apenas dentro de
itertools.combinations sobre os paradigmas que escreveram vetores, então um
paradigma que não escreveu nada nunca entrava em uma combinação e nunca gerava
violação: com 2 de 3 o gate declarava 'equivalent' e saía 0.

A assimetria era absurda -- faltar um vetor de um fold produzia violação
'disjoint' e saída 1; faltar todos os vetores de um paradigma inteiro passava.

E era alcançável: o sql_engine era o único paradigma cujos dois modelos saíam 0
ao falhar, então ele podia não escrever em nenhum dos dois estágios enquanto o
pipeline seguia.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import discover_paradigms


def _write_predictions(root: Path, paradigm: str, *, seed: int = 4,
                       stage: str = 'hierarchical') -> None:
    """Vetores idênticos entre paradigmas, como um Delta=0 verdadeiro."""
    rng = np.random.default_rng(seed)
    n = 12
    frame = pd.DataFrame({
        'fold': np.repeat([0, 1], n // 2),
        'model': 'simple_hierarchical',
        'row': list(range(n)),
        'entity': ['AAA', 'BBB'] * (n // 2),
        'y_true': rng.normal(size=n).round(6),
        'y_pred': rng.normal(size=n).round(6),
    })
    # Caminho pela função canônica: reconstruí-lo à mão foi como este fixture
    # gravou onde o carregador não procura, e o teste passou a medir nada.
    from core.prediction_store import predictions_path

    target = Path(predictions_path(paradigm, stage))
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setenv('DATASET_NAME', 'worldbank')
    import core.config as config
    monkeypatch.setattr(config, 'get_project_root', lambda: str(tmp_path))
    import importlib
    from statistical_validation import prediction_equivalence as module
    importlib.reload(module)
    return module, tmp_path


class TestEveryParadigmMustContribute:

    def test_all_present_is_equivalent(self, gate):
        module, root = gate
        for paradigm in discover_paradigms():
            _write_predictions(root, paradigm)
        report = module.verify()
        assert report['status'] == 'equivalent', report.get('detail')

    def test_one_paradigm_missing_is_not_equivalent(self, gate):
        """O caso que passava: 2 de 3 declarava equivalência."""
        module, root = gate
        paradigms = sorted(discover_paradigms())
        for paradigm in paradigms[:-1]:
            _write_predictions(root, paradigm)
        report = module.verify()
        assert report['status'] != 'equivalent', (
            'equivalência declarada sem um dos paradigmas'
        )
        assert report['status'] == 'insufficient_data'
        assert paradigms[-1] in report['detail']

    def test_the_missing_paradigm_is_named(self, gate):
        module, root = gate
        paradigms = sorted(discover_paradigms())
        for paradigm in paradigms[1:]:
            _write_predictions(root, paradigm)
        report = module.verify()
        assert report['paradigms_without_predictions'] == [paradigms[0]]

    def test_nothing_written_at_all_is_not_equivalent(self, gate):
        module, _ = gate
        assert module.verify()['status'] != 'equivalent'

    def test_a_divergence_is_still_caught_when_all_are_present(self, gate):
        """A correção não pode ter desligado a detecção que já funcionava."""
        module, root = gate
        paradigms = sorted(discover_paradigms())
        for paradigm in paradigms:
            _write_predictions(root, paradigm)
        # Uma única predição alterada em um paradigma.
        from core.prediction_store import predictions_path
        path = Path(predictions_path(paradigms[0], 'hierarchical'))
        frame = pd.read_parquet(path)
        frame.loc[0, 'y_pred'] = frame.loc[0, 'y_pred'] + 1e-12
        frame.to_parquet(path, index=False)
        report = module.verify()
        assert report['status'] != 'equivalent'
        assert report['violations']


class TestExitStatus:

    def _run(self, module, argv):
        try:
            return module.run(argv) if 'argv' in module.run.__code__.co_varnames \
                else module.run()
        except SystemExit as exit_code:
            return exit_code.code

    def test_missing_paradigm_exits_non_zero(self, gate, monkeypatch):
        module, root = gate
        for paradigm in sorted(discover_paradigms())[:-1]:
            _write_predictions(root, paradigm)
        monkeypatch.setattr(sys, 'argv', ['prediction_equivalence.py'])
        assert module.run() != 0

    def test_the_escape_hatch_is_explicit(self, gate, monkeypatch):
        """--allow-missing existe, e precisa ser pedido."""
        module, root = gate
        for paradigm in sorted(discover_paradigms())[:-1]:
            _write_predictions(root, paradigm)
        monkeypatch.setattr(sys, 'argv',
                            ['prediction_equivalence.py', '--allow-missing'])
        assert module.run() == 0

    def test_all_present_exits_zero(self, gate, monkeypatch):
        module, root = gate
        for paradigm in discover_paradigms():
            _write_predictions(root, paradigm)
        monkeypatch.setattr(sys, 'argv', ['prediction_equivalence.py'])
        assert module.run() == 0


class TestTheReportIsUsable:

    def test_the_absent_list_drives_the_decision(self):
        """Antes era só impresso; grep confirmava que ninguém o lia."""
        source = (_SRC / 'statistical_validation'
                  / 'prediction_equivalence.py').read_text()
        index = source.index("if missing:")
        decision = source[index:index + 400]
        assert "report['status'] = 'insufficient_data'" in decision
        assert 'return report' in decision


class TestTheGateCoversWhatIsPublished:
    """The benchmark overwrites the artifacts the gate certified.

    The gate runs at the end of stage 5, and the benchmark then re-executes
    setup, baseline and hierarchical `warmup + n` times per paradigm. Each
    execution rewrites the prediction files. What ends up archived is the last
    repetition's output, and nothing had looked at it: the equivalence claim
    covered files that no longer existed.

    Comparing digests across the benchmark also asserts something the paper
    wants: the repetitions are deterministic, so the latency distribution comes
    from runs that all produced the same predictions.
    """

    @pytest.fixture
    def harness(self, tmp_path, monkeypatch):
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import pipeline

        paradigms = sorted(discover_paradigms())
        written = {}
        for paradigm in paradigms:
            for stage in ('baseline', 'hierarchical'):
                path = tmp_path / f'{paradigm}_{stage}.parquet'
                path.write_bytes(f'{paradigm}:{stage}'.encode())
                written[(paradigm, stage)] = path

        monkeypatch.setattr(
            'core.prediction_store.predictions_path',
            lambda paradigm, stage: str(written[(paradigm, stage)]))
        monkeypatch.setattr(pipeline, '_discover', lambda: paradigms)
        return pipeline, written

    def test_untouched_artifacts_pass(self, harness):
        pipeline, _ = harness
        before = pipeline._prediction_digests()
        assert len(before) == len(discover_paradigms()) * 2
        pipeline._assert_benchmark_left_predictions_intact(before)

    def test_a_rewritten_artifact_halts(self, harness):
        """A repetition that predicts something else must not pass silently."""
        pipeline, written = harness
        before = pipeline._prediction_digests()
        target = next(iter(written.values()))
        target.write_bytes(b'a different repetition')
        with pytest.raises(ValueError, match='diferentes das'):
            pipeline._assert_benchmark_left_predictions_intact(before)

    def test_a_removed_artifact_halts(self, harness):
        pipeline, written = harness
        before = pipeline._prediction_digests()
        next(iter(written.values())).unlink()
        with pytest.raises(ValueError, match='removeu'):
            pipeline._assert_benchmark_left_predictions_intact(before)

    def test_an_artifact_that_appears_afterwards_halts(self, harness):
        """It would be published without ever passing the gate."""
        pipeline, written = harness
        target = next(iter(written.values()))
        target.unlink()
        before = pipeline._prediction_digests()
        assert str(target) not in before
        target.write_bytes(b'late arrival')
        with pytest.raises(ValueError, match='nao viu'):
            pipeline._assert_benchmark_left_predictions_intact(before)

    def test_the_digest_is_content_based(self, harness):
        """Mtime or size alone would miss an equal-length rewrite."""
        pipeline, written = harness
        target = next(iter(written.values()))
        original = target.read_bytes()
        before = pipeline._prediction_digests()
        target.write_bytes(bytes(len(original)))
        assert len(target.read_bytes()) == len(original)
        with pytest.raises(ValueError, match='diferentes das'):
            pipeline._assert_benchmark_left_predictions_intact(before)

    def test_the_pipeline_records_before_and_checks_after(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'pipeline.py').read_text()
        record = source.index('predictions_before = _prediction_digests()')
        benchmark = source.index('architectural_benchmark.py")])')
        check = source.index('_assert_benchmark_left_predictions_intact('
                             'predictions_before)')
        assert record < benchmark < check, (
            'the digests must bracket the benchmark, not sit on one side'
        )

    def test_an_absent_artifact_before_the_benchmark_halts(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'pipeline.py').read_text()
        assert 'Nenhum artefato de predicao antes do benchmark' in source
