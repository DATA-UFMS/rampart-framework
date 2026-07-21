#!/usr/bin/env python3
"""A tabela de latência é derivada dos artefatos, não transcrita.

Transcrever célula a célula é o mecanismo pelo qual uma tabela publicada deixa de
corresponder aos dados sem que nada acuse. O gerador lê o benchmark, a
significância e a procedência de cada painel, computa o vencedor por estágio e
carrega na legenda o commit, o instante e o orçamento de núcleos -- toda latência
é condicional a eles.

Duas decisões que a transcrição escondia ficam pinadas aqui:

  * O p reportado é o **maior** entre os pares. A afirmação de um estágio é "os
    paradigmas diferem aqui", e ela exige que todos os pares difiram.
  * Um p acima do limiar de Bonferroni é marcado (n.s.) em vez de omitido, e o
    negrito da célula não passa a sugerir o que o p não sustenta.
"""

import json
import re
import math
import sys
from pathlib import Path

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
    """Dois painéis, com o vencedor invertendo entre eles."""
    import derive_paper_tables as module
    monkeypatch.setattr(module, '_ROOT', tmp_path)

    paradigms = sorted(discover_paradigms())
    fast_small, fast_large = paradigms[0], paradigms[-1]
    layouts = {
        # No painel pequeno o primeiro paradigma ganha em tudo.
        'worldbank': {s: {p: (0.1 if p == fast_small else 1.0 + i)
                          for i, p in enumerate(paradigms)} for s in STAGES},
        # No grande o último ganha nos estágios de ML: é o crossover.
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
            {'pair': f'{a}_vs_{b}', 'phase': stage, 'n': 10,
             'wilcoxon_p': 0.00195}
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
        assert large['baseline'] == fast_large, 'o crossover não aparece'
        assert large['processing'] == fast_small

    def test_every_stage_has_a_cell_per_paradigm(self, tables):
        module, *_ = tables
        report = module.build(['worldbank'])
        for row in report['datasets']['worldbank']['stages']:
            assert set(row['cells']) == set(discover_paradigms())

    def test_a_panel_without_provenance_is_refused(self, tables, tmp_path):
        """Latência sem o commit e o orçamento não é comparável a nada."""
        module, *_ = tables
        (tmp_path / 'outputs' / 'worldbank'
         / 'scientific_config_snapshot.json').unlink()
        assert 'worldbank' not in module.build(['worldbank'])['datasets']

    @pytest.mark.parametrize('field', ['git_commit', 'timestamp'])
    def test_an_incomplete_snapshot_is_refused(self, tables, tmp_path, field):
        """Presente mas sem o campo: uma legenda com commit '?' é pior que nada."""
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
        """Painéis medidos com orçamentos distintos não vão na mesma tabela."""
        module, *_ = tables
        snapshot = (tmp_path / 'outputs' / 'inep_censo'
                    / 'scientific_config_snapshot.json')
        payload = json.loads(snapshot.read_text())
        payload['scientific_config']['engine_threads'] = 4
        snapshot.write_text(json.dumps(payload))
        report = module.build(['worldbank', 'inep_censo'])
        with pytest.raises(ValueError, match='orçamentos'):
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
            'reportar o menor p descreveria o par mais favorável'
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
        """0,00195 com n=10 é o piso do teste, não uma medida de precisão."""
        module, *_ = tables
        report = module.build(['worldbank'])
        row = report['datasets']['worldbank']['stages'][0]
        assert row['wilcoxon_floor'] == pytest.approx(2 / 2 ** 10)
        assert 'piso' in module.to_latex(report)


class TestTheDesignMustBeAbleToResolve:
    """Piso do teste contra limiar corrigido.

    Os dois são independentes: o piso vem das repetições (2/2^n para o Wilcoxon
    bilateral), o limiar vem do tamanho da família, que cresce com o número de
    paradigmas. Com um quarto paradigma a família passa de 15 para 30 e o
    limiar cai para 0,00167, abaixo do piso de 0,00195 -- nenhum estágio pode
    ser significativo, qualquer que seja o dado. A tabela saía normalmente, com
    todos os estágios marcados (n.s.).
    """

    def test_the_current_design_resolves(self, tables):
        """Base: 15 testes dão 0,00333, acima do piso de 0,00195."""
        module, *_ = tables
        row = module.build(['worldbank'])['datasets']['worldbank']['stages'][0]
        assert row['threshold'] > row['wilcoxon_floor']

    @staticmethod
    def _inflate_family(path):
        """Menor família que não resolve com n=10: alpha/m < 2/2^10 => m >= 26.

        Um quarto paradigma leva 3 pares a 6, e a família de 4 estágios mais o
        total passa de 15 para 30 -- acima desse limite.
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
        with pytest.raises(ValueError, match='piso do Wilcoxon'):
            module.build(['worldbank'])

    def test_the_message_says_how_many_repetitions_would_do(self, tables,
                                                            tmp_path):
        """Sem isso o operador sabe que parou, não o que mudar."""
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
            'o número sugerido não resolveria de fato'
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
        assert 'núcleos por engine' in caption

    def test_the_column_count_matches_the_paradigms(self, tables):
        module, *_ = tables
        table = module.to_latex(module.build(['worldbank']))
        line = next(l for l in table.splitlines() if 'begin{tabular}' in l)
        # Só a especificação entre chaves: a palavra "tabular" contém um r.
        spec = re.search(r'begin\{tabular\}\{([^}]*)\}', line).group(1)
        assert spec.count('r') == len(discover_paradigms()), spec
        assert spec.startswith('ll'), 'painel e estágio'
        assert spec.endswith('l'), 'coluna de p'

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
        with pytest.raises(ValueError, match='não registra o commit'):
            module.build(['worldbank'])

    def test_an_empty_commit_halts(self, tables, tmp_path):
        import json
        module, *_ = tables
        path = tmp_path / 'outputs' / 'worldbank' / 'scientific_config_snapshot.json'
        payload = json.loads(path.read_text())
        payload['git_commit'] = ''
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match='não registra o commit'):
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
        assert 'clone git' in str(exc.value)


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
        assert 'oito' in readme or 'no mínimo' in readme
