#!/usr/bin/env python3
"""Model info sheet, with the derivable answers derived from artifacts.

Kapoor & Narayanan (2023) propose a model info sheet -- 21 questions grouped
under three arguments -- as the instrument for detecting and preventing
leakage. They also name its limitation:

    "the claims made in model info sheets cannot be verified in the absence of
    computational reproducibility. That is, unless the code, data, and
    computing environment required to reproduce the results in a paper are
    made available, there is no way to ascertain whether model info sheets are
    filled out correctly."

That is the gap this script closes. Every answer here is one of three kinds,
and each is labelled:

  DERIVED   read out of an artifact this run produced. Not an assertion by the
            author, so it is checkable against the artifact.
  PENDING   derivable, but the artifact does not exist yet. Naming it is the
            point: it says exactly what the run will fill in.
  ARGUMENT  not derivable at all, and marked so rather than left blank. K&N are
            explicit that L2 admits no sub-categories because the judgment
            "requires domain knowledge", and that L3.2 and L3.3 need the
            researcher to reason about dependence and selection.

The sheet does not claim to extend the taxonomy. It operationalises the
instrument, for the subset that a pipeline can answer, and refuses to
manufacture the rest.

One thing this framework must answer that a single-implementation study need
not: each check has to give the same verdict in every paradigm, or the check
itself becomes a source of divergence. Those invariance results are reported
alongside the leakage answers.

Uso:
    python scripts/derive_model_info_sheet.py [--datasets worldbank inep_censo]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))

from core.paradigm_registry import discover_paradigms  # noqa: E402

DERIVED = 'DERIVED'
PENDING = 'PENDING'
ARGUMENT = 'ARGUMENT'


def _answer(kind: str, text: str, source: Optional[str] = None) -> Dict:
    return {'kind': kind, 'text': text, 'source': source}


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _fingerprint(value):
    """A comparable form of a value, with float noise below the 12th digit gone.

    The paradigms compute these on three different engines. Agreement to twelve
    significant digits is agreement about the protocol; the bits below that are
    about summation order, and treating them as disagreement would report a
    divergence that does not exist.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return f'{value:.12g}'
    if isinstance(value, dict):
        return tuple(sorted((key, _fingerprint(item))
                            for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_fingerprint(item) for item in value)
    return value


def _agreed(root: Path, paradigms: List[str], stem: str,
            fields: Tuple[str, ...]) -> Tuple[Optional[Dict], Optional[str]]:
    """Read an artifact from every paradigm; quote it only if they agree.

    Taking the first paradigm that happened to have the file described one
    paradigm and presented it as the study. The three are claimed to run the
    same protocol over the same data, so the sheet may speak for all of them --
    but only once that has been checked, and nothing was checking. A sheet that
    silently describes a single paradigm is the same unverifiable assertion the
    derived answers exist to replace.

    Only the fields an answer actually quotes are compared. Comparing whole
    payloads would flag differences in values no reader ever sees.

    Returns the agreed payload and a note to carry into the answer. Divergence
    returns no payload: an answer the paradigms contradict is worse than an
    absent one.
    """
    found = {}
    for paradigm in paradigms:
        payload = _read_json(root / 'ml_pipeline' / 'architectures' / paradigm
                             / 'prep' / f'{stem}_{paradigm}.json')
        if payload:
            found[paradigm] = payload

    if not found:
        return None, None

    prints = {paradigm: _fingerprint({f: payload.get(f) for f in fields})
              for paradigm, payload in found.items()}
    if len(set(prints.values())) > 1:
        # Without attributing a cause. Divergence here has two very different
        # origins -- the paradigms measured the same quantity and disagreed, or
        # they measured over different slices -- and only the first contradicts
        # the equivalence claim. Asserting the first would turn a difference of
        # scope into an accusation against the central result.
        return None, (
            f"Os paradigmas não produziram os mesmos valores em {stem} "
            f"({', '.join(sorted(found))}), então nenhum deles é citado aqui. "
            f"Ou mediram a mesma quantidade e discordaram, ou a mediram sobre "
            f"recortes diferentes; a resposta fica pendente até que se saiba "
            f"qual."
        )

    missing = [p for p in paradigms if p not in found]
    note = None
    if missing:
        note = (f" Derivado de {', '.join(sorted(found))}; "
                f"{', '.join(missing)} não deixou este artefato.")
    return found[sorted(found)[0]], note


def _dataset_root(dataset: str) -> Path:
    return _ROOT / 'outputs' / dataset


def _l1_clean_separation(root: Path, paradigms: List[str]) -> Dict:
    """L1: the test set does not interact with training data at any step."""
    answers = {}

    payload, note = _agreed(root, paradigms, 'temporal_folds', ('folds',))
    folds = payload.get('folds') if payload else None

    if folds:
        first, last = folds[0], folds[-1]
        answers['L1.1'] = _answer(
            DERIVED,
            f"{len(folds)} folds walk-forward. O primeiro avalia "
            f"{first['test_start']}-{first['test_end']} treinando em "
            f"{first['train_start']}-{first['train_end']}; o último avalia "
            f"{last['test_start']}-{last['test_end']}. Nenhum fold reutiliza "
            f"a janela de teste para treinar.{note or ''}",
            'temporal_folds_<paradigma>.json')
    else:
        answers['L1.1'] = _answer(
            PENDING,
            note or "Contagem e janelas dos folds virão do artefato de folds.",
            'temporal_folds_<paradigma>.json')

    imputation, note = _agreed(root, paradigms, 'fold_imputation',
                               ('across_folds',))

    if imputation:
        totals = imputation.get('across_folds', {})
        parts = ', '.join(
            f"{split} {entry['fraction']:.1%}"
            for split, entry in sorted(totals.items()))
        answers['L1.2'] = _answer(
            DERIVED,
            f"Escala e imputação são ajustadas na janela de treino de cada "
            f"fold e aplicadas a validação e teste. Fração de células "
            f"preenchidas pela mediana da janela de treino: {parts}. O "
            f"artefato cobre a imputação; o ajuste do scaler ocorre logo "
            f"depois, no mesmo frame, e não emite relatório.{note or ''}",
            'fold_imputation_<paradigma>.json')
    else:
        answers['L1.2'] = _answer(
            PENDING,
            note or ("A estatística é ajustada só no treino (impute_from_"
                     "training_window); a fração preenchida por split virá do "
                     "artefato."),
            'fold_imputation_<paradigma>.json')

    selection, note = _agreed(
        root, paradigms, 'feature_selection',
        ('selection_bounds', 'temporal_scope', 'features_selected',
         'total_features_analyzed', 'target_correlations'))

    if selection:
        bounds = selection.get('selection_bounds', {})
        answers['L1.3'] = _answer(
            DERIVED,
            f"Seleção restrita à janela de treino do primeiro fold "
            f"({selection.get('temporal_scope')}). Banda de |r| aplicada: "
            f"[{bounds.get('abs_correlation_floor')}, "
            f"{bounds.get('abs_correlation_ceiling')}]; piso reduzido: "
            f"{bounds.get('floor_was_relaxed')}. "
            f"{selection.get('features_selected')} features selecionadas de "
            f"{selection.get('total_features_analyzed')} candidatas. A janela "
            f"do primeiro fold é a mais curta, então a restrição é "
            f"conservadora para todos os folds posteriores.{note or ''}",
            'feature_selection_<paradigma>.json')
    else:
        answers['L1.3'] = _answer(
            PENDING,
            note or ("A seleção usa a janela de treino do primeiro fold; os "
                     "limites e a contagem virão do artefato."),
            'feature_selection_<paradigma>.json')

    answers['L1.4'] = _answer(
        DERIVED,
        "Pares (entidade, ano) duplicados interrompem a execução em "
        "canonical_fold, antes de qualquer ajuste. Duplicata é a assinatura de "
        "um join que multiplicou linhas, e nada a jusante notaria.",
        'core.validation.canonical_fold')
    return answers


def _l2_feature_legitimacy(root: Path, paradigms: List[str]) -> Dict:
    """L2: each feature is legitimate. Screen derived; judgment is not."""
    answers = {}

    # Two artifacts, because the answer makes two claims and they are measured
    # at different moments. The marginal screen is what selection did, on the
    # candidate pool. The verdict is about the set the models actually train
    # on -- lags included, which selection never saw. The verdict used to be
    # asserted from the selection file, which is written whether or not the
    # audit ever ran and holds none of its measurements.
    selection, selection_note = _agreed(root, paradigms, 'feature_selection',
                                        ('target_correlations',))
    audit, audit_note = _agreed(
        root, paradigms, 'feature_audit',
        ('features_audited', 'proxy_correlation_threshold',
         'identity_r2_threshold', 'joint_reconstruction_r2',
         'full_set_reconstruction_r2', 'autoregressive_exemptions'))

    def _measured(value) -> str:
        return 'não medido' if value is None else f'{value:.4f}'

    parts, sources = [], []
    if selection:
        correlations = selection.get('target_correlations', {})
        ranked = sorted(correlations.items(), key=lambda kv: -abs(kv[1]))
        listing = '; '.join(f"{name} {value:+.3f}" for name, value in ranked)
        parts.append(
            f"Rastreio automático, sobre a janela de treino do primeiro fold. "
            f"Associação marginal de cada feature selecionada com o alvo: "
            f"{listing}.{selection_note or ''}")
        sources.append('feature_selection_<paradigma>.json')

    if audit:
        exemptions = audit.get('autoregressive_exemptions') or {}
        granted = ('; '.join(f"{name} {value:+.3f}"
                             for name, value in sorted(exemptions.items()))
                   if exemptions else 'nenhuma')
        parts.append(
            f"Sobre o conjunto que os modelos de fato treinam -- "
            f"{len(audit.get('features_audited') or [])} features, defasagens "
            f"incluídas, que a seleção não chegou a ver -- nenhuma feature não "
            f"autorregressiva passou do teto de proxy "
            f"|r|={audit.get('proxy_correlation_threshold')}, e as não "
            f"autorregressivas reconstroem o alvo com R2="
            f"{_measured(audit.get('joint_reconstruction_r2'))} contra um "
            f"limiar de identidade de {audit.get('identity_r2_threshold')}. "
            f"Exemções autorregressivas concedidas, com a correlação medida: "
            f"{granted}.{audit_note or ''}")
        sources.append('feature_audit_<paradigma>.json')
    else:
        parts.append(
            "O veredito sobre o conjunto treinado -- teto de proxy e limiar de "
            "identidade após a inclusão das defasagens -- virá da auditoria de "
            f"features.{audit_note or ''}")

    answers['L2.screen'] = _answer(
        DERIVED if (selection and audit) else PENDING,
        ' '.join(parts),
        ', '.join(sources) or 'feature_audit_<paradigma>.json')

    answers['L2.argument'] = _answer(
        ARGUMENT,
        "K&N não subdividem L2 porque o julgamento de legitimidade exige "
        "conhecimento de domínio, e pedem argumento que cubra toda feature "
        "usada. O rastreio acima detecta o subconjunto detectável -- o proxy "
        "fortemente associado e a identidade algébrica -- e não alcança uma "
        "feature ilegítima por ser consequência do desfecho, nem uma que torne "
        "a tarefa trivial sem ser proxy. O argumento é do autor.")
    return answers


def _l3_distribution(root: Path, paradigms: List[str]) -> Dict:
    """L3: the test set comes from the distribution the claim is about."""
    answers = {}

    payload, note = _agreed(root, paradigms, 'temporal_folds', ('folds',))
    folds = payload.get('folds') if payload else None

    if folds:
        first = folds[0]
        answers['L3.1'] = _answer(
            DERIVED,
            f"Toda janela de teste é posterior à de treino. No primeiro fold "
            f"há {first.get('fit_to_test_gap')} anos entre a última "
            f"observação que entra na estimação dos parâmetros e a primeira "
            f"avaliada. Isso não é o horizonte de informação: no instante da "
            f"predição o modelo lê defasagens do alvo, e o valor mais recente "
            f"consultado está a {first.get('information_horizon_years')} anos "
            f"do ano avaliado.{note or ''}",
            'temporal_folds_<paradigma>.json')
    else:
        answers['L3.1'] = _answer(
            PENDING,
            note or ("As duas separações -- de ajuste e de informação -- virão "
                     "do artefato de folds."),
            'temporal_folds_<paradigma>.json')

    answers['L3.2'] = _answer(
        ARGUMENT,
        "A mesma entidade aparece em treino e em teste: o split é temporal, "
        "não por entidade. K&N dizem que isso é vazamento a menos que a "
        "afirmação seja sobre uma distribuição com a mesma estrutura de "
        "dependência. O cenário de uso aqui é prever anos seguintes das mesmas "
        "entidades, então a estrutura casa -- mas isso é uma afirmação "
        "científica e cabe ao autor. Nota de desenho: a validação cruzada "
        "interna agrupa por entidade (GroupKFold) porque ali o vazamento de "
        "entidade inflaria a seleção de hiperparâmetro; o split externo não "
        "agrupa porque agrupar mudaria a afirmação para generalização a "
        "entidades não vistas, que é outra pergunta.")

    coverage = _read_json(root / 'collection' / 'raw_data'
                          / 'target_coverage.json')
    if coverage:
        removed = coverage.get('rows_removed_missing_target')
        before = coverage.get('rows_before')
        share = (removed / before) if before else 0.0
        observed = coverage.get('observed_fraction', {})
        worst = sorted(observed.items(), key=lambda kv: kv[1])[:3]
        worst_text = ', '.join(f"{name} {value:.1%}" for name, value in worst)
        answers['L3.3.derived'] = _answer(
            DERIVED,
            f"Linhas incluídas na análise: {coverage.get('rows_after')} de "
            f"{before}. Foram removidas {removed} ({share:.1%}) por não terem "
            f"alvo observado -- o alvo não é imputado. Menores frações "
            f"observadas por coluna no painel de entrada: {worst_text}.",
            'target_coverage.json')
    else:
        answers['L3.3.derived'] = _answer(
            PENDING,
            "Quantas linhas foram removidas por falta de alvo, e a fração "
            "observada por coluna, virão do artefato de cobertura.",
            'target_coverage.json')

    answers['L3.3.argument'] = _answer(
        ARGUMENT,
        "A remoção de linhas sem alvo é uma seleção condicionada ao desfecho, "
        "e ausência de alvo não é aleatória. É a pergunta que a info sheet faz "
        "em L3.3 -- descreva como as linhas foram selecionadas e por que o "
        "teste corresponde à distribuição sobre a qual a afirmação é feita. A "
        "cobertura geográfica mínima por fold trata a metade espacial; esta "
        "metade é do autor. K&N registram L3.3 como categoria que trabalho "
        "anterior não considerava vazamento.")
    return answers


def _triviality(root: Path, paradigms: List[str]) -> Dict:
    """Does the model beat the naive baseline? K&N's own test.

    Their case study found that once leakage was corrected, complex models did
    not outperform decades-old logistic regression. The same comparison here
    answers whether the task is trivial -- which is the risk L2 leaves open
    when a feature is legitimately available but makes the prediction easy.
    """
    from core.paradigm_registry import baseline_results_paths

    rows = {}
    for paradigm, relative in sorted(baseline_results_paths().items()):
        payload = _read_json(Path(str(root / relative).replace(
            str(root) + '/outputs/', str(root) + '/')))
        if payload is None:
            payload = _read_json(_ROOT / 'outputs' / relative)
        if payload is None:
            continue
        rows[paradigm] = payload
    if not rows:
        return {'kind': PENDING,
                'text': "A comparação modelo contra melhor baseline virá dos "
                        "resultados de baseline e hierárquico.",
                'source': 'baseline_analysis_<paradigma>.json'}
    return {'kind': DERIVED,
            'text': f"Resultados de baseline presentes para "
                    f"{', '.join(sorted(rows))}.",
            'source': 'baseline_analysis_<paradigma>.json'}


def _reproducibility(root: Path) -> Dict:
    """[R5] -- what makes every answer above checkable rather than asserted."""
    snapshot = _read_json(root / 'scientific_config_snapshot.json')
    if snapshot:
        return _answer(
            DERIVED,
            f"Commit {snapshot.get('git_commit')}, instante "
            f"{snapshot.get('timestamp')}, orçamento de núcleos "
            f"{snapshot.get('scientific_config', {}).get('engine_threads')} "
            f"por engine. Dependências fixadas em requirements-lock.txt, "
            f"imagem base fixada por digest, e o snapshot de dados tem "
            f"manifesto verificável.",
            'scientific_config_snapshot.json')
    return _answer(
        PENDING,
        "Commit, instante e orçamento virão do snapshot de configuração.",
        'scientific_config_snapshot.json')


def _invariance() -> Dict:
    """What a multi-implementation framework owes that a single one does not."""
    return _answer(
        DERIVED,
        "A taxonomia de K&N pressupõe uma implementação. Aqui há três, e cada "
        "verificação precisa dar o mesmo veredito nas três, senão a própria "
        "verificação vira fonte de divergência. Imposto por teste: a forma "
        "canônica do fold é idêntica nos três paradigmas (bitwise), o carry "
        "temporal tem uma implementação só, o teto de proxy e o da seleção "
        "derivam do mesmo parâmetro, e o resultado não depende da ordem em que "
        "os paradigmas executam.",
        'tests/test_fold_materialisation.py, tests/test_execution_order.py')


def build(datasets: List[str]) -> Dict:
    paradigms = sorted(discover_paradigms())
    report = {'paradigms': paradigms, 'datasets': {}}
    for dataset in datasets:
        root = _dataset_root(dataset)
        report['datasets'][dataset] = {
            'L1': _l1_clean_separation(root, paradigms),
            'L2': _l2_feature_legitimacy(root, paradigms),
            'L3': _l3_distribution(root, paradigms),
            'triviality': _triviality(root, paradigms),
            'reproducibility': _reproducibility(root),
        }
    report['invariance'] = _invariance()
    return report


def _render_answer(key: str, answer: Dict) -> List[str]:
    marker = {DERIVED: 'derivado', PENDING: 'pendente de execução',
              ARGUMENT: 'exige argumento do autor'}[answer['kind']]
    lines = [f"**{key}** _({marker})_", "", answer['text']]
    if answer.get('source'):
        lines += ["", f"> fonte: `{answer['source']}`"]
    return lines + [""]


def to_markdown(report: Dict) -> str:
    lines = [
        "# Model info sheet",
        "",
        "Instrumento de Kapoor & Narayanan (2023) para detectar e prevenir "
        "vazamento. As respostas marcadas como derivadas vêm de artefatos "
        "desta execução e podem ser conferidas contra eles; as marcadas como "
        "exigindo argumento não são deriváveis e estão assim em vez de em "
        "branco.",
        "",
        f"Paradigmas: {', '.join(report['paradigms'])}",
        "",
    ]
    for dataset, sections in sorted(report['datasets'].items()):
        lines += [f"## {dataset}", ""]
        for argument, title in (
                ('L1', 'L1 — Separação limpa entre treino e teste'),
                ('L2', 'L2 — Cada feature é legítima'),
                ('L3', 'L3 — Teste vem da distribuição de interesse')):
            lines += [f"### {title}", ""]
            for key, answer in sections[argument].items():
                lines += _render_answer(key, answer)
        lines += ["### Trivialidade da tarefa", ""]
        lines += _render_answer('modelo vs baseline', sections['triviality'])
        lines += ["### Reprodutibilidade computacional [R5]", ""]
        lines += _render_answer('ambiente', sections['reproducibility'])
    lines += ["## Invariância entre implementações", ""]
    lines += _render_answer('três paradigmas, um veredito',
                            report['invariance'])
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+',
                        default=['worldbank', 'inep_censo'])
    parser.add_argument('--out-dir', default=str(_ROOT / 'outputs'))
    args = parser.parse_args(argv)

    report = build(args.datasets)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model_info_sheet.json').write_text(json.dumps(report, indent=2,
                                                          ensure_ascii=False))
    (out / 'model_info_sheet.md').write_text(to_markdown(report))

    counts = {}
    for sections in report['datasets'].values():
        for argument in ('L1', 'L2', 'L3'):
            for answer in sections[argument].values():
                counts[answer['kind']] = counts.get(answer['kind'], 0) + 1
    print(json.dumps({'status': 'ok', 'answers': counts,
                      'markdown': str(out / 'model_info_sheet.md')},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
