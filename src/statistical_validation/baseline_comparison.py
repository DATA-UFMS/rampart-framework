#!/usr/bin/env python3
"""O modelo hierárquico bate a baseline ingênua?

É o método do estudo de caso de Kapoor & Narayanan (2023). Eles pegaram quatro
papers que afirmavam superioridade de ML complexo sobre regressão logística,
corrigiram o vazamento, e mediram de novo: sem os erros, os modelos complexos
não superavam a LR de décadas atrás em nenhum caso.

Esta é a mesma medida, e ela responde a pergunta que L2 deixa aberta. K&N
recusam subdividir L2 porque legitimidade exige julgamento de domínio, e
apontam duas maneiras de uma feature ser ilegítima: ser proxy do desfecho, e
tornar a predição trivial por já estar disponível no instante da predição. O
rastreio automático pega a primeira. Esta comparação é o que mede a segunda.

A diferença é informativa nos dois sentidos, e nenhum é bom sem qualificação:

  diferença ≈ 0   o ML não acrescenta nada sobre repetir o último valor
                  observado. O banco de provas continua válido para comparar
                  paradigmas, mas o conteúdo de aprendizado é decorativo.
  diferença alta  vale conferir se alguma feature está fazendo o trabalho
                  trivialmente. Qual baseline venceu diz muito: se a ingênua
                  com defasagem é a melhor, o alvo é sobretudo autocorrelato.

Lê os vetores de predição, não as métricas agregadas de cada paradigma. Três
razões: é uma fonte só com um esquema só, contra três layouts diferentes de
JSON de baseline; a métrica passa a ser calculada aqui, do mesmo jeito para os
dois estágios; e são exatamente os vetores sobre os quais a equivalência
bitwise é afirmada, então a comparação herda essa garantia.

Do que um framework de uma implementação não precisa: com Δ=0, os três
paradigmas predizem o mesmo, logo a diferença tem de ser idêntica nos três.
Divergência aqui é divergência na afirmação central, e sai reportada.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))

from core.config import get_absolute_output_path  # noqa: E402
from core.paradigm_registry import discover_paradigms  # noqa: E402
from core.prediction_store import predictions_path  # noqa: E402
from core.scientific_config import SCIENTIFIC_CONFIG  # noqa: E402
from statistical_validation.equivalence_estimation import (  # noqa: E402
    DEFAULT_SEED, bootstrap_ci)

RESULTS_DIR = get_absolute_output_path('statistics')

#: Abaixo disto os três paradigmas não estão predizendo o mesmo, e a
#: comparação deixa de ser entre engines. É a tolerância do Δ=0, não uma
#: escolha de modelagem: predições idênticas dão R2 idêntico.
PARADIGM_AGREEMENT_TOLERANCE = 1e-9


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """R2 fora da amostra. None quando o alvo não varia no fold."""
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return None
    true_values = y_true[finite]
    residual = ((true_values - y_pred[finite]) ** 2).sum()
    total = ((true_values - true_values.mean()) ** 2).sum()
    if total <= 0:
        return None
    return float(1.0 - residual / total)


def _stage_scores(paradigm: str, stage: str) -> Dict[int, Dict[str, float]]:
    """{fold: {modelo: R2}} para um estágio de um paradigma.

    O estágio vem do caminho e não do quadro: load_predictions concatena os
    dois arquivos e perde essa distinção, que é justamente a que separa a
    baseline do modelo.
    """
    path = predictions_path(paradigm, stage)
    if not os.path.exists(path):
        return {}

    frame = pd.read_parquet(path)
    scores: Dict[int, Dict[str, float]] = {}
    for (fold, model), group in frame.groupby(['fold', 'model']):
        value = _r_squared(group['y_true'].to_numpy(dtype=float),
                           group['y_pred'].to_numpy(dtype=float))
        if value is not None:
            scores.setdefault(int(fold), {})[str(model)] = value
    return scores


def compare(paradigm: str, bootstrap_iters: int) -> Optional[Dict]:
    """Modelo contra a melhor baseline, fold a fold."""
    baselines = _stage_scores(paradigm, 'baseline')
    models = _stage_scores(paradigm, 'hierarchical')
    shared = sorted(set(baselines) & set(models))
    if not shared:
        return None

    per_fold: List[Dict] = []
    for fold in shared:
        best_name, best_score = max(baselines[fold].items(),
                                    key=lambda pair: pair[1])
        # O melhor modelo do estágio hierárquico, pelo mesmo critério com que
        # a melhor baseline é escolhida -- comparar o melhor de um contra a
        # média do outro seria comparar coisas diferentes.
        model_name, model_score = max(models[fold].items(),
                                      key=lambda pair: pair[1])
        per_fold.append({
            'fold': fold,
            'best_baseline': best_name,
            'best_baseline_r2': best_score,
            'model': model_name,
            'model_r2': model_score,
            'gap': model_score - best_score,
        })

    gaps = np.array([row['gap'] for row in per_fold], dtype=float)
    point, (low, high), method = bootstrap_ci(
        gaps, iters=bootstrap_iters, seed=DEFAULT_SEED)

    winners: Dict[str, int] = {}
    for row in per_fold:
        winners[row['best_baseline']] = winners.get(row['best_baseline'], 0) + 1

    return {
        'paradigm': paradigm,
        'n_folds': len(per_fold),
        'per_fold': per_fold,
        'mean_gap': point,
        'gap_ci95': [low, high],
        'gap_ci95_method': method,
        # Um intervalo que cobre zero diz que o modelo não foi mostrado
        # superior à baseline -- que é o achado de K&N, não um defeito deste
        # pipeline.
        'beats_baseline': bool(low > 0.0),
        'baseline_wins': winners,
        'folds_where_baseline_wins': int((gaps < 0).sum()),
    }


def _agreement(results: Dict[str, Dict]) -> Dict:
    """Com Δ=0 a diferença é a mesma nos três; se não for, o Δ=0 não vale."""
    means = {paradigm: entry['mean_gap']
             for paradigm, entry in results.items()
             if entry and np.isfinite(entry['mean_gap'])}
    if len(means) < 2:
        return {'checked': False,
                'reason': 'menos de dois paradigmas com resultado'}
    spread = max(means.values()) - min(means.values())
    return {
        'checked': True,
        'max_absolute_difference': float(spread),
        'tolerance': PARADIGM_AGREEMENT_TOLERANCE,
        'consistent': bool(spread <= PARADIGM_AGREEMENT_TOLERANCE),
        'mean_gap_by_paradigm': means,
    }


def analyze(bootstrap_iters: Optional[int] = None) -> Dict:
    iterations = int(bootstrap_iters if bootstrap_iters is not None
                     else SCIENTIFIC_CONFIG['bootstrap_iters'])
    results = {}
    for paradigm in sorted(discover_paradigms()):
        outcome = compare(paradigm, iterations)
        if outcome is not None:
            results[paradigm] = outcome
    return {'by_paradigm': results,
            'cross_paradigm_agreement': _agreement(results),
            'bootstrap_iters': iterations,
            'metric': 'r2_out_of_sample'}


def to_latex(report: Dict) -> str:
    def escape(text) -> str:
        return str(text).replace('_', r'\_').replace('%', r'\%')

    lines = [
        '% Modelo hierárquico contra a melhor baseline por fold',
        '\\begin{table}[htb]',
        '\\centering',
        '\\caption{Diferença de $R^2$ fora da amostra entre o modelo '
        'hierárquico e a melhor baseline por fold. Um intervalo que cobre '
        'zero indica que a superioridade do modelo não foi estabelecida.}',
        '\\begin{tabular}{lrrrl}',
        '\\toprule',
        'Paradigma & Folds & $\\Delta R^2$ & IC95 & Baseline vencedora \\\\',
        '\\midrule',
    ]
    for paradigm, entry in sorted(report['by_paradigm'].items()):
        winners = ', '.join(
            f"{escape(name)} ({count})"
            for name, count in sorted(entry['baseline_wins'].items(),
                                      key=lambda pair: -pair[1]))
        low, high = entry['gap_ci95']
        lines.append(
            f"{escape(paradigm)} & {entry['n_folds']} & "
            f"{entry['mean_gap']:.4f} & "
            f"[{low:.4f}, {high:.4f}] & {winners} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)


def write_outputs(report: Dict) -> Tuple[str, str]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, 'baseline_comparison.json')
    tex_path = os.path.join(RESULTS_DIR, 'baseline_comparison.tex')
    with open(json_path, 'w') as handle:
        json.dump(report, handle, indent=2)
    with open(tex_path, 'w') as handle:
        handle.write(to_latex(report) + '\n')
    return json_path, tex_path


def main() -> int:
    report = analyze()
    if not report['by_paradigm']:
        print('  Nenhum par de predicoes baseline/hierarquico; nada a comparar.')
        return 0

    json_path, tex_path = write_outputs(report)
    agreement = report['cross_paradigm_agreement']
    if agreement.get('checked') and not agreement['consistent']:
        raise ValueError(
            f"Os paradigmas discordam sobre a diferença contra a baseline "
            f"({agreement['mean_gap_by_paradigm']}). Com predições idênticas "
            f"o R2 é idêntico, então isto contradiz a equivalência bitwise."
        )

    print(json.dumps({'status': 'ok', 'json': json_path, 'tex': tex_path,
                      'paradigms': sorted(report['by_paradigm'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
