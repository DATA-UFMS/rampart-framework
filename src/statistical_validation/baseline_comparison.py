#!/usr/bin/env python3
"""Does the hierarchical model beat the naive baseline?

It is the method of the Kapoor & Narayanan (2023) case study. They took four
papers claiming the superiority of complex ML over logistic regression,
corrected the leakage, and measured again: without the errors, the complex
models did not beat the decades-old LR in any case.

This is the same measurement, and it answers the question L2 leaves open. K&N
refuse to subdivide L2 because legitimacy requires domain judgement, and they
point out two ways for a feature to be illegitimate: being a proxy of the
outcome, and making the prediction trivial by already being available at the
instant of prediction. The automatic screening catches the first. This
comparison is what measures the second.

The difference is informative in both directions, and neither is good without
qualification:

  difference ≈ 0   ML adds nothing over repeating the last observed value. The
                   testbed remains valid for comparing paradigms, but the
                   learning content is decorative.
  large difference it is worth checking whether some feature is doing the work
                   trivially. Which baseline won says a lot: if the naive one
                   with a lag is the best, the target is above all
                   autocorrelated.

Reads the prediction vectors, not each paradigm's aggregate metrics. Three
reasons: it is a single source with a single schema, against three different
layouts of baseline JSON; the metric comes to be computed here, in the same way
for both stages; and they are exactly the vectors over which bitwise
equivalence is asserted, so the comparison inherits that guarantee.

What a single-implementation framework does not need: with Δ=0, the three
paradigms predict the same, hence the difference has to be identical in all
three. Divergence here is divergence in the central claim, and gets reported.
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

#: Below this the three paradigms are not predicting the same, and the
#: comparison stops being between engines. It is the tolerance of Δ=0, not a
#: modelling choice: identical predictions give identical R2.
PARADIGM_AGREEMENT_TOLERANCE = 1e-9


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Out-of-sample R2. None when the target does not vary within the fold."""
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
    """{fold: {model: R2}} for one stage of one paradigm.

    The stage comes from the path and not from the frame: load_predictions
    concatenates the two files and loses that distinction, which is exactly the
    one that separates the baseline from the model.
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
    """Model against the best baseline, fold by fold."""
    baselines = _stage_scores(paradigm, 'baseline')
    models = _stage_scores(paradigm, 'hierarchical')
    shared = sorted(set(baselines) & set(models))
    if not shared:
        return None

    per_fold: List[Dict] = []
    for fold in shared:
        best_name, best_score = max(baselines[fold].items(),
                                    key=lambda pair: pair[1])
        # The best model of the hierarchical stage, by the same criterion with
        # which the best baseline is chosen -- comparing the best of one against
        # the mean of the other would be comparing different things.
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
        # An interval that covers zero says the model has not been shown
        # superior to the baseline -- which is K&N's finding, not a defect of
        # this pipeline.
        'beats_baseline': bool(low > 0.0),
        'baseline_wins': winners,
        'folds_where_baseline_wins': int((gaps < 0).sum()),
    }


def _agreement(results: Dict[str, Dict]) -> Dict:
    """With Δ=0 the difference is the same in all three; if it is not, Δ=0 does not hold."""
    means = {paradigm: entry['mean_gap']
             for paradigm, entry in results.items()
             if entry and np.isfinite(entry['mean_gap'])}
    if len(means) < 2:
        return {'checked': False,
                'reason': 'fewer than two paradigms with a result'}
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
        '% Hierarchical model against the best baseline per fold',
        '\\begin{table}[htb]',
        '\\centering',
        '\\caption{Out-of-sample $R^2$ difference between the hierarchical '
        'model and the best baseline per fold. An interval that covers '
        'zero indicates that the model\'s superiority has not been established.}',
        '\\begin{tabular}{lrrrl}',
        '\\toprule',
        'Paradigm & Folds & $\\Delta R^2$ & 95\\% CI & Winning baseline \\\\',
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
        print('  No baseline/hierarchical prediction pair; nothing to compare.')
        return 0

    json_path, tex_path = write_outputs(report)
    agreement = report['cross_paradigm_agreement']
    if agreement.get('checked') and not agreement['consistent']:
        raise ValueError(
            f"The paradigms disagree about the difference against the baseline "
            f"({agreement['mean_gap_by_paradigm']}). With identical predictions "
            f"the R2 is identical, so this contradicts bitwise equivalence."
        )

    print(json.dumps({'status': 'ok', 'json': json_path, 'tex': tex_path,
                      'paradigms': sorted(report['by_paradigm'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
