#!/usr/bin/env python3
"""Cross-paradigm prediction equivalence.

Asserts that every paradigm predicted the same values for the same rows. This is
the condition under which a latency comparison between paradigms is meaningful:
without it, the paradigms may be timed while doing different work.

Three failures are distinguished, because they have different causes:

  disjoint      one paradigm has no vector where another does, so there is
                nothing to compare
  misaligned    the paradigms evaluated different rows, visible in the entity
                sequence or the observed targets
  divergent     the paradigms evaluated the same rows and predicted differently

Comparison is bitwise. float_precision_tolerance is reported alongside a
divergence to characterise its magnitude, not to excuse it: a difference within
tolerance is still a difference, and the framework asserts equality.

Exit status is non-zero on any violation, so the pipeline halts before the
benchmark rather than timing paradigms that are not doing the same work.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import get_absolute_output_path
from core.paradigm_registry import discover_paradigms
from core.prediction_store import load_predictions
from core.scientific_config import SCIENTIFIC_CONFIG

REPORT_PATH = get_absolute_output_path(
    'outputs/statistics/prediction_equivalence.json')


def _vectors(frame: pd.DataFrame) -> Dict[Tuple[Any, str], pd.DataFrame]:
    """Index a paradigm's predictions by (fold, model), ordered by row."""
    grouped: Dict[Tuple[Any, str], pd.DataFrame] = {}
    for (fold, model), rows in frame.groupby(['fold', 'model'], sort=True):
        grouped[(fold, model)] = rows.sort_values('row').reset_index(drop=True)
    return grouped


def _compare_vector(
    left: pd.DataFrame, right: pd.DataFrame, tolerance: float
) -> Optional[Dict[str, Any]]:
    """Compare one (fold, model) vector pair. None when equivalent."""
    if len(left) != len(right):
        return {
            'kind': 'misaligned',
            'reason': 'row count differs',
            'left_rows': int(len(left)),
            'right_rows': int(len(right)),
        }

    left_entities = left['entity'].tolist()
    right_entities = right['entity'].tolist()
    if any(e is not None for e in left_entities) and left_entities != right_entities:
        mismatches = [
            i for i, (a, b) in enumerate(zip(left_entities, right_entities)) if a != b
        ]
        return {
            'kind': 'misaligned',
            'reason': 'entity sequence differs',
            'first_mismatch_row': int(mismatches[0]),
            'mismatching_rows': int(len(mismatches)),
        }

    observed_left = left['y_true'].to_numpy()
    observed_right = right['y_true'].to_numpy()
    if not np.array_equal(observed_left, observed_right):
        differing = int(np.count_nonzero(observed_left != observed_right))
        return {
            'kind': 'misaligned',
            'reason': 'observed targets differ, so the evaluated rows differ',
            'differing_rows': differing,
            'max_abs_difference': float(
                np.nanmax(np.abs(observed_left - observed_right))
            ),
        }

    predicted_left = left['y_pred'].to_numpy()
    predicted_right = right['y_pred'].to_numpy()
    if np.array_equal(predicted_left, predicted_right):
        return None

    difference = np.abs(predicted_left - predicted_right)
    return {
        'kind': 'divergent',
        'reason': 'predictions are not bitwise identical',
        'differing_rows': int(np.count_nonzero(predicted_left != predicted_right)),
        'max_abs_difference': float(np.nanmax(difference)),
        'within_float_tolerance': bool(np.nanmax(difference) <= tolerance),
    }


def verify(tolerance: Optional[float] = None) -> Dict[str, Any]:
    """Compare every pair of paradigms and return a report."""
    if tolerance is None:
        tolerance = float(SCIENTIFIC_CONFIG.get('float_precision_tolerance', 1e-9))

    paradigms = sorted(discover_paradigms())
    loaded: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for paradigm in paradigms:
        frame = load_predictions(paradigm)
        if frame is None or frame.empty:
            missing.append(paradigm)
        else:
            loaded[paradigm] = frame

    report: Dict[str, Any] = {
        'method': 'bitwise_prediction_equivalence',
        'float_precision_tolerance': tolerance,
        'paradigms_discovered': paradigms,
        'paradigms_with_predictions': sorted(loaded),
        'paradigms_without_predictions': missing,
        'comparisons': [],
        'violations': [],
    }

    # Every registered paradigm must have written its vectors. The claim is
    # that ALL THREE predict the same, and a missing paradigm never enters a
    # pair combination, hence never generates a violation: a single missing
    # vector produced a 'disjoint' violation and exit 1, while missing every
    # vector of a paradigm passed as 'equivalent'.
    if missing:
        report['status'] = 'insufficient_data'
        report['detail'] = (
            f"paradigms without prediction vectors: {', '.join(missing)}. "
            f"Equivalence is asserted over every registered paradigm, so it "
            f"cannot be established while one of them produced nothing."
        )
        return report

    if len(loaded) < 2:
        report['status'] = 'insufficient_data'
        report['detail'] = (
            'fewer than two paradigms produced prediction vectors, so '
            'equivalence cannot be asserted'
        )
        return report

    indexed = {name: _vectors(frame) for name, frame in loaded.items()}

    for left_name, right_name in itertools.combinations(sorted(loaded), 2):
        left, right = indexed[left_name], indexed[right_name]
        shared = sorted(set(left) & set(right), key=lambda k: (str(k[0]), k[1]))
        only_left = sorted(set(left) - set(right), key=lambda k: (str(k[0]), k[1]))
        only_right = sorted(set(right) - set(left), key=lambda k: (str(k[0]), k[1]))

        comparison = {
            'pair': f'{left_name}_vs_{right_name}',
            'vectors_compared': len(shared),
            'equivalent_vectors': 0,
        }

        for key in only_left + only_right:
            report['violations'].append({
                'pair': comparison['pair'],
                'fold': _plain(key[0]),
                'model': key[1],
                'kind': 'disjoint',
                'reason': (
                    f"vector present only in "
                    f"{left_name if key in left else right_name}"
                ),
            })

        for key in shared:
            verdict = _compare_vector(left[key], right[key], tolerance)
            if verdict is None:
                comparison['equivalent_vectors'] += 1
            else:
                verdict.update({
                    'pair': comparison['pair'],
                    'fold': _plain(key[0]),
                    'model': key[1],
                })
                report['violations'].append(verdict)

        report['comparisons'].append(comparison)

    report['status'] = 'equivalent' if not report['violations'] else 'violation'
    report['vectors_compared'] = sum(
        c['vectors_compared'] for c in report['comparisons']
    )
    return report


def _plain(value: Any) -> Any:
    """Convert numpy scalars so the report serialises as JSON."""
    return value.item() if hasattr(value, 'item') else value


def _describe(report: Dict[str, Any]) -> None:
    print("Cross-paradigm prediction equivalence")
    print(f"  paradigms: {', '.join(report['paradigms_with_predictions']) or 'none'}")
    if report['paradigms_without_predictions']:
        print(f"  without predictions: "
              f"{', '.join(report['paradigms_without_predictions'])}")

    for comparison in report['comparisons']:
        print(f"  {comparison['pair']}: "
              f"{comparison['equivalent_vectors']}/{comparison['vectors_compared']} "
              f"vectors identical")

    if report['status'] == 'equivalent':
        print(f"\nResult: {report['vectors_compared']} vectors bitwise identical")
    elif report['status'] == 'insufficient_data':
        print(f"\nResult: {report['detail']}")
    else:
        print(f"\nResult: {len(report['violations'])} violation(s)")
        for violation in report['violations'][:10]:
            print(f"  [{violation['kind']}] {violation['pair']} "
                  f"fold={violation['fold']} model={violation['model']}: "
                  f"{violation['reason']}")


def run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Assert paradigms predicted the same values for the same rows'
    )
    parser.add_argument(
        '--allow-missing', action='store_true',
        help='exit zero when fewer than two paradigms produced predictions',
    )
    args = parser.parse_args(argv)

    report = verify()
    _describe(report)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nReport: {REPORT_PATH}")

    if report['status'] == 'equivalent':
        return 0
    if report['status'] == 'insufficient_data':
        return 0 if args.allow_missing else 1
    return 1


if __name__ == '__main__':
    sys.exit(run())
