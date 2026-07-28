#!/usr/bin/env python3
"""How far the temporal dependence reaches, and therefore how wide the gap must be.

P2 puts a gap between the training window and the evaluation windows. The value
was 2 years and the README cited Roberts et al. (2017) for it, but that paper
gives the *criterion* rather than the number, and nothing in the repository
measured the quantity the criterion is about. This script measures it.

The criterion, as the authors themselves implement it in `blockCV`: blocks
should be substantially bigger than the range of autocorrelation *in the model
residual*, and a buffer the size of that range gives a good error estimate. The
qualifier matters here more than usual. This model reads lags of the target as
features, precisely to absorb temporal structure, so the raw series and what the
model fails to explain are different quantities and answer different questions:

    raw target, entity mean removed .... 0.56 at lag 2 (World Bank)
    out-of-sample residual ............. 0.03 at lag 2, within entity

Measuring the first and comparing it against the gap would have condemned a gap
that is in fact adequate.

There is a second decomposition, and it is the one worth reading twice. Two
thirds of the residual variance is a per-entity offset: the model is
persistently wrong for the same countries, every year. That component does not
decay with lag -- 0.78 at lag 1 and 0.59 at lag 5 -- and no temporal buffer
touches it. It is L3.2 in Kapoor & Narayanan's taxonomy, non-independence
between training and test rows, which this framework declares as requiring an
argument from the author rather than claiming to solve. The measurement puts a
number on that declaration instead of leaving it qualitative.

Usage:
    python scripts/validation/measure_dependence_range.py
    python scripts/validation/measure_dependence_range.py --results DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src'))

from core.scientific_config import SCIENTIFIC_CONFIG  # noqa: E402

#: Below this the dependence is treated as spent. Cohen's (1988) boundary for a
#: negligible correlation, the same convention the feature selection floor uses,
#: so the two thresholds in this framework do not come from different places.
NEGLIGIBLE = 0.10

#: Lags to probe. Beyond the sixth the overlap between a fold's test window and
#: its own shifted copy is too thin for the estimate to mean anything.
MAX_LAG = 6


def _residuals(results_path: Path, folds_path: Path) -> pd.DataFrame:
    """Out-of-sample residuals from a completed run, with their year restored.

    The prediction artifact stores rows in the order `canonical_fold` produced
    them -- sorted by (entity, year) -- and the fold configuration says which
    years the test window spans. That is enough to recover the year of every
    row without the artifact carrying it.
    """
    results = json.loads(results_path.read_text())
    windows = {f['fold_id']: (f['test_start'], f['test_end'])
               for f in json.loads(folds_path.read_text())['folds']}

    rows: List[Dict] = []
    for fold in results.get('folds', []):
        fold_id = fold.get('fold_id')
        test = (fold.get('models', {}).get('simple_hierarchical', {})
                .get('test', {}))
        truth = np.asarray(test.get('y_true', []), dtype=float)
        predicted = np.asarray(test.get('predictions', []), dtype=float)
        if not len(truth) or fold_id not in windows:
            continue

        start, end = windows[fold_id]
        years = list(range(start, end + 1))
        if len(truth) % len(years):
            raise ValueError(
                f"fold {fold_id}: {len(truth)} rows do not divide into "
                f"{len(years)} test years, so the year of each row cannot be "
                f"recovered from the ordering")

        for index, (actual, fitted) in enumerate(zip(truth, predicted)):
            rows.append({
                'fold': fold_id,
                'entity': index // len(years),
                'year': years[index % len(years)],
                'residual': float(actual - fitted),
            })
    return pd.DataFrame(rows)


def _autocorrelation(frame: pd.DataFrame, column: str, lag: int
                     ) -> Optional[float]:
    """Correlation of a column with itself `lag` years earlier, within entity."""
    left = frame[['entity', 'year', column]]
    right = (left.rename(columns={column: '_lagged'})
             .assign(year=lambda d: d['year'] + lag))
    paired = left.merge(right[['entity', 'year', '_lagged']],
                        on=['entity', 'year'])
    if len(paired) < 30:
        return None
    value = paired[column].corr(paired['_lagged'])
    return None if pd.isna(value) else float(value)


def measure(results_path: Path, folds_path: Path) -> Dict:
    frame = _residuals(results_path, folds_path)
    if frame.empty:
        raise ValueError(f"no out-of-sample residuals in {results_path}")

    # Split the residual into what stays with an entity and what moves with
    # time. Reading the total alone conflates a buffer question with a
    # non-independence question, and only the first has a buffer for an answer.
    entity_mean = frame.groupby('entity')['residual'].transform('mean')
    frame['within_entity'] = frame['residual'] - entity_mean

    total_variance = float(frame['residual'].var())
    within_variance = float(frame['within_entity'].var())
    entity_share = (1.0 - within_variance / total_variance
                    if total_variance > 0 else None)

    by_lag = {}
    for lag in range(1, MAX_LAG + 1):
        by_lag[lag] = {
            'total': _autocorrelation(frame, 'residual', lag),
            'within_entity': _autocorrelation(frame, 'within_entity', lag),
        }

    spent_at = next(
        (lag for lag in sorted(by_lag)
         if by_lag[lag]['within_entity'] is not None
         and abs(by_lag[lag]['within_entity']) < NEGLIGIBLE),
        None)

    return {
        'observations': int(len(frame)),
        'folds': int(frame['fold'].nunique()),
        'entities': int(frame['entity'].nunique()),
        'years': [int(frame['year'].min()), int(frame['year'].max())],
        'residual_variance': total_variance,
        'within_entity_variance': within_variance,
        # The part no temporal buffer can reach: K&N's L3.2.
        'entity_share_of_variance': entity_share,
        'autocorrelation_by_lag': by_lag,
        'negligible_threshold': NEGLIGIBLE,
        'dependence_spent_at_lag': spent_at,
        'configured_gap_years': int(SCIENTIFIC_CONFIG['temporal_gap_years']),
        'gap_covers_dependence': (spent_at is not None
                                  and spent_at <= int(
                                      SCIENTIFIC_CONFIG['temporal_gap_years'])),
    }


def _report(name: str, measured: Dict) -> None:
    print(f"\n{name}: {measured['observations']} out-of-sample residuals, "
          f"{measured['folds']} folds, {measured['entities']} entities, "
          f"{measured['years'][0]}-{measured['years'][1]}")
    share = measured['entity_share_of_variance']
    if share is not None:
        print(f"  {share:.0%} of the residual variance is a persistent "
              f"per-entity offset (K&N L3.2; no buffer reaches it)")
    print(f"  {'lag':>4} {'total':>8} {'within entity':>15}")
    for lag, values in sorted(measured['autocorrelation_by_lag'].items()):
        total = values['total']
        within = values['within_entity']
        if total is None and within is None:
            continue
        mark = ''
        if lag == measured['dependence_spent_at_lag']:
            mark = f"  <- below {measured['negligible_threshold']}"
        print(f"  {lag:>4} {total if total is None else f'{total:>8.3f}'} "
              f"{within if within is None else f'{within:>15.3f}'}{mark}")
    spent = measured['dependence_spent_at_lag']
    print(f"  dependence spent at lag "
          f"{spent if spent is not None else f'> {MAX_LAG}'}; "
          f"configured gap {measured['configured_gap_years']} -> "
          f"{'adequate' if measured['gap_covers_dependence'] else 'TOO NARROW'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results', default=None,
        help='directory of a completed run (default: outputs/<dataset>)')
    parser.add_argument('--json', action='store_true',
                        help='emit the measurement as JSON')
    args = parser.parse_args(argv)

    from core.config import get_absolute_output_path
    base = (Path(args.results) if args.results
            else Path(get_absolute_output_path('outputs')))

    architectures = base / 'ml_pipeline' / 'architectures'
    if not architectures.is_dir():
        print(f"No completed run under {base}. Run the pipeline first, or "
              f"pass --results with the directory of one.")
        return 2

    measurements = {}
    for prep in sorted(architectures.glob('*/prep')):
        paradigm = prep.parent.name
        folds = prep / f'temporal_folds_{paradigm}.json'
        results = next((prep.parent / 'models' / 'hierarchical_results')
                       .glob('hierarchical_analysis_*.json'), None)
        if not (folds.exists() and results):
            continue
        measurements[paradigm] = measure(results, folds)
        if not args.json:
            _report(paradigm, measurements[paradigm])

    if not measurements:
        print(f"No hierarchical results under {architectures}.")
        return 2

    if args.json:
        print(json.dumps(measurements, indent=2))
        return 0

    narrow = [name for name, m in measurements.items()
              if not m['gap_covers_dependence']]
    if narrow:
        print(f"\nThe configured gap does not cover the measured dependence "
              f"in: {', '.join(narrow)}")
        return 1
    print(f"\nThe configured gap covers the measured dependence in every "
          f"paradigm.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
