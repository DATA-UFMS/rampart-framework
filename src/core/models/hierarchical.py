#!/usr/bin/env python3
"""Hierarchical model shared by the three paradigms.

These functions were three copies, one per paradigm. The extraction is probably
behaviour-preserving, and not a bet: compared by AST with print calls and name
literals normalised, the three bodies are identical, and none of them reads any
attribute of self -- they are pure functions that had been written as
methods.

The verification is empirical as well as structural. Before the extraction, the
three paradigms produced bitwise identical predictions on the same input, for
the three shrinkage values; the same comparison runs afterwards, against the
same hashes.

The paradigm name comes in as an argument because it is the only thing that
varied between the copies, and it serves only to label the result.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.prediction_store import PredictionRecorder, predictions_path
from core.scientific_config import SCIENTIFIC_CONFIG


def simple_hierarchical_model(X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series,
                              countries_train: pd.Series,
                              countries_test: pd.Series,
                              residual_shrinkage: float = 0.8,
                              *, architecture: str) -> Dict:
    """
    Simple hierarchical model: per-country means + residuals with regularised Ridge.
    """
    # Read once: the return payload describes the grid even when the
    # residual branch below is not taken.
    _hm = SCIENTIFIC_CONFIG['hierarchical_model']
    global_mean = y_train.mean()
    n_countries = countries_train.nunique()
    total_samples = len(y_train)
    
    # Compute per-country means with adaptive shrinkage
    country_means = {}
    country_residuals_X = []
    country_residuals_y = []
    residual_groups = []
    country_sample_counts = {}
    
    print(f"Distributed hierarchical processing: {n_countries} countries, {total_samples} samples")
    
    for country in countries_train.unique():
        country_mask = countries_train == country
        country_y = y_train[country_mask]
        country_samples = len(country_y)
        country_sample_counts[country] = country_samples
        
        # James-Stein type shrinkage: k=5 as prior strength (Efron & Morris, 1975)
        shrinkage_factor = country_samples / (country_samples + 5.0)
        raw_country_mean = country_y.mean()
        country_mean_shrunk = (shrinkage_factor * raw_country_mean + 
                             (1 - shrinkage_factor) * global_mean)
        country_means[country] = country_mean_shrunk
        
        country_X = X_train[country_mask]
        country_residuals = country_y - country_mean_shrunk

        country_residuals_X.append(country_X)
        country_residuals_y.extend(country_residuals)
        residual_groups.extend([country] * country_samples)

    if len(country_residuals_X) > 0:
        residuals_X = pd.concat(country_residuals_X, ignore_index=True)
        residuals_y = np.array(country_residuals_y)
        
        features_count = residuals_X.shape[1]
        samples_count = len(residuals_y)
        
        # Alpha selection via inner CV (Hoerl & Kennard, 1970)
        alphas = np.logspace(_hm['ridge_alpha_log10_start'],
                             _hm['ridge_alpha_log10_stop'],
                             _hm['ridge_alpha_count'])
        # RidgeCV rejects cv < 2; with fewer residual rows than that,
        # cv=None selects alpha by generalised cross-validation instead
        # of raising.
        # The inner CV partition, deliberate rather than accidental.
        #
        # cv=<int> makes RidgeCV use KFold without shuffle, and since the
        # residuals are concatenated by entity the contiguous blocks were entity
        # blocks: alpha selection had been doing leave-some-entities-out without
        # anyone having chosen it, and would change silently if the
        # concatenation order changed.
        #
        # Declared as GroupKFold by entity, which preserves the partition and
        # makes it independent of row order. It is not leakage in either of the
        # two forms -- every residual comes from the training window.
        #
        # TimeSeriesSplit would be more coherent with the task, which is
        # temporal extrapolation and not generalisation to new entities. It
        # would require carrying the year through each paradigm's _prepare_data,
        # which is engine specific; it is recorded as an open design choice, and
        # not as an implementation detail.
        n_residuals = len(residuals_X)
        n_groups = len(set(residual_groups))
        inner_folds = min(_hm['ridge_cv_folds'], n_groups)
        if inner_folds >= 2:
            splitter = GroupKFold(n_splits=inner_folds)
            cv = list(splitter.split(residuals_X, residuals_y,
                                     groups=residual_groups))
        else:
            # Fewer than two entities: with no groups to separate, RidgeCV falls
            # back to generalised cross-validation.
            cv = None
        ridge_cv = RidgeCV(alphas=alphas, cv=cv)
        ridge_cv.fit(residuals_X, residuals_y)
        final_alpha = ridge_cv.alpha_
        residual_model = ridge_cv

        print(f"      Simple hierarchical ({architecture}):")
        print(f"         {features_count} features x {samples_count} residual samples")
        print(f"         alpha selected by RidgeCV: {final_alpha:.2f}")
        print(f"         Shrinkage applied to {n_countries} countries")
    else:
        residual_model = None
        features_count = 0
        samples_count = 0
        final_alpha = 0.0
    
    predictions = []
    for idx, (_, row) in enumerate(X_test.iterrows()):
        country = countries_test.iloc[idx]

        if country in country_means:
            base_pred = country_means[country]
        else:
            base_pred = global_mean

        if residual_model is not None:
            row_features = row.values.reshape(1, -1)
            residual_pred = residual_model.predict(row_features)[0]
            final_pred = base_pred + (residual_shrinkage * residual_pred)
        else:
            final_pred = base_pred

        predictions.append(final_pred)

    predictions = np.array(predictions)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        'model_name': 'simple_hierarchical',
        'architecture': architecture,
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'predictions': predictions.tolist(),
        'y_true': y_test.tolist(),
        'entities': [str(c) for c in countries_test],
        'country_effects': {str(k): float(v) for k, v in country_means.items()},
        'country_sample_counts': {str(k): int(v) for k, v in country_sample_counts.items()},
        'regularization_applied': f'RidgeCV: alpha={final_alpha:.2f} (logspace 0.1-1000, inner cv)',
        'features_count': features_count,
        'regularization_details': {
            'ridgecv_alpha': float(final_alpha),
            'shrinkage_applied': True,
            'alpha_selection': (
                f"RidgeCV with logspace("
                f"{_hm['ridge_alpha_log10_start']}, "
                f"{_hm['ridge_alpha_log10_stop']}, "
                f"{_hm['ridge_alpha_count']})"
            ),
            'residual_shrinkage': float(residual_shrinkage)
        }
    }


def write_imputation_report(reports, *, architecture: str) -> str:
    """Persist the fold-level imputation reports next to the fold artifacts.

    The reports were produced on every fold and discarded. How much of each
    training and evaluation window is fabricated appeared in no artifact --
    only the collection-stage imputation did, and that is the part bounded by
    the carry limit. The fold-scoped fill is the unbounded one: every cell the
    carry did not reach gets the training-window median.
    """
    import json
    import os
    from datetime import datetime

    from core.config import get_absolute_output_path

    directory = get_absolute_output_path(
        f'ml_pipeline/architectures/{architecture}/prep')
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory,
                        f'fold_imputation_{architecture}.json')

    per_fold = {str(fold_id): report for fold_id, report in reports}
    totals = {}
    for _, report in reports:
        for split, entry in report.get('filled_cells', {}).items():
            bucket = totals.setdefault(split, {'rows': 0, 'total': 0})
            bucket['rows'] += entry['rows']
            bucket['total'] += entry['total']
    for bucket in totals.values():
        bucket['fraction'] = (bucket['total'] / bucket['rows']
                              if bucket['rows'] else 0.0)

    with open(path, 'w') as handle:
        json.dump({'architecture': architecture,
                   'creation_timestamp': datetime.now().isoformat(),
                   'run_id': os.environ.get('RAMPART_RUN_ID'),
                   'folds': per_fold,
                   'across_folds': totals}, handle, indent=2)
    print(f"   Fold-level imputation -> {path}")
    return path



def write_feature_audit(reports, *, architecture: str) -> str:
    """Persist the P3 audit of the matrix each fold's model trains on.

    The audit ran and raised when it had to, but its report was assigned to an
    attribute nothing read. What it holds is the evidence behind the L2 screen:
    the measured association of every feature with the target, which
    autoregressive exemptions were granted, how much of the target the set
    reconstructs, and whether the design matrix has the rank its feature count
    implies. A screen whose findings are discarded is a claim without a record.

    Per fold, and shaped like the imputation report beside it, because the two
    are the same kind of thing: the receipts of the protocols that need the
    materialised fold and therefore cannot live in the base class.

    `checks_across_folds` is what the gate reads. A check that came out
    indeterminate in any fold -- too few complete rows for the reconstruction to
    be determined, say -- must not be summarised as one that passed.
    """
    import json
    import os
    from datetime import datetime

    from core.config import get_absolute_output_path

    directory = get_absolute_output_path(
        f'ml_pipeline/architectures/{architecture}/prep')
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'feature_audit_{architecture}.json')

    per_fold = {str(fold_id): report for fold_id, report in reports}
    across = {}
    for _, report in reports:
        for check, outcome in report.get('checks', {}).items():
            seen = across.setdefault(check, set())
            seen.add(outcome)
    # Worst outcome wins: one indeterminate fold makes the check indeterminate.
    summary = {check: ('indeterminate' if 'indeterminate' in outcomes
                       else ('ran' if 'ran' in outcomes else 'not_applicable'))
               for check, outcomes in across.items()}

    with open(path, 'w') as handle:
        json.dump({'architecture': architecture,
                   'creation_timestamp': datetime.now().isoformat(),
                   'run_id': os.environ.get('RAMPART_RUN_ID'),
                   'folds': per_fold,
                   'checks_across_folds': summary}, handle, indent=2)
    print(f"   Feature audit -> {path}")
    return path


def write_prediction_artifact(all_results: Dict, *, architecture: str) -> None:
    """Persist the test prediction vectors of every fold and model.

    Cross-paradigm equivalence is asserted over these vectors; the aggregate
    metrics stored alongside them cannot establish it.
    """
    recorder = PredictionRecorder(architecture)
    for fold in all_results.get('folds', []):
        fold_id = fold.get('fold_id')
        for model_name, splits in fold.get('models', {}).items():
            evaluation = splits.get('test', {})
            if 'predictions' not in evaluation or 'y_true' not in evaluation:
                continue
            recorder.record(
                fold=fold_id,
                model=model_name,
                y_true=evaluation['y_true'],
                y_pred=evaluation['predictions'],
                entities=evaluation.get('entities'),
            )

    path = predictions_path(architecture, 'hierarchical')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    written = recorder.write(path)
    if written:
        print(f"Prediction vectors written: {written}")


def write_baseline_predictions(recorder, *, architecture: str) -> None:
    """Persist the test prediction vectors of the baselines.

    Takes the recorder rather than reading it from self: the version in each
    paradigm was identical -- verified by AST with the name normalised -- and the
    only state dependency was that attribute.
    """
    path = predictions_path(architecture, 'baseline')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    written = recorder.write(path)
    if written:
        print(f"Prediction vectors written: {written}")
