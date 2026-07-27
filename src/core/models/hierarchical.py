#!/usr/bin/env python3
"""Modelo hierárquico compartilhado pelos três paradigmas.

Estas funções eram três cópias, uma por paradigma. A extração é provavelmente
preservadora, e não uma aposta: comparados por AST com chamadas a print e
literais de nome normalizados, os três corpos são idênticos, e nenhum deles
lê qualquer atributo de self -- são funções puras que estavam escritas como
métodos.

A verificação é empírica além de estrutural. Antes da extração, os três
paradigmas produziam predições bitwise idênticas sobre a mesma entrada, para os
três valores de shrinkage; a mesma comparação roda depois, contra os mesmos
hashes.

O nome do paradigma entra como argumento porque é a única coisa que variava
entre as cópias, e serve apenas para rotular o resultado.
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
    Modelo hierárquico simples: médias por país + resíduos com Ridge regularizado.
    """
    # Read once: the return payload describes the grid even when the
    # residual branch below is not taken.
    _hm = SCIENTIFIC_CONFIG['hierarchical_model']
    global_mean = y_train.mean()
    n_countries = countries_train.nunique()
    total_samples = len(y_train)
    
    # Calcular médias por país com shrinkage adaptativo
    country_means = {}
    country_residuals_X = []
    country_residuals_y = []
    residual_groups = []
    country_sample_counts = {}
    
    print(f"Processamento hierárquico distribuído: {n_countries} países, {total_samples} amostras")
    
    for country in countries_train.unique():
        country_mask = countries_train == country
        country_y = y_train[country_mask]
        country_samples = len(country_y)
        country_sample_counts[country] = country_samples
        
        # Shrinkage tipo James-Stein: k=5 como prior strength (Efron & Morris, 1975)
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
        
        # Seleção de alpha via CV interna (Hoerl & Kennard, 1970)
        alphas = np.logspace(_hm['ridge_alpha_log10_start'],
                             _hm['ridge_alpha_log10_stop'],
                             _hm['ridge_alpha_count'])
        # RidgeCV rejects cv < 2; with fewer residual rows than that,
        # cv=None selects alpha by generalised cross-validation instead
        # of raising.
        # Partição da CV interna, deliberada em vez de acidental.
        #
        # cv=<int> faz o RidgeCV usar KFold sem shuffle, e como os resíduos são
        # concatenados por entidade os blocos contíguos eram blocos de entidade:
        # a seleção de alpha vinha fazendo leave-some-entities-out sem que
        # ninguém a tivesse escolhido, e mudaria em silêncio se a ordem de
        # concatenação mudasse.
        #
        # Declarado como GroupKFold pela entidade, o que preserva a partição e a
        # torna independente da ordem das linhas. Não é leakage em nenhuma das
        # duas formas -- todos os resíduos vêm da janela de treino.
        #
        # TimeSeriesSplit seria mais coerente com a tarefa, que é extrapolação
        # temporal e não generalização para entidades novas. Exigiria carregar o
        # ano através do _prepare_data de cada paradigma, que é específico de
        # engine; fica registrado como escolha de desenho em aberto, e não como
        # detalhe de implementação.
        n_residuals = len(residuals_X)
        n_groups = len(set(residual_groups))
        inner_folds = min(_hm['ridge_cv_folds'], n_groups)
        if inner_folds >= 2:
            splitter = GroupKFold(n_splits=inner_folds)
            cv = list(splitter.split(residuals_X, residuals_y,
                                     groups=residual_groups))
        else:
            # Menos de duas entidades: sem grupos para separar, o RidgeCV recai
            # na validação cruzada generalizada.
            cv = None
        ridge_cv = RidgeCV(alphas=alphas, cv=cv)
        ridge_cv.fit(residuals_X, residuals_y)
        final_alpha = ridge_cv.alpha_
        residual_model = ridge_cv

        print(f"      Simple hierarchical ({architecture}):")
        print(f"         {features_count} features x {samples_count} samples de resíduos")
        print(f"         alpha selecionado por RidgeCV: {final_alpha:.2f}")
        print(f"         Shrinkage aplicado em {n_countries} países")
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
        'regularization_applied': f'RidgeCV: alpha={final_alpha:.2f} (logspace 0.1-1000, cv interno)',
        'features_count': features_count,
        'regularization_details': {
            'ridgecv_alpha': float(final_alpha),
            'shrinkage_applied': True,
            'alpha_selection': (
                f"RidgeCV com logspace("
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
    print(f"   Imputacao por fold -> {path}")
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
    print(f"   Auditoria de features -> {path}")
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
    """Persiste os vetores de predição de teste dos baselines.

    Recebe o recorder em vez de lê-lo de self: a versão em cada paradigma era
    idêntica -- verificado por AST com o nome normalizado -- e a única dependência
    de estado era esse atributo.
    """
    path = predictions_path(architecture, 'baseline')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    written = recorder.write(path)
    if written:
        print(f"Prediction vectors written: {written}")
