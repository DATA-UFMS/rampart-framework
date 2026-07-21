#!/usr/bin/env python3
"""
Modelo Hierárquico para Arquitetura Polars DataFrame.

Implementa modelos hierárquicos para arquitetura Polars DataFrame com leitura lazy
e processamento eficiente de memória para predição de dropout escolar.

Otimizado com computação seletiva para minimizar materializações desnecessárias.
"""

import time

import pandas as pd
import polars as pl
import numpy as np
import json
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.*')

core_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core')
core_path = os.path.abspath(core_path)
if core_path not in sys.path:
    sys.path.append(core_path)

from config import get_absolute_output_path

project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)
from core.validation import audit_feature_set
from core.models.hierarchical import (
    simple_hierarchical_model as shared_simple_hierarchical_model,
    write_prediction_artifact as shared_write_prediction_artifact)
from core.validation import impute_from_training_window
from core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED, setup_reproducibility

setup_reproducibility()


class HierarchicalModelDataFrameLib:
    """
    Modelo Hierárquico para Arquitetura Polars DataFrame.

    Implementa modelos hierárquicos com leitura lazy Polars e processamento
    eficiente de memória.
    """

    def __init__(self):
        """
        Inicializa modelo hierárquico Polars DataFrame.
        """
        print("Inicializando Modelo Hierárquico Polars")

        self.target_col = 'dropout_rate_dataframe_lib'

        self._setup_normal_mode()

        self.results_path = get_absolute_output_path("ml_pipeline/architectures/dataframe_lib/models/hierarchical_results")
        os.makedirs(self.results_path, exist_ok=True)

        self._load_normal_summary()

    def _setup_normal_mode(self):
        """
        Setup para modo normal.
        """
        self.data_path = get_absolute_output_path("ml_pipeline/architectures/dataframe_lib/prep/master_data_dataframe_lib.parquet")
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/dataframe_lib/prep/temporal_folds_dataframe_lib.json")

        if not os.path.exists(self.data_path) or not os.path.exists(self.folds_path):
            raise FileNotFoundError("Dados Polars DataFrame não encontrados")

        print("   Carregando dados com LAZY EVALUATION...")
        self.df_lazy = pl.scan_parquet(self.data_path)
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']

    def _load_normal_summary(self):
        """
        Carregar resumo dos dados normais.
        """
        stats_df = self.df_lazy.select([
            pl.col('year').min().alias('year_min'),
            pl.col('year').max().alias('year_max'),
            pl.col('country_code').n_unique().alias('n_countries'),
            pl.len().alias('n_records')
        ]).collect()

        stats = stats_df.to_dicts()[0]

        n_records = stats['n_records']
        n_cols = len(self.df_lazy.collect_schema().names())

        print(f"Dados carregados: {n_records} registros, {n_cols} variáveis, "
              f"{stats['n_countries']} países")
        print(f"Período: {stats['year_min']}-{stats['year_max']}")
        print(f"Target: {self.target_col}")
        print(f"Folds: {len(self.folds)}")

        if self.target_col not in self.df_lazy.collect_schema().names():
            raise ValueError(f"Target {self.target_col} não encontrado nos dados")

        selection_path = get_absolute_output_path("ml_pipeline/architectures/dataframe_lib/prep/feature_selection_dataframe_lib.json")
        if os.path.exists(selection_path):
            try:
                with open(selection_path, 'r') as f:
                    selection_data = json.load(f)
                selected = selection_data.get('selected_features', [])

                # Incluir lag do target se existir
                schema_names = self.df_lazy.collect_schema().names()
                if 'dropout_rate_lag_2' in schema_names and 'dropout_rate_lag_2' not in selected:
                    selected.append('dropout_rate_lag_2')
                if 'dropout_rate_lag_3' in schema_names and 'dropout_rate_lag_3' not in selected:
                    selected.append('dropout_rate_lag_3')

                # Filtrar por existência
                existing = [c for c in selected if c in schema_names]
                self.available_features = existing
                print(f"Features disponíveis (seleção científica): {len(self.available_features)}")
                # The lags above bypassed run_feature_selection, so the
                # set the models train on is audited here.
                self.feature_audit = audit_feature_set(
                    self.df_lazy, existing, self.target_col, SCIENTIFIC_CONFIG)
            except Exception as e:
                print(f"Falha ao carregar seleção de features: {e}")
                raise
        else:
            raise FileNotFoundError(f"Seleção de features não encontrada: {selection_path}. Execute setup.py antes.")

    def _prepare_data(self, data_lazy):
        """
        Materializar um fold com computação seletiva.

        Materialização apenas. Toda estatística -- a mediana que preenche
        ausentes -- vive em core.validation.impute_from_training_window, ajustada
        na janela de treino do fold. Três implementações de uma estatística são
        três chances de os paradigmas calcularem coisas diferentes, e a afirmação
        de equivalência assume que eles diferem apenas em como movem dados.

        Sem parâmetro de referência: materializar não precisa da janela de
        treino, só ajustar estatística precisa.
        """
        data_filtered = data_lazy.filter(pl.col(self.target_col).is_not_null()).sort(['country_code', 'year'])

        X_lazy = data_filtered.select(self.available_features)

        # Materializar para operações pandas
        X_df = X_lazy.collect().to_pandas()
        y_series = data_filtered.select(pl.col(self.target_col)).collect().to_pandas().iloc[:, 0]
        countries_series = data_filtered.select(pl.col('country_code')).collect().to_pandas().iloc[:, 0]

        return X_df, y_series, countries_series

    def simple_hierarchical_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 countries_train: pd.Series, countries_test: pd.Series,
                                 residual_shrinkage: float = 0.8) -> Dict:
        """Delega à implementação compartilhada (core.models.hierarchical).

        Os três paradigmas computavam isto de forma idêntica -- verificado por
        AST e por igualdade bitwise das predições sobre a mesma entrada.
        """
        return shared_simple_hierarchical_model(
            X_train, y_train, X_test, y_test, countries_train, countries_test,
            residual_shrinkage=residual_shrinkage, architecture='dataframe_lib')

    def random_forest_hierarchical(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 countries_train: pd.Series, countries_test: pd.Series,
                                 max_depth: int = 6, min_samples_leaf: int = 8) -> Dict:
        """
        Random Forest hierárquico com country effects como features.
        """
        country_means = {}
        global_mean = y_train.mean()

        for country in countries_train.unique():
            country_mask = countries_train == country
            country_y = y_train[country_mask]
            country_means[country] = country_y.mean()

        X_train_augmented = X_train.copy()
        X_test_augmented = X_test.copy()

        train_country_effects = [country_means.get(country, global_mean) for country in countries_train]
        test_country_effects = [country_means.get(country, global_mean) for country in countries_test]

        X_train_augmented['country_effect'] = train_country_effects
        X_test_augmented['country_effect'] = test_country_effects

        _hm = SCIENTIFIC_CONFIG['hierarchical_model']
        rf_model = RandomForestRegressor(
            n_estimators=_hm['rf_n_estimators'],
            max_depth=max_depth,
            min_samples_split=_hm['rf_min_samples_split'],
            min_samples_leaf=min_samples_leaf,
            max_features=_hm['rf_max_features'],
            random_state=RANDOM_SEED,
            n_jobs=_hm['rf_n_jobs']
        )

        rf_model.fit(X_train_augmented, y_train)
        predictions = rf_model.predict(X_test_augmented)

        print(f"      Random Forest: {X_train_augmented.shape[1]} features totais "
              f"({X_train_augmented.shape[1]-1} base + 1 country_effect) × {X_train_augmented.shape[0]} samples")

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        feature_names = list(X_train_augmented.columns)
        feature_importance = dict(zip(feature_names, rf_model.feature_importances_))

        return {
            'model_name': 'random_forest_hierarchical',
            'architecture': 'dataframe_lib',
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'predictions': predictions.tolist(),
            'y_true': y_test.tolist(),
            'entities': [str(c) for c in countries_test],
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'country_effects': {str(k): float(v) for k, v in country_means.items()},
            'regularization_applied': (
                f"Regularizado: n_est={_hm['rf_n_estimators']}, "
                f"depth={max_depth}, split={_hm['rf_min_samples_split']}, "
                f"leaf={min_samples_leaf}"),
            'rf_params': {'n_estimators': 200, 'max_depth': int(max_depth), 'min_samples_split': 15, 'min_samples_leaf': int(min_samples_leaf)},
            'country_effect_importance': feature_importance.get('country_effect', 0),
            'features_count': X_train_augmented.shape[1]
        }

    def run_fold_analysis(self, fold_info: Dict) -> Dict:
        """
        Executar análise hierárquica completa para um fold específico.
        """
        # Latência decomposta: o carregamento do fold é do engine,
        # o ajuste é comum aos três paradigmas, que materializam em
        # pandas antes do scikit-learn. Medir o estágio inteiro
        # atribuía ao paradigma uma parcela que ele não controla.
        _load_t0 = time.perf_counter()
        fold_id = fold_info['fold_id']
        print(f"\nAnalisando Fold {fold_id} Polars (NORMAL)...")

        # Registrar features usadas para auditoria
        try:
            used = {
                'architecture': 'dataframe_lib',
                'fold_id': int(fold_id),
                'target': self.target_col,
                'total_features': len(self.available_features),
                'features': list(self.available_features),
            }
            used_path = os.path.join(self.results_path, f"used_features_fold_{fold_id}.json")
            with open(used_path, 'w') as f:
                json.dump(used, f, indent=2)
        except (OSError, IOError, TypeError):
            pass

        # Exclusão de gap years
        train_lazy = self.df_lazy.filter(
            (pl.col('year') >= fold_info['train_start']) &
            (pl.col('year') <= fold_info['train_end'])
        ).filter(
            ~((pl.col('year') >= fold_info['train_gap_start']) &
              (pl.col('year') <= fold_info['train_gap_end']))
        )
        val_lazy = self.df_lazy.filter(
            (pl.col('year') >= fold_info['val_start']) &
            (pl.col('year') <= fold_info['val_end'])
        )
        test_lazy = self.df_lazy.filter(
            (pl.col('year') >= fold_info['test_start']) &
            (pl.col('year') <= fold_info['test_end'])
        ).filter(
            ~((pl.col('year') >= fold_info['val_gap_start']) &
              (pl.col('year') <= fold_info['val_gap_end']))
        )

        # Contar registros
        n_train = train_lazy.select(pl.len()).collect().item()
        n_val = val_lazy.select(pl.len()).collect().item()
        n_test = test_lazy.select(pl.len()).collect().item()

        print(f"Dados Normais: Train={n_train}, Val={n_val}, Test={n_test}")

        X_train, y_train, countries_train = self._prepare_data(train_lazy)
        X_val, y_val, countries_val = self._prepare_data(val_lazy)
        X_test, y_test, countries_test = self._prepare_data(test_lazy)

        _fold_load_s = time.perf_counter() - _load_t0
        _fit_t0 = time.perf_counter()

        # P5: imputação e scaler ajustados exclusivamente no treino
        # (Kaufman et al. 2012). A imputação vem antes porque o scaler não
        # aceita ausentes, e ambas as estatísticas saem da mesma janela.
        (X_train, X_val, X_test), imputation_report = \
            impute_from_training_window(X_train, X_val, X_test)
        if imputation_report['columns_without_training_observation']:
            print(f"   [WARN] Sem observação no treino, deixadas ausentes: "
                  f"{imputation_report['columns_without_training_observation']}")

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print(f"   {len(self.available_features)} features, Train={X_train_scaled.shape}, Val={X_val_scaled.shape}, Test={X_test_scaled.shape}")
        print(f"   Países: Train={countries_train.nunique()}, Val={countries_val.nunique()}, Test={countries_test.nunique()}")

        # Modelos hierárquicos
        # HPO: seleção de hiperparâmetros via grid search no conjunto de
        # validação. Modelo final retreinado no treino completo para avaliação
        # no teste. Previne leakage (Kapoor & Narayanan, 2023).
        models = {}

        # 1. Simple Hierarchical (tuning de residual_shrinkage por validação)
        best_shrink = 0.8
        best_val_r2 = -1e9
        for rs in SCIENTIFIC_CONFIG['hierarchical_model']['residual_shrinkage_grid']:
            tmp = self.simple_hierarchical_model(X_train_scaled, y_train, X_val_scaled, y_val, countries_train, countries_val, residual_shrinkage=rs)
            if tmp['r2'] > best_val_r2:
                best_val_r2 = tmp['r2']
                best_shrink = rs
                val_simple = tmp
        test_simple = self.simple_hierarchical_model(X_train_scaled, y_train, X_test_scaled, y_test, countries_train, countries_test, residual_shrinkage=best_shrink)
        models['simple_hierarchical'] = {'val': val_simple, 'test': test_simple}

        # 2. Random Forest Hierarchical (tuning leve por validação)
        best_params = (6, 8)
        best_val_r2 = -1e9
        _hm = SCIENTIFIC_CONFIG['hierarchical_model']
        for depth in _hm['rf_max_depth_grid']:
            for leaf in _hm['rf_min_samples_leaf_grid']:
                tmp = self.random_forest_hierarchical(X_train_scaled, y_train, X_val_scaled, y_val, countries_train, countries_val, max_depth=depth, min_samples_leaf=leaf)
                if tmp['r2'] > best_val_r2:
                    best_val_r2 = tmp['r2']
                    best_params = (depth, leaf)
                    val_rf = tmp
        test_rf = self.random_forest_hierarchical(X_train_scaled, y_train, X_test_scaled, y_test, countries_train, countries_test, max_depth=best_params[0], min_samples_leaf=best_params[1])
        models['random_forest_hierarchical'] = {'val': val_rf, 'test': test_rf}

        # Análise dos gaps
        print(f"\n   Resultados hierárquicos (Val -> Test):")
        simple_gap = val_simple['r2'] - test_simple['r2']
        rf_gap = val_rf['r2'] - test_rf['r2']
        rf_country_imp = val_rf.get('country_effect_importance', 0)

        print(f"      Simple Hierarchical: Val R²={val_simple['r2']:.3f}, Test R²={test_simple['r2']:.3f}, Gap={simple_gap:+.3f}")
        print(f"      Random Forest:       Val R²={val_rf['r2']:.3f}, Test R²={test_rf['r2']:.3f}, Gap={rf_gap:+.3f}")
        print(f"         Country Effect: {rf_country_imp:.3f} (Target: 0.2-0.4)")

        # Interpretação dos gaps
        if abs(simple_gap) <= 0.15:
            print(f"      Simple: Gap corrigido - dentro da meta científica")
        elif abs(simple_gap) <= 0.2:
            print(f"      Simple: Gap moderado - aceitável para dados educacionais")
        else:
            print(f"      Simple: Gap ainda elevado - necessita regularização adicional")

        if abs(rf_gap) <= 0.15:
            print(f"      RF: Gap corrigido - excelente regularização aplicada")
        elif abs(rf_gap) <= 0.2:
            print(f"      RF: Gap moderado - regularização adequada")
        else:
            print(f"      RF: Gap ainda elevado - considerar regularização adicional")

        _fit_predict_s = time.perf_counter() - _fit_t0

        return {
            'fold_load_s': _fold_load_s,
            'fit_predict_s': _fit_predict_s,
            'fold_id': fold_id,
            'architecture': 'dataframe_lib',
            'mode': 'normal',
            'total_features': len(self.available_features),
            'models': models
        }

    def _write_prediction_artifact(self, all_results: Dict) -> None:
        """Delega à implementação compartilhada."""
        shared_write_prediction_artifact(all_results, architecture='dataframe_lib')

    def run_hierarchical_analysis(self):
        """
        Executar análise hierárquica completa para arquitetura Polars DataFrame.
        """
        print("Análise hierárquica Polars")
        print("   RidgeCV (Hoerl & Kennard 1970), Shrinkage James-Stein (Efron & Morris 1975)")

        _meta = SCIENTIFIC_CONFIG['hierarchical_model']
        all_results = {
            'architecture': 'dataframe_lib',
            'version': 'hierarchical_analysis',
            'mode': 'normal',
            'target': self.target_col,
            'total_features': len(self.available_features),
            'corrections_applied': {
                'simple_hierarchical': (
                    f"RidgeCV com alphas logspace("
                    f"{_meta['ridge_alpha_log10_start']}, "
                    f"{_meta['ridge_alpha_log10_stop']}, "
                    f"{_meta['ridge_alpha_count']}) + Shrinkage James-Stein"),
                'random_forest': (
                    f"Regularizado: n_est={_meta['rf_n_estimators']}, "
                    f"depth in {tuple(_meta['rf_max_depth_grid'])}, "
                    f"split={_meta['rf_min_samples_split']}, "
                    f"leaf in {tuple(_meta['rf_min_samples_leaf_grid'])}"),
                'regularization_approach': 'RidgeCV (Hoerl & Kennard 1970) + Shrinkage (Efron & Morris 1975)'
            },
            'folds': []
        }

        valid_folds = 0
        for fold_info in self.folds:
            _fold_t0 = time.perf_counter()
            fold_results = self.run_fold_analysis(fold_info)
            if fold_results is not None:
                fold_results['fold_duration_s'] = time.perf_counter() - _fold_t0
                all_results['folds'].append(fold_results)
                valid_folds += 1

                # Log dos resultados
                fold_id = fold_info['fold_id']
                for model_name, model_results in fold_results['models'].items():
                    val_r2 = model_results['val']['r2']
                    test_r2 = model_results['test']['r2']
                    gap = val_r2 - test_r2
                    print(f"   {model_name}: Val R²={val_r2:.3f}, Test R²={test_r2:.3f}, Gap={gap:+.3f}")

        if valid_folds == 0:
            print("Nenhum fold válido processado!")
            return all_results

        # Performance agregada
        print("Performance agregada LAZY EVALUATION:")

        for model_name in ['simple_hierarchical', 'random_forest_hierarchical']:
            val_r2s = [fold['models'][model_name]['val']['r2'] for fold in all_results['folds']
                      if model_name in fold['models']]
            test_r2s = [fold['models'][model_name]['test']['r2'] for fold in all_results['folds']
                       if model_name in fold['models']]

            if len(test_r2s) > 0:
                val_mean = np.mean(val_r2s)
                test_mean = np.mean(test_r2s)
                test_std = np.std(test_r2s)
                gap_mean = val_mean - test_mean

                print(f"\n   {model_name}:")
                print(f"      Val:  R² = {val_mean:.3f}")
                print(f"      Test: R² = {test_mean:.3f} ± {test_std:.3f}")
                print(f"      Gap:  {gap_mean:+.3f}")

                # Análise do gap
                abs_gap = abs(gap_mean)
                if abs_gap <= 0.15:
                    print(f"      Regularização efetiva - Gap dentro da meta científica")
                elif abs_gap <= 0.2:
                    print(f"      Gap aceitável")
                else:
                    print(f"      Necessita regularização adicional")

                all_results[f'{model_name}_summary'] = {
                    'val_mean_r2': float(val_mean),
                    'test_mean_r2': float(test_mean),
                    'test_std_r2': float(test_std),
                    'generalization_gap': float(gap_mean)
                }

        print("Resumo hierárquico LAZY EVALUATION")

        # Melhor modelo
        if 'simple_hierarchical_summary' in all_results and 'random_forest_hierarchical_summary' in all_results:
            simple_test = all_results['simple_hierarchical_summary']['test_mean_r2']
            rf_test = all_results['random_forest_hierarchical_summary']['test_mean_r2']

            if rf_test > simple_test:
                print(f"   Melhor modelo: Random Forest Hierárquico")
                print(f"   R² Teste: {rf_test:.3f}")
            else:
                print(f"   Melhor modelo: Simple Hierarchical")
                print(f"   R² Teste: {simple_test:.3f}")

        self._write_prediction_artifact(all_results)

        results_file = f"{self.results_path}/hierarchical_analysis_dataframe_lib_results_normal.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResultados Polars (NORMAL) salvos: {results_file}")

        return all_results


if __name__ == "__main__":
    # Sem status de saída a falha chega ao pipeline como sucesso: subprocess
    # check=True lê apenas o código de retorno.
    try:
        model = HierarchicalModelDataFrameLib()
        results = model.run_hierarchical_analysis()
        print("\nAnálise hierárquica Polars concluída!")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
