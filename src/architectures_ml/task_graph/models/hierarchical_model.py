#!/usr/bin/env python3
"""
Modelo Hierárquico para Arquitetura Data Lake.

Implementa modelos hierárquicos para arquitetura Data Lake com processamento
distribuído e schema-on-read para predição de dropout escolar.

Otimizado com batch compute para reduzir chamadas Dask.
"""

import time

import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
import json
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
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
from core.prediction_store import PredictionRecorder, predictions_path
from core.validation import impute_from_training_window
from core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED, setup_reproducibility

setup_reproducibility()

class HierarchicalModelTaskGraph:
    """
    Modelo Hierárquico para Arquitetura Data Lake.

    Implementa modelos hierárquicos com processamento distribuído e schema-on-read.
    """
    
    def __init__(self):
        print("Inicializando Modelo Hierárquico Dask")

        self.target_col = 'dropout_rate_task_graph'

        self._setup_normal_mode()

        self.results_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/models/hierarchical_results")
        os.makedirs(self.results_path, exist_ok=True)

        self._load_normal_summary()
    
    def _setup_normal_mode(self):
        """Setup para modo normal."""
        self.data_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/prep/master_data_task_graph.parquet")
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/prep/temporal_folds_task_graph.json")
        
        if not os.path.exists(self.data_path) or not os.path.exists(self.folds_path):
            raise FileNotFoundError("Dados Data Lake não encontrados")
        
        print("   Carregando dados com Dask...")
        self.ddf = dd.read_parquet(self.data_path, engine='pyarrow')
        self._needs_persist = True
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']
    
    def _load_normal_summary(self):
        """Carregar resumo dos dados normais com batch compute otimizado."""
        # Batch compute para estatísticas básicas
        stats_batch = {
            'year_min': self.ddf['year'].min(),
            'year_max': self.ddf['year'].max(),
            'n_countries': self.ddf['country_code'].nunique(),
            'unique_countries': self.ddf['country_code'].unique()
        }
        computed_stats = dask.compute(stats_batch)[0]
        
        n_records = self.ddf.shape[0].compute()
        print(f"   Dados: {n_records} registros, {self.ddf.npartitions} partições")
        print(f"   Período: {computed_stats['year_min']}-{computed_stats['year_max']}")
        print(f"   Países: {computed_stats['n_countries']}")
        print(f"   Target: {self.target_col}")
        print(f"   Folds: {len(self.folds)}")
        
        if self.target_col not in self.ddf.columns:
            raise ValueError(f"Target {self.target_col} não encontrado nos dados")
        

        
        selection_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/prep/feature_selection_task_graph.json")
        if os.path.exists(selection_path):
            try:
                with open(selection_path, 'r') as f:
                    selection_data = json.load(f)
                selected = selection_data.get('selected_features', [])
                # Incluir lag do target se existir
                if 'dropout_rate_lag_2' in self.ddf.columns and 'dropout_rate_lag_2' not in selected:
                    selected.append('dropout_rate_lag_2')
                if 'dropout_rate_lag_3' in self.ddf.columns and 'dropout_rate_lag_3' not in selected:
                    selected.append('dropout_rate_lag_3')
                # Filtrar por existência
                existing = [c for c in selected if c in self.ddf.columns]
                self.available_features = existing
                print(f"Features disponíveis (seleção científica): {len(self.available_features)}")
                # The lags above bypassed run_feature_selection, so the
                # set the models train on is audited here.
                self.feature_audit = audit_feature_set(
                    self.ddf, existing, self.target_col, SCIENTIFIC_CONFIG)
            except Exception as e:
                print(f"Falha ao carregar seleção de features: {e}")
                raise
        else:
            raise FileNotFoundError(f"Seleção de features não encontrada: {selection_path}. Execute setup.py antes.")
    
    def _prepare_data(self, data_ddf, reference_ddf):
        """
        Preparar dados usando processamento distribuído com batch compute.

        P5 (escopo de preprocessing): medianas/médias para imputação
        são computadas a partir de reference_ddf (= train), nunca do
        conjunto completo. Chamadas usam _prepare_data(val, train),
        _prepare_data(test, train).
        """
        ref_pd = reference_ddf[self.available_features].compute()
        medians = {}
        for feature in self.available_features:
            median_val = ref_pd[feature].median()
            medians[feature] = median_val if not pd.isna(median_val) else 0.0
        
        X_ddf = data_ddf[self.available_features]
        for feature, median_val in medians.items():
            X_ddf = X_ddf.assign(**{feature: X_ddf[feature].fillna(median_val)})
        
        final_data = {
            'X': X_ddf,
            'y': data_ddf[self.target_col],
            'countries': data_ddf['country_code'],
            'year': data_ddf['year'],
        }
        computed_final = dask.compute(final_data)[0]

        X_out, y_out, c_out, yr_out = computed_final['X'], computed_final['y'], computed_final['countries'], computed_final['year']
        valid_mask = y_out.notna()
        X_out, y_out, c_out, yr_out = X_out[valid_mask], y_out[valid_mask], c_out[valid_mask], yr_out[valid_mask]
        sort_idx = pd.DataFrame({'country_code': c_out, 'year': yr_out}).sort_values(['country_code', 'year']).index
        return X_out.loc[sort_idx].reset_index(drop=True), y_out.loc[sort_idx].reset_index(drop=True), c_out.loc[sort_idx].reset_index(drop=True)
    
    def simple_hierarchical_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 countries_train: pd.Series, countries_test: pd.Series,
                                 residual_shrinkage: float = 0.8) -> Dict:
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
            n_residuals = len(residuals_X)
            inner_folds = min(_hm['ridge_cv_folds'], n_residuals)
            ridge_cv = RidgeCV(alphas=alphas,
                               cv=inner_folds if inner_folds >= 2 else None)
            ridge_cv.fit(residuals_X, residuals_y)
            final_alpha = ridge_cv.alpha_
            residual_model = ridge_cv

            print(f"      Simple Hierarchical Dask:")
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
            'architecture': 'task_graph',
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
        
        print(f"      Random Forest: {X_train_augmented.shape[1]} features totais ({X_train_augmented.shape[1]-1} base + 1 country_effect) × {X_train_augmented.shape[0]} samples")
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        feature_names = list(X_train_augmented.columns)
        feature_importance = dict(zip(feature_names, rf_model.feature_importances_))

        return {
            'model_name': 'random_forest_hierarchical',
            'architecture': 'task_graph',
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
        """Executar análise hierárquica completa para um fold específico."""
        # Latência decomposta: o carregamento do fold é do engine,
        # o ajuste é comum aos três paradigmas, que materializam em
        # pandas antes do scikit-learn. Medir o estágio inteiro
        # atribuía ao paradigma uma parcela que ele não controla.
        _load_t0 = time.perf_counter()
        fold_id = fold_info['fold_id']
        print(f"\nAnalisando Fold {fold_id} Dask (NORMAL)...")

        # Registrar features usadas para auditoria
        try:
            used = {
                'architecture': 'task_graph',
                'fold_id': int(fold_id),
                'target': self.target_col,
                'total_features': len(self.available_features),
                'features': list(self.available_features),
            }
            used_path = os.path.join(self.results_path, f"used_features_fold_{fold_id}.json")
            with open(used_path, 'w') as f:
                json.dump(used, f, indent=2)
        except Exception:
            pass

        train_ddf = self.ddf[
            (self.ddf['year'] >= fold_info['train_start']) &
            (self.ddf['year'] <= fold_info['train_end'])
        ]
        train_ddf = train_ddf[
            ~((train_ddf['year'] >= fold_info['train_gap_start']) &
              (train_ddf['year'] <= fold_info['train_gap_end']))
        ]
        val_ddf = self.ddf[
            (self.ddf['year'] >= fold_info['val_start']) &
            (self.ddf['year'] <= fold_info['val_end'])
        ]
        test_ddf = self.ddf[
            (self.ddf['year'] >= fold_info['test_start']) &
            (self.ddf['year'] <= fold_info['test_end'])
        ]
        test_ddf = test_ddf[
            ~((test_ddf['year'] >= fold_info['val_gap_start']) &
              (test_ddf['year'] <= fold_info['val_gap_end']))
        ]
        n_train, n_val, n_test = dask.compute(
            train_ddf.shape[0], val_ddf.shape[0], test_ddf.shape[0]
        )
        print(f"Dados Normais: Train={n_train}, Val={n_val}, Test={n_test}")

        X_train, y_train, countries_train = self._prepare_data(train_ddf, train_ddf)
        X_val, y_val, countries_val = self._prepare_data(val_ddf, train_ddf)
        X_test, y_test, countries_test = self._prepare_data(test_ddf, train_ddf)
        
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
            'architecture': 'task_graph',
            'mode': 'normal',
            'total_features': len(self.available_features),
            'models': models
        }
    
    def _write_prediction_artifact(self, all_results: Dict) -> None:
        """Persist the test prediction vectors of every fold and model.

        Cross-paradigm equivalence is asserted over these vectors; the aggregate
        metrics stored alongside them cannot establish it.
        """
        recorder = PredictionRecorder('task_graph')
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

        path = predictions_path('task_graph', 'hierarchical')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        written = recorder.write(path)
        if written:
            print(f"Prediction vectors written: {written}")

    def run_hierarchical_analysis(self):
        """Executar análise hierárquica completa para arquitetura Data Lake."""
        if getattr(self, '_needs_persist', False):
            self.ddf = self.ddf.persist()
            self._needs_persist = False
        print("Análise hierárquica Dask")
        print("   RidgeCV (Hoerl & Kennard 1970), Shrinkage James-Stein (Efron & Morris 1975)")

        _meta = SCIENTIFIC_CONFIG['hierarchical_model']
        all_results = {
            'architecture': 'task_graph',
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
        print("Performance agregada SCHEMA-ON-READ:")
        
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
        
        print("Resumo hierárquico SCHEMA-ON-READ")
        
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

        results_file = f"{self.results_path}/hierarchical_analysis_task_graph_results_normal.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResultados Dask (NORMAL) salvos: {results_file}")
        
        return all_results

if __name__ == "__main__":
    model = HierarchicalModelTaskGraph()
    results = model.run_hierarchical_analysis()
    print("\nAnálise hierárquica Dask concluída!")
