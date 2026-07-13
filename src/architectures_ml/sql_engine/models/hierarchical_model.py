#!/usr/bin/env python3
"""
Modelo Hierárquico para Arquitetura Data Warehouse.

Implementa modelos hierárquicos para arquitetura Data Warehouse com ML Consumer pattern,
utilizando views DuckDB para queries diretas e connection pooling para performance.
"""

import time

import pandas as pd
import numpy as np
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

_actual_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _actual_project_root not in sys.path:
    sys.path.insert(0, _actual_project_root)

from core.prediction_store import PredictionRecorder, predictions_path
from core.scientific_config import RANDOM_SEED, setup_reproducibility

setup_reproducibility()

try:
    from core.config import get_absolute_output_path
except ImportError:
    def get_absolute_output_path(relative_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        return os.path.join(project_root, 'outputs', relative_path)

connection_manager_path = os.path.join(project_root, 'src', 'collection', 'sql_engine')
if connection_manager_path not in sys.path:
    sys.path.append(connection_manager_path)

try:
    from collection.sql_engine.connection_manager import DuckDBConnectionManager, SQLProcessingError
except ImportError:
    try:
        from connection_manager import DuckDBConnectionManager, SQLProcessingError
    except ImportError as e:
        raise ImportError(f"Não foi possível importar DuckDBConnectionManager: {e}")

class HierarchicalModelSQLFirst:
    """
    Modelo Hierárquico para Arquitetura Data Warehouse.

    Implementa ML Data Warehouse Consumer pattern com queries diretas às views,
    utilizando 13 features científicas selecionadas.
    """
    
    def __init__(self):
        print("Inicializando Modelo Hierárquico DuckDB")

        self.target_col = 'dropout_rate_sql_engine'

        print("   Pattern: ML Consumer com views")
        
        dataset_name = os.environ.get('DATASET_NAME', 'worldbank')
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/prep/temporal_folds_sql_engine.json")
        self.results_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/models/hierarchical_results")
        self.db_path = get_absolute_output_path(f'collection/sql_engine/{dataset_name}_data.duckdb')
        
        os.makedirs(self.results_path, exist_ok=True)
        
        if not os.path.exists(self.folds_path):
            raise FileNotFoundError(f"Folds Data Warehouse não encontrados: {self.folds_path}")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"DuckDB Data Warehouse não encontrado: {self.db_path}")
        
        try:
            self.conn_manager = DuckDBConnectionManager(
                db_path=self.db_path,
                max_retries=3,
                retry_delay=1.0
            )
            print(f"   Connection Manager: {self.db_path}")
        except Exception as e:
            raise RuntimeError(f"Falha ao inicializar Connection Manager: {e}")
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']
        
        self._verify_views()
        self._load_data_summary()
    
    def _verify_views(self):
        """Verificar se views necessárias existem no Data Warehouse."""
        print("   Verificando views...")

        try:
            count = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
            if count > 0:
                print(f"   Views verificadas: {count} registros")
            else:
                raise RuntimeError("Views do setup vazias")
        except SQLProcessingError as e:
            raise RuntimeError(f"Erro ao verificar views do setup: {e}")
    
    def _load_data_summary(self):
        """Carregar resumo dos dados via queries diretas às views."""
        print("   Carregando resumo dos dados...")
        
        try:
            total_records = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
            min_year = self.conn_manager.execute_scalar("SELECT MIN(year) FROM analytics_wide")
            max_year = self.conn_manager.execute_scalar("SELECT MAX(year) FROM analytics_wide")
            total_countries = self.conn_manager.execute_scalar("SELECT COUNT(DISTINCT country_code) FROM analytics_wide")
            
            print(f"   Dados: {total_records} observações")
            print(f"   Período: {min_year}-{max_year}")
            print(f"   Países: {total_countries}")
            print(f"   Target: {self.target_col}")
            print(f"   Folds: {len(self.folds)}")
            
            target_exists = self.conn_manager.execute_scalar(f"""
                SELECT COUNT(*) > 0 
                FROM information_schema.columns 
                WHERE table_name = 'analytics_wide' 
                AND column_name = '{self.target_col}'
            """)
            
            if not target_exists:
                raise ValueError(f"Target {self.target_col} não encontrado na tabela analytics_wide")
            
            target_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
            target_std = self.conn_manager.execute_scalar(f"SELECT STDDEV({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
            target_min = self.conn_manager.execute_scalar(f"SELECT MIN({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
            target_max = self.conn_manager.execute_scalar(f"SELECT MAX({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
            
            print(f"   Target stats: mean={target_mean:.2f}%, std={target_std:.2f}%")
            print(f"   Target range: [{target_min:.2f}%, {target_max:.2f}%]")
                
        except SQLProcessingError as e:
            raise RuntimeError(f"Erro ao carregar resumo via views: {e}")
    
    def _load_ml_fold_data(self, fold_id: int, split: str) -> pd.DataFrame:
        """Carregar dados do fold via queries diretas às views."""
        view_name = f"vw_fold_{fold_id}_{split}"

        try:
            query = f"""
                SELECT *
                FROM {view_name}
                WHERE {self.target_col} IS NOT NULL
                ORDER BY country_code, year
            """
            
            df = self.conn_manager.execute_sql(query)
            
            if df.empty:
                raise SQLProcessingError(f"No data returned from view {view_name}")
            
            return df
            
        except SQLProcessingError as e:
            raise RuntimeError(f"Erro ao carregar dados do fold {fold_id} split {split}: {e}")
    
    def _prepare_features(self, train_clean):
        """Preparar lista de features baseado no modo."""
        selection_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/prep/feature_selection_sql_engine.json")

        if os.path.exists(selection_path):
            # Modo Normal: carregar features selecionadas
            with open(selection_path, 'r') as f:
                selection_data = json.load(f)
            available_features = selection_data['selected_features']
            print(f"   {len(available_features)} features do feature selection")
            print(f"   Método: {selection_data.get('selection_method', 'N/A')}")
        else:
            raise FileNotFoundError(f"Seleção de features não encontrada: {selection_path}. Execute setup.py antes.")

        if 'dropout_rate_lag_2' in train_clean.columns and 'dropout_rate_lag_2' not in available_features:
            available_features.append('dropout_rate_lag_2')
        if 'dropout_rate_lag_3' in train_clean.columns and 'dropout_rate_lag_3' not in available_features:
            available_features.append('dropout_rate_lag_3')

        all_columns = list(train_clean.columns)
        available_features = [feat for feat in available_features if feat in all_columns]

        return available_features
    
    def _prepare_data(self, data, reference_data, available_features):
        """
        Preparar dados para treinamento.

        P5 (escopo de preprocessing): imputação usa mediana de
        reference_data (= train), nunca do conjunto completo.
        Chamadas: _prepare_data(val, train, ...), _prepare_data(test, train, ...).
        """
        # P5: imputar com mediana do treino; fallback 0 para features sem dados
        X = data[available_features].fillna(reference_data[available_features].median()).fillna(0)
        y = data[self.target_col]
        countries = data['country_code']
        return X, y, countries
    
    def simple_hierarchical_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 countries_train: pd.Series, countries_test: pd.Series,
                                 residual_shrinkage: float = 0.8) -> Dict:
        """
        Modelo hierárquico simples: médias por país + resíduos com Ridge regularizado.
        """
        global_mean = y_train.mean()
        n_countries = countries_train.nunique()
        total_samples = len(y_train)
        
        # Calcular médias por país com shrinkage adaptativo
        country_means = {}
        country_residuals_X = []
        country_residuals_y = []
        country_sample_counts = {}
        
        print(f"      Processamento adaptativo DuckDB: {n_countries} países, {total_samples} amostras")
        
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
            alphas = np.logspace(-1, 3, 20)  # 0.1 a 1000
            ridge_cv = RidgeCV(alphas=alphas, cv=min(3, len(residuals_X)))
            ridge_cv.fit(residuals_X, residuals_y)
            final_alpha = ridge_cv.alpha_
            residual_model = ridge_cv

            print(f"      Simple Hierarchical DuckDB:")
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
            'architecture': 'sql_engine',
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
                'alpha_selection': 'RidgeCV com logspace(-1, 3, 20)',
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
        
        # Mesmos hiperparâmetros do Data Lake para comparação justa
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=max_depth,
            min_samples_split=15,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            random_state=RANDOM_SEED,
            n_jobs=1
        )
        
        rf_model.fit(X_train_augmented, y_train)
        predictions = rf_model.predict(X_test_augmented)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        feature_names = list(X_train_augmented.columns)
        feature_importance = dict(zip(feature_names, rf_model.feature_importances_))

        return {
            'model_name': 'random_forest_hierarchical',
            'architecture': 'sql_engine',
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'predictions': predictions.tolist(),
            'y_true': y_test.tolist(),
            'entities': [str(c) for c in countries_test],
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'country_effects': {str(k): float(v) for k, v in country_means.items()},
            'regularization_applied': f'Regularizado: n_est=200, depth={max_depth}, split=15, leaf={min_samples_leaf}',
            'country_effect_importance': feature_importance.get('country_effect', 0),
            'rf_params': {'n_estimators': 200, 'max_depth': int(max_depth), 'min_samples_split': 15, 'min_samples_leaf': int(min_samples_leaf)},
            'features_count': X_train_augmented.shape[1]
        }
    
    def run_fold_analysis(self, fold_info: Dict) -> Dict:
        """Executar análise completa para um fold via ML Data Warehouse Consumer pattern."""
        fold_id = fold_info['fold_id']
        print(f"\nFold {fold_id}: Train({fold_info['train_start']}-{fold_info['train_end']}) -> Val({fold_info['val_start']}-{fold_info['val_end']}) -> Test({fold_info['test_start']}-{fold_info['test_end']})")
        
        try:
            train_data = self._load_ml_fold_data(fold_id, 'train')
            val_data = self._load_ml_fold_data(fold_id, 'val')
            test_data = self._load_ml_fold_data(fold_id, 'test')
            
            print(f"   Dados: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            print(f"   Gaps: Train-Val={fold_info['val_start']-fold_info['train_end']-1}yr, Val-Test={fold_info['test_start']-fold_info['val_end']-1}yr")

        except Exception as e:
            print(f"   Erro ao carregar dados do fold {fold_id}: {e}")
            return {}
        
        if len(train_data) == 0 or len(test_data) == 0:
            print(f"   Fold {fold_id}: Dados insuficientes")
            return {}
        
        available_features = self._prepare_features(train_data)

        try:
            used = {
                'architecture': 'sql_engine',
                'fold_id': int(fold_id),
                'target': self.target_col,
                'total_features': len(available_features),
                'features': list(available_features),
            }
            used_path = os.path.join(self.results_path, f"used_features_fold_{fold_id}.json")
            with open(used_path, 'w') as f:
                json.dump(used, f, indent=2)
        except Exception:
            pass
        
        X_train, y_train, countries_train = self._prepare_data(train_data, train_data, available_features)
        X_val, y_val, countries_val = self._prepare_data(val_data, train_data, available_features)
        X_test, y_test, countries_test = self._prepare_data(test_data, train_data, available_features)
        
        # P5: scaler ajustado exclusivamente no treino (Kaufman et al. 2012)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        print(f"   Executando modelos hierárquicos:")
        
        # Modelos hierárquicos
        # HPO: seleção de hiperparâmetros via grid search no conjunto de
        # validação. Modelo final retreinado no treino completo para avaliação
        # no teste. Previne leakage (Kapoor & Narayanan, 2023).
        models = {}

        # 1. Simple Hierarchical (tuning de residual_shrinkage)
        best_shrink = 0.8
        best_val_r2 = -1e9
        for rs in [0.6, 0.8, 1.0]:
            tmp = self.simple_hierarchical_model(
                X_train_scaled, y_train, X_val_scaled, y_val,
                countries_train, countries_val, residual_shrinkage=rs
            )
            if tmp['r2'] > best_val_r2:
                best_val_r2 = tmp['r2']
                best_shrink = rs
                val_simple = tmp
        test_simple = self.simple_hierarchical_model(
            X_train_scaled, y_train, X_test_scaled, y_test,
            countries_train, countries_test, residual_shrinkage=best_shrink
        )
        models['simple_hierarchical'] = {'val': val_simple, 'test': test_simple}

        # 2. Random Forest Hierarchical (tuning leve)
        best_params = (6, 8)
        best_val_r2 = -1e9
        for depth in [5, 6, 7]:
            for leaf in [5, 8, 12]:
                tmp = self.random_forest_hierarchical(
                    X_train_scaled, y_train, X_val_scaled, y_val,
                    countries_train, countries_val,
                    max_depth=depth, min_samples_leaf=leaf
                )
                if tmp['r2'] > best_val_r2:
                    best_val_r2 = tmp['r2']
                    best_params = (depth, leaf)
                    val_rf = tmp
        test_rf = self.random_forest_hierarchical(
            X_train_scaled, y_train, X_test_scaled, y_test,
            countries_train, countries_test,
            max_depth=best_params[0], min_samples_leaf=best_params[1]
        )
        models['random_forest_hierarchical'] = {'val': val_rf, 'test': test_rf}
        
        # Análise dos gaps
        print(f"   Resultados hierárquicos (Val -> Test):")
        simple_gap = val_simple['r2'] - test_simple['r2']
        rf_gap = val_rf['r2'] - test_rf['r2']
        rf_country_imp = val_rf.get('country_effect_importance', 0)
        
        print(f"      Simple Hierarchical: Val R²={val_simple['r2']:.3f}, Test R²={test_simple['r2']:.3f}, Gap={simple_gap:+.3f}")
        print(f"      Random Forest:       Val R²={val_rf['r2']:.3f}, Test R²={test_rf['r2']:.3f}, Gap={rf_gap:+.3f}")
        print(f"         Country Effect: {rf_country_imp:.3f} (Target: 0.2-0.4)")
        
        # Interpretação dos gaps
        if abs(simple_gap) <= 0.15:
            print(f"      Simple: Gap dentro da meta")
        elif abs(simple_gap) <= 0.2:
            print(f"      Simple: Gap moderado")
        else:
            print(f"      Simple: Gap elevado")

        if abs(rf_gap) <= 0.15:
            print(f"      RF: Gap dentro da meta")
        elif abs(rf_gap) <= 0.2:
            print(f"      RF: Gap moderado")
        else:
            print(f"      RF: Gap elevado")
        
        return {
            'fold_id': fold_id,
            'architecture': 'sql_engine',
            'total_features': len(available_features),
            'models': models
        }
    
    def _write_prediction_artifact(self, all_results: Dict) -> None:
        """Persist the test prediction vectors of every fold and model.

        Cross-paradigm equivalence is asserted over these vectors; the aggregate
        metrics stored alongside them cannot establish it.
        """
        recorder = PredictionRecorder('sql_engine')
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

        path = predictions_path('sql_engine')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        written = recorder.write(path)
        if written:
            print(f"Prediction vectors written: {written}")

    def run_hierarchical_analysis(self):
        """Executar análise hierárquica completa via ML Data Warehouse Consumer."""
        print("Análise hierárquica DuckDB")
        print("   RidgeCV (Hoerl & Kennard 1970), Shrinkage James-Stein (Efron & Morris 1975)")
        
        try:
            all_results = {
                'architecture': 'sql_engine',
                'version': 'hierarchical_analysis',
                'target': self.target_col,
                'mode': 'normal',
                'corrections_applied': {
                    'simple_hierarchical': 'RidgeCV com alphas logspace(0.1, 1000) + Shrinkage James-Stein',
                    'random_forest': 'Regularizado: n_est=200, depth=6, split=15, leaf=8',
                    'regularization_approach': 'RidgeCV (Hoerl & Kennard 1970) + Shrinkage (Efron & Morris 1975)'
                },
                'folds': []
            }
            
            for fold_info in self.folds_config['folds']:
                _fold_t0 = time.perf_counter()
                fold_results = self.run_fold_analysis(fold_info)
                if fold_results:
                    fold_results['fold_duration_s'] = time.perf_counter() - _fold_t0
                    all_results['folds'].append(fold_results)
                    
                    # Log dos resultados
                    fold_id = fold_info['fold_id']
                    if 'models' in fold_results:
                        for model_name, model_results in fold_results['models'].items():
                            val_r2 = model_results['val']['r2']
                            test_r2 = model_results['test']['r2']
                            gap = val_r2 - test_r2
                            print(f"   {model_name}: Val R²={val_r2:.3f}, Test R²={test_r2:.3f}, Gap={gap:+.3f}")
            
            # Performance agregada
            if all_results['folds']:
                print(f"\nPerformance agregada DuckDB:")
                
                for model_name in ['simple_hierarchical', 'random_forest_hierarchical']:
                    val_r2s = []
                    test_r2s = []
                    
                    for fold in all_results['folds']:
                        if 'models' in fold and model_name in fold['models']:
                            val_r2s.append(fold['models'][model_name]['val']['r2'])
                            test_r2s.append(fold['models'][model_name]['test']['r2'])
                    
                    if val_r2s and test_r2s:
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
                            print(f"      Regularização efetiva - gap dentro da meta")
                        elif abs_gap <= 0.2:
                            print(f"      Gap aceitável")
                        else:
                            print(f"      Necessita regularização adicional")
                
                print(f"\nResumo hierárquico DuckDB")
            
            self._write_prediction_artifact(all_results)

            results_file = f"{self.results_path}/hierarchical_analysis_sql_engine_results.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\nResultados salvos: {results_file}")
            
            return all_results
            
        except Exception as e:
            print(f"Erro na análise hierárquica: {e}")
            return {
                'architecture': 'sql_engine',
                'error': str(e),
                'folds': []
            }
        finally:
            # Cleanup connection
            if hasattr(self, 'conn_manager') and self.conn_manager:
                try:
                    self.conn_manager.close_connection()
                    print("   Connection Manager fechado")
                except Exception as e:
                    print(f"   Erro ao fechar Connection Manager: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hierarchical Model Data Warehouse')
    args = parser.parse_args()

    try:
        model = HierarchicalModelSQLFirst()
        results = model.run_hierarchical_analysis()
        print(f"\nAnálise hierárquica DuckDB concluída!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
