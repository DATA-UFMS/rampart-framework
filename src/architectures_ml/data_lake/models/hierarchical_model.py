#!/usr/bin/env python3
"""
Modelo Hierárquico para Arquitetura Data Lake.

Implementa modelos hierárquicos para arquitetura Data Lake com processamento
distribuído e schema-on-read para predição de dropout escolar.

Otimizado com batch compute para reduzir chamadas Dask.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
import json
import os
import sys
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from typing import Dict
warnings.filterwarnings('ignore')

# Adicionar path para configuração
core_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core')
core_path = os.path.abspath(core_path)
if core_path not in sys.path:
    sys.path.append(core_path)

from config import get_absolute_output_path

class HierarchicalModelDataLake:
    """
    Modelo Hierárquico para Arquitetura Data Lake.
    
    Implementa modelos hierárquicos com processamento distribuído e schema-on-read.
    Suporta modo normal (13 features científicas) e enhanced (19 features + engineered).
    """
    
    def __init__(self, use_enhanced_features: bool = False):
        print("Inicializando Modelo Hierárquico Data Lake")
        print("=" * 60)
        print("Arquitetura: Data Lake com processamento distribuído")
        
        self.use_enhanced_features = use_enhanced_features
        self.enhanced_data = {}
        self.target_col = 'dropout_rate_data_lake'
        
        if use_enhanced_features:
            print("Modo: Enhanced Features (pipeline de feature engineering)")
            self._setup_enhanced_mode()
        else:
            print("Modo: Folds Normais (setup básico)")
            self._setup_normal_mode()
        
        self.results_path = get_absolute_output_path("ml_pipeline/architectures/data_lake/models/hierarchical_results")
        os.makedirs(self.results_path, exist_ok=True)
        
        if self.use_enhanced_features:
            self._load_enhanced_summary()
        else:
            self._load_normal_summary()
    
    def _setup_enhanced_mode(self):
        """Setup para modo enhanced features."""
        self.enhanced_base_path = get_absolute_output_path("ml_pipeline/feature_engineering/data_lake/enhanced")
        self.enhanced_metadata_path = f"{self.enhanced_base_path}/incremental_feature_engineering_metadata.json"
        
        if not os.path.exists(self.enhanced_metadata_path):
            print(f"Enhanced features não encontradas. Fazendo fallback para modo normal...")
            self.use_enhanced_features = False
            self._setup_normal_mode()
            return
        
        print(f"Enhanced features encontradas: {self.enhanced_base_path}")
        with open(self.enhanced_metadata_path, 'r') as f:
            self.enhanced_metadata = json.load(f)
            self.folds = self.enhanced_metadata['enhanced_folds_data']
            self.folds = [{'fold_id': int(k.split('_')[1]), **v} for k, v in self.folds.items()]
    
    def _setup_normal_mode(self):
        """Setup para modo normal."""
        self.data_path = get_absolute_output_path("ml_pipeline/architectures/data_lake/prep/master_data_data_lake.parquet")
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/data_lake/prep/temporal_folds_data_lake.json")
        
        if not os.path.exists(self.data_path) or not os.path.exists(self.folds_path):
            raise FileNotFoundError("Dados Data Lake não encontrados")
        
        print("Carregando dados Data Lake com processamento distribuído...")
        self.ddf = dd.read_parquet(self.data_path, engine='pyarrow')
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
        
        print(f"Dados carregados: {len(self.ddf)} registros, {self.ddf.npartitions} partições")
        print(f"Período: {computed_stats['year_min']}-{computed_stats['year_max']}")
        print(f"Países: {computed_stats['n_countries']}")
        print(f"Target: {self.target_col}")
        print(f"Folds: {len(self.folds)}")
        
        if self.target_col not in self.ddf.columns:
            raise ValueError(f"Target {self.target_col} não encontrado nos dados")
        
        self.country_encoder = LabelEncoder()
        self.country_encoder.fit(computed_stats['unique_countries'])
        
        # Alinhar features com seleção científica salva no setup
        selection_path = get_absolute_output_path("ml_pipeline/architectures/data_lake/prep/feature_selection_data_lake.json")
        if os.path.exists(selection_path):
            try:
                with open(selection_path, 'r') as f:
                    selection_data = json.load(f)
                selected = selection_data.get('selected_features', [])
                self.selection_path = selection_path
                # Incluir lag do target se existir
                if 'dropout_rate_lag_2' in self.ddf.columns and 'dropout_rate_lag_2' not in selected:
                    selected.append('dropout_rate_lag_2')
                if 'dropout_rate_lag_3' in self.ddf.columns and 'dropout_rate_lag_3' not in selected:
                    selected.append('dropout_rate_lag_3')
                # Filtrar por existência
                existing = [c for c in selected if c in self.ddf.columns]
                self.available_features = existing
                print(f"Features disponíveis (seleção científica): {len(self.available_features)}")
            except Exception as e:
                print(f"Falha ao carregar seleção de features: {e}")
                raise
        else:
            raise FileNotFoundError(f"Seleção de features não encontrada: {selection_path}. Execute setup.py antes.")
    
    def _load_enhanced_summary(self):
        """Carregar resumo dos dados enhanced."""
        print(f"Folds enhanced: {len(self.folds)}")
        print(f"Target: {self.target_col}")
        print(f"Features adicionais incluídas do feature engineering")
        
        sample_fold = self._load_fold_enhanced_data(0)
        if sample_fold:
            sample_ddf = dd.concat([sample_fold['train'], sample_fold['val'], sample_fold['test']], ignore_index=True)
            
            # Batch compute para estatísticas enhanced
            enhanced_stats = {
                'year_min': sample_ddf['year'].min(),
                'year_max': sample_ddf['year'].max(),
                'n_countries': sample_ddf['country_code'].nunique(),
                'unique_countries': sample_ddf['country_code'].unique(),
                'target_mean': sample_ddf[self.target_col].mean() if self.target_col in sample_ddf.columns else None,
                'target_std': sample_ddf[self.target_col].std() if self.target_col in sample_ddf.columns else None
            }
            computed_enhanced = dask.compute(enhanced_stats)[0]
            
            print(f"Exemplo (fold 0): {len(sample_ddf)} registros, {sample_ddf.npartitions} partições")
            print(f"Período: {computed_enhanced['year_min']}-{computed_enhanced['year_max']}")
            print(f"Países: {computed_enhanced['n_countries']}")
            
            self.country_encoder = LabelEncoder()
            self.country_encoder.fit(computed_enhanced['unique_countries'])
            
            self.available_features = [col for col in sample_ddf.columns
                                     if col not in ['country_code', 'year', self.target_col]]
            
            print(f"Features enhanced disponíveis: {len(self.available_features)}")
            
            if self.target_col in sample_ddf.columns and computed_enhanced['target_mean'] is not None:
                print(f"Target stats: μ={computed_enhanced['target_mean']:.2f}%, σ={computed_enhanced['target_std']:.2f}%")
            else:
                raise ValueError(f"Target {self.target_col} não encontrado nos dados enhanced")
        else:
            print("Não foi possível carregar dados enhanced para configuração inicial")
            self.available_features = []
            self.country_encoder = LabelEncoder()
    
    def _load_fold_enhanced_data(self, fold_id: int) -> Dict:
        """Carregar dados de um fold específico no modo enhanced."""
        if fold_id in self.enhanced_data:
            return self.enhanced_data[fold_id]
        
        fold_dir = f"{self.enhanced_base_path}/fold_{fold_id}"
        if not os.path.exists(fold_dir):
            return None
        
        try:
            train_path = f"{fold_dir}/train_data_lake_enhanced.parquet"
            val_path = f"{fold_dir}/val_data_lake_enhanced.parquet"
            test_path = f"{fold_dir}/test_data_lake_enhanced.parquet"
            
            fold_data = {
                'train': dd.read_parquet(train_path, engine='pyarrow'),
                'val': dd.read_parquet(val_path, engine='pyarrow'),
                'test': dd.read_parquet(test_path, engine='pyarrow')
            }
            
            self.enhanced_data[fold_id] = fold_data
            return fold_data
        except Exception as e:
            print(f"Erro ao carregar fold {fold_id} enhanced: {e}")
            return None
    
    def _prepare_data(self, data_ddf, reference_ddf):
        """Preparar dados usando processamento distribuído com batch compute."""
        # Batch compute para medianas/médias
        medians_batch = {}
        means_batch = {}
        
        for feature in self.available_features:
            try:
                # Quantil 0.5 (mediana) com dask quantile, mais robusto
                medians_batch[f"{feature}_median"] = reference_ddf[feature].quantile(0.5)
            except Exception:
                means_batch[f"{feature}_mean"] = reference_ddf[feature].mean()
        
        if medians_batch:
            computed_medians = dask.compute(medians_batch)[0]
        else:
            computed_medians = {}
            
        if means_batch:
            computed_means = dask.compute(means_batch)[0]
        else:
            computed_means = {}
        
        medians = {}
        for feature in self.available_features:
            if f"{feature}_median" in computed_medians:
                median_val = computed_medians[f"{feature}_median"]
                medians[feature] = median_val if not pd.isna(median_val) else 0.0
            elif f"{feature}_mean" in computed_means:
                mean_val = computed_means[f"{feature}_mean"]
                medians[feature] = mean_val if not pd.isna(mean_val) else 0.0
            else:
                medians[feature] = 0.0
        
        X_ddf = data_ddf[self.available_features]
        for feature, median_val in medians.items():
            X_ddf = X_ddf.assign(**{feature: X_ddf[feature].fillna(median_val)})
        
        # Batch compute para dados finais
        final_data = {
            'X': X_ddf,
            'y': data_ddf[self.target_col],
            'countries': data_ddf['country_code']
        }
        computed_final = dask.compute(final_data)[0]
        
        return computed_final['X'], computed_final['y'], computed_final['countries']
    
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
        
        print(f"Processamento hierárquico distribuído: {n_countries} países, {total_samples} amostras")
        
        for country in countries_train.unique():
            country_mask = countries_train == country
            country_y = y_train[country_mask]
            country_samples = len(country_y)
            country_sample_counts[country] = country_samples
            
            # Shrinkage adaptativo (James-Stein)
            shrinkage_factor = min(0.8, country_samples / (country_samples + 5.0))
            raw_country_mean = country_y.mean()
            country_mean_shrunk = (shrinkage_factor * raw_country_mean + 
                                 (1 - shrinkage_factor) * global_mean)
            country_means[country] = country_mean_shrunk
            
            # Calcular resíduos
            country_X = X_train[country_mask]
            country_residuals = country_y - country_mean_shrunk
            
            country_residuals_X.append(country_X)
            country_residuals_y.extend(country_residuals)
        
        # Treinar modelo para resíduos
        if len(country_residuals_X) > 0:
            residuals_X = pd.concat(country_residuals_X, ignore_index=True)
            residuals_y = np.array(country_residuals_y)
            
            features_count = residuals_X.shape[1]
            samples_count = len(residuals_y)
            
            # Regularização adaptativa
            feature_sample_ratio = features_count / samples_count
            base_alpha = 20.0
            complexity_penalty = 1 + (feature_sample_ratio ** 2) * 10
            adaptive_alpha = min(200.0, base_alpha * complexity_penalty)
            
            # Cross-validation para α
            final_alpha = adaptive_alpha
            if samples_count > 30:
                test_alphas = [adaptive_alpha * 0.5, adaptive_alpha, adaptive_alpha * 2.0]
                best_score = -np.inf
                
                for alpha_test in test_alphas:
                    try:
                        ridge_test = Ridge(alpha=alpha_test)
                        scores = cross_val_score(ridge_test, residuals_X, residuals_y, 
                                               cv=min(3, samples_count//10), scoring='r2')
                        mean_score = np.mean(scores)
                        if mean_score > best_score:
                            best_score = mean_score
                            final_alpha = alpha_test
                    except:
                        continue
            
            residual_model = Ridge(alpha=final_alpha)
            residual_model.fit(residuals_X, residuals_y)
            
            print(f"      Simple Hierarchical Data Lake:")
            print(f"         {features_count} features × {samples_count} samples de resíduos")
            print(f"         α adaptativo: {final_alpha:.1f} (base: {base_alpha}, ratio: {feature_sample_ratio:.3f})")
            print(f"         Shrinkage aplicado em {n_countries} países")
        else:
            residual_model = None
            features_count = 0
            samples_count = 0
            final_alpha = 0.0
        
        # Predição no teste
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
        
        # Métricas
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'model_name': 'simple_hierarchical',
            'architecture': 'data_lake',
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'predictions': predictions.tolist(),
            'y_true': y_test.tolist(),
            'country_effects': {str(k): float(v) for k, v in country_means.items()},
            'regularization_applied': f'Adaptativo: α={final_alpha:.1f} com Shrinkage e CV',
            'features_count': features_count,
            'regularization_details': {
                'adaptive_alpha': float(final_alpha),
                'shrinkage_applied': True,
                'cv_optimization': samples_count > 30,
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
        # Calcular médias por país
        country_means = {}
        global_mean = y_train.mean()
        
        for country in countries_train.unique():
            country_mask = countries_train == country
            country_y = y_train[country_mask]
            country_means[country] = country_y.mean()
        
        # Adicionar country effects como features
        X_train_augmented = X_train.copy()
        X_test_augmented = X_test.copy()
        
        train_country_effects = [country_means.get(country, global_mean) for country in countries_train]
        test_country_effects = [country_means.get(country, global_mean) for country in countries_test]
        
        X_train_augmented['country_effect'] = train_country_effects
        X_test_augmented['country_effect'] = test_country_effects
        
        # Random Forest regularizado
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=max_depth,
            min_samples_split=15,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_augmented, y_train)
        predictions = rf_model.predict(X_test_augmented)
        
        print(f"      Random Forest: {X_train_augmented.shape[1]} features totais ({X_train_augmented.shape[1]-1} base + 1 country_effect) × {X_train_augmented.shape[0]} samples")
        
        # Métricas
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Importância de features
        feature_names = list(X_train_augmented.columns)
        feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
        
        return {
            'model_name': 'random_forest_hierarchical',
            'architecture': 'data_lake',
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'predictions': predictions.tolist(),
            'y_true': y_test.tolist(),
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'country_effects': {str(k): float(v) for k, v in country_means.items()},
            'regularization_applied': 'Regularizado: n_est=200, depth=6, split=15, leaf=8',
            'rf_params': {'max_depth': int(max_depth), 'min_samples_leaf': int(min_samples_leaf)},
            'country_effect_importance': feature_importance.get('country_effect', 0),
            'features_count': X_train_augmented.shape[1]
        }
    
    def run_fold_analysis(self, fold_info: Dict) -> Dict:
        """Executar análise hierárquica completa para um fold específico."""
        fold_id = fold_info['fold_id']
        mode_text = "ENHANCED" if self.use_enhanced_features else "NORMAL"
        print(f"\nAnalisando Fold {fold_id} Data Lake ({mode_text})...")
        
        # Registrar features usadas para auditoria
        try:
            used = {
                'architecture': 'data_lake',
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
        
        if self.use_enhanced_features:
            fold_data = self._load_fold_enhanced_data(fold_id)
            if not fold_data:
                print(f"Fold {fold_id}: Dados enhanced não disponíveis")
                return None
            
            train_data = fold_data['train']
            val_data = fold_data['val']
            test_data = fold_data['test']
            print(f"Dados Enhanced: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        else:
            train_ddf = self.ddf[
                (self.ddf['year'] >= fold_info['train_start']) & 
                (self.ddf['year'] <= fold_info['train_end'])
            ]
            val_ddf = self.ddf[
                (self.ddf['year'] >= fold_info['val_start']) & 
                (self.ddf['year'] <= fold_info['val_end'])
            ]
            test_ddf = self.ddf[
                (self.ddf['year'] >= fold_info['test_start']) & 
                (self.ddf['year'] <= fold_info['test_end'])
            ]
            print(f"Dados Normais: Train={len(train_ddf)}, Val={len(val_ddf)}, Test={len(test_ddf)}")
        
        if self.use_enhanced_features:
            X_train, y_train, countries_train = self._prepare_data(train_data, train_data)
            X_val, y_val, countries_val = self._prepare_data(val_data, train_data)
            X_test, y_test, countries_test = self._prepare_data(test_data, train_data)
        else:
            X_train, y_train, countries_train = self._prepare_data(train_ddf, train_ddf)
            X_val, y_val, countries_val = self._prepare_data(val_ddf, train_ddf)
            X_test, y_test, countries_test = self._prepare_data(test_ddf, train_ddf)
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        print(f"   EXECUTANDO MODELOS HIERÁRQUICOS:")
        print(f"   CONTADOR DE FEATURES: {len(self.available_features)} features disponíveis")
        print(f"   Dados preparados - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        print(f"   DIMENSÕES: Train={X_train_scaled.shape}, Val={X_val_scaled.shape}, Test={X_test_scaled.shape}")
        print(f"   PAÍSES: Train={countries_train.nunique()}, Val={countries_val.nunique()}, Test={countries_test.nunique()}")
        
        # Modelos hierárquicos
        models = {}
        
        # 1. Simple Hierarchical (tuning de residual_shrinkage por validação)
        best_shrink = 0.8
        best_val_r2 = -1e9
        for rs in [0.6, 0.8, 1.0]:
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
        for depth in [5, 6, 7]:
            for leaf in [5, 8, 12]:
                tmp = self.random_forest_hierarchical(X_train_scaled, y_train, X_val_scaled, y_val, countries_train, countries_val, max_depth=depth, min_samples_leaf=leaf)
                if tmp['r2'] > best_val_r2:
                    best_val_r2 = tmp['r2']
                    best_params = (depth, leaf)
                    val_rf = tmp
        test_rf = self.random_forest_hierarchical(X_train_scaled, y_train, X_test_scaled, y_test, countries_train, countries_test, max_depth=best_params[0], min_samples_leaf=best_params[1])
        models['random_forest_hierarchical'] = {'val': val_rf, 'test': test_rf}
        
        # Análise dos gaps
        print(f"\n   RESULTADOS HIERÁRQUICOS (Val → Test):")
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
        
        return {
            'fold_id': fold_id,
            'architecture': 'data_lake',
            'mode': 'enhanced' if self.use_enhanced_features else 'normal',
            'total_features': len(self.available_features),
            'models': models
        }
    
    def run_hierarchical_analysis(self):
        """Executar análise hierárquica completa para arquitetura Data Lake."""
        mode_text = "ENHANCED" if self.use_enhanced_features else "NORMAL"
        print(f"Análise Hierárquica Completa Data Lake ({mode_text})")
        print("=" * 60)
        print("Arquitetura: Data Lake para ML Hierárquico")
        print("Objetivo: Avaliar capacidade hierárquica arquitetural")
        print("Pattern: Data Lake com processamento flexível")
        print("Configuração: Ridge α=50.0, Random Forest regularizado")
        
        all_results = {
            'architecture': 'data_lake',
            'version': 'hierarchical_analysis',
            'mode': 'enhanced' if self.use_enhanced_features else 'normal',
            'use_enhanced_features': self.use_enhanced_features,
            'target': self.target_col,
            'total_features': len(self.available_features),
            'corrections_applied': {
                'simple_hierarchical': 'Adaptativo: α adaptativo + Shrinkage + CV',
                'random_forest': 'Regularizado: n_est=200, depth=6, split=15, leaf=8',
                'regularization_approach': 'Adaptativo com Shrinkage e Cross-Validation'
            },
            'folds': []
        }
        
        valid_folds = 0
        for fold_info in self.folds:
            fold_results = self.run_fold_analysis(fold_info)
            if fold_results is not None:
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
        print(f"\nPERFORMANCE AGREGADA SCHEMA-ON-READ ({mode_text}):")
        print("INTERPRETAÇÃO GAPS HIERÁRQUICOS:")
        print("   • Gap ≤0.15: Excelente - objetivo das correções atingido")
        print("   • Gap 0.15-0.2: Bom - aceitável para dados educacionais") 
        print("   • Gap >0.2: Necessita ajustes adicionais")
        
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
                    print(f"      MELHORIA SIGNIFICATIVA - Gap aceitável")
                else:
                    print(f"      Necessita regularização adicional")
                
                all_results[f'{model_name}_summary'] = {
                    'val_mean_r2': float(val_mean),
                    'test_mean_r2': float(test_mean),
                    'test_std_r2': float(test_std),
                    'generalization_gap': float(gap_mean)
                }
        
        # Resumo executivo
        print(f"\nRESUMO EXECUTIVO - HIERÁRQUICO SCHEMA-ON-READ:")
        print(f"   PESQUISA: Modelos hierárquicos para arquitetura Data Lake")
        print(f"   Arquitetura: Data Lake (Processamento Distribuído)")
        print(f"   Pattern: Processamento flexível com Parquet")
        print(f"   Versão: Corrigida com regularização científica")
        
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
        
        mode_suffix = "_enhanced" if self.use_enhanced_features else "_normal"
        results_file = f"{self.results_path}/hierarchical_analysis_data_lake_results{mode_suffix}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResultados Data Lake ({mode_text}) salvos: {results_file}")
        
        return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Modelo Hierárquico Data Lake')
    parser.add_argument('--enhanced', action='store_true', 
                       help='Usar enhanced features do feature engineering')
    args = parser.parse_args()
    
    print(f"Modo selecionado: {'Enhanced Features' if args.enhanced else 'Folds Normais'}")
    model = HierarchicalModelDataLake(use_enhanced_features=args.enhanced)
    results = model.run_hierarchical_analysis()
    print(f"\nAnálise hierárquica Data Lake concluída!")
    print("   Regularização avançada aplicada com abordagem científica")
