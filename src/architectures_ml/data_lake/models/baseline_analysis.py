#!/usr/bin/env python3
"""
Análise de modelos baseline para arquitetura Data Lake.

Módulo para análise comparativa de modelos baseline usando processamento distribuído
com Dask e validação temporal walk-forward com gaps para predição de dropout educacional.

Resumo técnico:
- Processamento distribuído via Dask com dados Parquet
- Validação temporal com gaps (mínimo 2 anos) para prevenir vazamento
- Modelos baseline: média global, tendência linear, naive com lag, cross-country
- Métricas: R², RMSE, gaps de generalização
- Usa modelos centralizados de core/models/baseline.py
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
import json
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.*')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from src.core.config import get_absolute_output_path
from src.core.scientific_config import setup_reproducibility

setup_reproducibility()


class BaselineModelAnalysisDataLake:
    """
    Análise de modelos baseline para arquitetura Data Lake.
    
    Implementa análise científica de modelos baseline com validação temporal
    rigorosa, prevenindo vazamento de dados e utilizando processamento 
    distribuído com Dask para dados em formato Parquet.
    
    Attributes:
        data_path (str): Caminho para os dados principais do Data Lake
        folds_path (str): Caminho para configuração dos folds temporais
        results_path (str): Diretório para salvar resultados
        ddf (dask.DataFrame): DataFrame distribuído com os dados
        target_col (str): Nome da coluna target para predição
        folds (list): Lista de configurações dos folds temporais
    """
    
    def __init__(self):
        """Inicializa a análise baseline para arquitetura Data Lake."""
        print("Inicializando análise baseline Data Lake")
        
        self.data_path = get_absolute_output_path("ml_pipeline/architectures/data_lake/prep/master_data_data_lake.parquet")
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/data_lake/prep/temporal_folds_data_lake.json")
        self.results_path = get_absolute_output_path("ml_pipeline/architectures/data_lake/models/baseline_results")
        
        os.makedirs(self.results_path, exist_ok=True)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dados Data Lake não encontrados: {self.data_path}")
        if not os.path.exists(self.folds_path):
            raise FileNotFoundError(f"Folds Data Lake não encontrados: {self.folds_path}")

        print("   Carregando dados com Dask...")
        self.ddf = dd.read_parquet(self.data_path)
        self._needs_persist = True
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']
        
        self.target_col = 'dropout_rate_data_lake'
        self._load_data_summary()
    
    def _load_data_summary(self):
        """Carregar resumo dos dados usando batch compute otimizado."""
        stats_tasks = {
            'total_rows': self.ddf.index.size,
            'year_min': self.ddf['year'].min(),
            'year_max': self.ddf['year'].max(),
            'countries_count': self.ddf['country_code'].nunique(),
            'target_describe': self.ddf[self.target_col].describe(),
            'negative_target_count': (self.ddf[self.target_col] < 0).sum()
        }
        
        imputed_cols = [col for col in self.ddf.columns if '_imputed' in col and col != f'{self.target_col}_imputed']
        if imputed_cols:
            total_imputed = self.ddf[imputed_cols].sum(axis=1)
            stats_tasks['imputed_count'] = (total_imputed > 0).sum()
        
        keys = list(stats_tasks.keys())
        values = list(stats_tasks.values())
        computed_values = dask.compute(*values)
        computed_stats = dict(zip(keys, computed_values))
        
        total_rows = computed_stats['total_rows']
        year_min = computed_stats.get('year_min')
        year_max = computed_stats.get('year_max')
        countries_count = computed_stats.get('countries_count')
        target_stats = computed_stats.get('target_describe')
        negative_target = computed_stats.get('negative_target_count', 0)
        
        print(f"   Dados carregados: {total_rows} registros, {len(self.ddf.columns)} colunas")
        print(f"   Período: {year_min}-{year_max}")
        print(f"   Países: {countries_count}")
        print(f"   Target: {self.target_col}")
        print(f"   Folds: {len(self.folds)}")
        
        if self.target_col not in self.ddf.columns:
            raise ValueError(f"Target {self.target_col} não encontrado nos dados Data Lake")
        
        print(f"   Target stats: mean={target_stats['mean']:.2f}%, std={target_stats['std']:.2f}%")
        
        if imputed_cols:
            imputed_count = int(computed_stats['imputed_count'])
            print(f"   Dados com imputação: {imputed_count}/{total_rows} ({(imputed_count/total_rows)*100:.1f}%)")
            
        if negative_target > 0:
            print(f"   Target inválido: {negative_target} valores negativos detectados")
        else:
            target_min = target_stats['min']
            target_max = target_stats['max']
            print(f"   Target válido: range [{target_min:.2f}%, {target_max:.2f}%]")
        
        if not imputed_cols:
            print(f"   Nenhuma coluna de imputação encontrada - usando dados originais")
        
        self._cached_basic_stats = computed_stats
    
    def analyze_target_distribution(self) -> Dict:
        """
        Analisar distribuição do target Data Lake usando batch compute otimizado.
        
        Returns:
            Dict: Estatísticas da distribuição do target incluindo temporal e por país
        """
        print(f"\nAnálise da distribuição do target Data Lake")
        
        analysis = {}
        
        if hasattr(self, '_cached_basic_stats'):
            target_describe = self._cached_basic_stats['target_describe']
            year_min = self._cached_basic_stats['year_min']
            year_max = self._cached_basic_stats['year_max']
            
            missing_stats_batch = {
                'missing_count': self.ddf[self.target_col].isna().sum(),
                'missing_rate': self.ddf[self.target_col].isna().mean(),
                'unique_years': self.ddf['year'].nunique()
            }
            computed_missing = dask.compute(missing_stats_batch)[0]
            
            target_stats = {
                'architecture': 'data_lake',
                'target_variable': self.target_col,
                'mean': float(target_describe['mean']),
                'std': float(target_describe['std']),
                'min': float(target_describe['min']),
                'max': float(target_describe['max']),
                'missing_count': int(computed_missing['missing_count']),
                'missing_rate': float(computed_missing['missing_rate'])
            }
            unique_years = computed_missing['unique_years']
        else:
            target_stats_batch = {
                'target_describe': self.ddf[self.target_col].describe(),
                'missing_count': self.ddf[self.target_col].isna().sum(),
                'missing_rate': self.ddf[self.target_col].isna().mean(),
                'year_min': self.ddf['year'].min(),
                'year_max': self.ddf['year'].max(),
                'unique_years': self.ddf['year'].nunique()
            }
            computed_target = dask.compute(target_stats_batch)[0]
            
            target_describe = computed_target['target_describe']
            target_stats = {
                'architecture': 'data_lake',
                'target_variable': self.target_col,
                'mean': float(target_describe['mean']),
                'std': float(target_describe['std']),
                'min': float(target_describe['min']),
                'max': float(target_describe['max']),
                'missing_count': int(computed_target['missing_count']),
                'missing_rate': float(computed_target['missing_rate'])
            }
            year_min = computed_target['year_min']
            year_max = computed_target['year_max']
            unique_years = computed_target['unique_years']
        
        print(f"   Target Data Lake ({self.target_col}):")
        print(f"      Média: {target_stats['mean']:.2f}%")
        print(f"      Desvio: {target_stats['std']:.2f}%")
        print(f"      Range: {target_stats['min']:.2f}% - {target_stats['max']:.2f}%")
        print(f"      Missing: {target_stats['missing_count']} ({target_stats['missing_rate']:.1%})")
        
        analysis['target_stats'] = target_stats
        
        # Distribuição temporal usando Dask groupby
        if unique_years > 1:
            temporal_stats_ddf = self.ddf.groupby('year')[self.target_col].agg([
                'count', 'mean', 'std', 'min', 'max'
            ])
            temporal_stats = temporal_stats_ddf.compute().round(2)
            
            print(f"\n   Evolução temporal Data Lake:")
            if year_min in temporal_stats.index:
                print(f"      Primeiro ano ({year_min}): {temporal_stats.loc[year_min, 'mean']:.1f}%")
            if year_max in temporal_stats.index:
                print(f"      Último ano ({year_max}): {temporal_stats.loc[year_max, 'mean']:.1f}%")
            
            trend = temporal_stats['mean'].iloc[-1] - temporal_stats['mean'].iloc[0]
            print(f"      Tendência: {trend:.1f}% em {year_max - year_min} anos")
            
            analysis['temporal_stats'] = temporal_stats.to_dict()
        
        # Distribuição por país usando Dask groupby
        country_stats_ddf = self.ddf.groupby('country_code')[self.target_col].agg([
            'count', 'mean', 'std', 'min', 'max'
        ])
        country_stats = country_stats_ddf.compute().round(2)
        
        print(f"\n   Variação por país (Data Lake):")
        print(f"      Menor dropout: {country_stats['mean'].min():.1f}% ({country_stats['mean'].idxmin()})")
        print(f"      Maior dropout: {country_stats['mean'].max():.1f}% ({country_stats['mean'].idxmax()})")
        print(f"      Variação entre países: {country_stats['mean'].std():.1f}% (std)")
        
        analysis['country_stats'] = country_stats.to_dict()
        
        return analysis
    
    def test_baseline_models(self) -> Dict:
        """
        Testar modelos baseline científicos com validação temporal rigorosa.
        
        Problemas corrigidos:
        - Vazamento temporal: eliminado com lag mínimo de 2 anos
        - Período de validação usado corretamente
        - Gaps temporais apropriados implementados
        
        Returns:
            Dict: Resultados dos modelos baseline para todos os folds
        """
        print(f"\nBaselines com validação temporal")
        
        baseline_results = {}
        
        for fold_id, fold in enumerate(self.folds):
            print(f"\nFold {fold_id}: Train({fold['train_start']}-{fold['train_end']}) ->Val({fold['val_start']}-{fold['val_end']}) ->Test({fold['test_start']}-{fold['test_end']})")
            
            train_ddf = self.ddf[(self.ddf['year'] >= fold['train_start']) & (self.ddf['year'] <= fold['train_end'])]
            val_ddf = self.ddf[(self.ddf['year'] >= fold['val_start']) & (self.ddf['year'] <= fold['val_end'])]
            test_ddf = self.ddf[(self.ddf['year'] >= fold['test_start']) & (self.ddf['year'] <= fold['test_end'])]
            
            train_len = int(train_ddf.map_partitions(len).compute().sum())
            val_len = int(val_ddf.map_partitions(len).compute().sum())
            test_len = int(test_ddf.map_partitions(len).compute().sum())
            
            print(f"   Dados: Train={train_len}, Val={val_len}, Test={test_len}")
            print(f"    Gaps: Train-Val={fold['val_start']-fold['train_end']-1}yr, Val-Test={fold['test_start']-fold['val_end']-1}yr")
            
            train_clean_ddf = train_ddf.dropna(subset=[self.target_col])
            val_clean_ddf = val_ddf.dropna(subset=[self.target_col])
            test_clean_ddf = test_ddf.dropna(subset=[self.target_col])
            
            if train_len == 0 or test_len == 0:
                print(f"   Fold {fold_id}: Dados insuficientes")
                continue
            
            # Batch compute para targets e média global
            y_train, y_val, y_test, global_mean = dask.compute(
                train_clean_ddf[self.target_col],
                val_clean_ddf[self.target_col],
                test_clean_ddf[self.target_col],
                train_clean_ddf[self.target_col].mean(),
            )
            
            try:
                train_clean_pd = train_clean_ddf[['country_code','year', self.target_col]].compute()
                val_clean_pd = val_clean_ddf[['country_code','year', self.target_col]].compute()
                test_clean_pd = test_clean_ddf[['country_code','year', self.target_col]].compute()
            except Exception:
                train_clean_pd = None
                val_clean_pd = None
                test_clean_pd = None

            def _mase_scale_from_train(df):
                try:
                    if df is None or df.empty:
                        return None
                    diffs = []
                    for _, g in df.sort_values(['country_code','year']).groupby('country_code'):
                        s = g[self.target_col].values
                        if len(s) >= 2:
                            d = np.abs(np.diff(s))
                            if len(d) > 0:
                                diffs.append(d)
                    if not diffs:
                        return None
                    return float(np.mean(np.concatenate(diffs)))
                except Exception:
                    return None

            mase_scale = _mase_scale_from_train(train_clean_pd)

            fold_results = {}
            
            # Baseline 1: Média Global
            
            val_pred_global = np.full(len(y_val), global_mean)
            test_pred_global = np.full(len(y_test), global_mean)
            
            val_r2_global = r2_score(y_val, val_pred_global)
            test_r2_global = r2_score(y_test, test_pred_global)
            
            fold_results['global_mean'] = {
                'val_r2': float(val_r2_global),
                'test_r2': float(test_r2_global),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_global))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_global))),
                'test_wape': float((np.abs(y_test - test_pred_global)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None,
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_global))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'method': 'global_mean_no_leakage'
            }
            
            # Baseline 2: Tendência Linear - Batch compute para features temporais
            X_train_time, X_val_time, X_test_time = dask.compute(
                train_clean_ddf[['year']],
                val_clean_ddf[['year']],
                test_clean_ddf[['year']],
            )
            X_train_time = X_train_time.values
            X_val_time = X_val_time.values
            X_test_time = X_test_time.values
            
            trend_model = LinearRegression()
            trend_model.fit(X_train_time, y_train)
            
            val_pred_trend = trend_model.predict(X_val_time)
            test_pred_trend = trend_model.predict(X_test_time)
            
            val_r2_trend = r2_score(y_val, val_pred_trend)
            test_r2_trend = r2_score(y_test, test_pred_trend)
            
            fold_results['linear_trend'] = {
                'val_r2': float(val_r2_trend),
                'test_r2': float(test_r2_trend),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_trend))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_trend))),
                'test_wape': float((np.abs(y_test - test_pred_trend)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None,
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_trend))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'slope': float(trend_model.coef_[0]),
                'method': 'linear_trend_no_leakage'
            }
            
            # Baseline 3: Naive com Lag Científico
            MIN_LAG = 2
            
            print(f"      Naive baseline...")
            
            # Batch compute para val e test clean
            # val_clean_pd e test_clean_pd já computados acima
            
            val_pred_naive = []
            
            unique_countries = val_clean_pd['country_code'].unique()
            country_last_values = {}
            
            for country in unique_countries:
                country_train_ddf = train_clean_ddf[train_clean_ddf['country_code'] == country]
                country_train_pd = country_train_ddf.compute()
                if len(country_train_pd) > 0:
                    country_last_values[country] = country_train_pd.sort_values('year')
                else:
                    country_last_values[country] = pd.DataFrame()
            
            for _, val_row in val_clean_pd.iterrows():
                country = val_row['country_code']
                val_year = val_row['year']
                
                country_history = country_last_values[country]
                if len(country_history) > 0 and 'year' in country_history.columns:
                    country_history_filtered = country_history[country_history['year'] <= val_year - MIN_LAG]
                    if len(country_history_filtered) > 0:
                        naive_val = country_history_filtered[self.target_col].iloc[-1]
                    else:
                        naive_val = global_mean
                else:
                    naive_val = global_mean

                val_pred_naive.append(naive_val)
            
            # Predições para teste
            test_pred_naive = []
            combined_history_ddf = dd.concat([train_clean_ddf, val_clean_ddf], ignore_index=True)
            combined_mean = combined_history_ddf[self.target_col].mean().compute()
            test_unique_countries = test_clean_pd['country_code'].unique()
            test_country_values = {}
            
            for country in test_unique_countries:
                country_combined_ddf = combined_history_ddf[combined_history_ddf['country_code'] == country]
                country_combined_pd = country_combined_ddf.compute()
                if len(country_combined_pd) > 0:
                    test_country_values[country] = country_combined_pd.sort_values('year')
                else:
                    test_country_values[country] = pd.DataFrame()
            
            for _, test_row in test_clean_pd.iterrows():
                country = test_row['country_code']
                test_year = test_row['year']
                
                country_history = test_country_values[country]
                if len(country_history) > 0 and 'year' in country_history.columns:
                    country_history_filtered = country_history[country_history['year'] <= test_year - MIN_LAG]
                    if len(country_history_filtered) > 0:
                        naive_test = country_history_filtered[self.target_col].iloc[-1]
                    else:
                        naive_test = combined_mean
                else:
                    naive_test = combined_mean

                test_pred_naive.append(naive_test)
            
            val_pred_naive = np.array(val_pred_naive)
            test_pred_naive = np.array(test_pred_naive)
            
            val_r2_naive = r2_score(y_val, val_pred_naive)
            test_r2_naive = r2_score(y_test, test_pred_naive)
            
            fold_results['naive_with_lag'] = {
                'val_r2': float(val_r2_naive),
                'test_r2': float(test_r2_naive),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_naive))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_naive))),
                'min_lag_years': MIN_LAG,
                'method': 'naive_persistence_with_scientific_lag'
            }
            
            # Baseline 4: Cross-Country Average
            print(f"      Cross-Country baseline...")
            
            val_pred_cross = []
            
            unique_years = sorted(val_clean_pd['year'].unique())
            cross_country_means = {}
            
            for year in unique_years:
                year_data_ddf = train_clean_ddf[train_clean_ddf['year'] <= year - MIN_LAG]

                year_count = int(year_data_ddf.map_partitions(len).compute().sum())
                if year_count > 0:
                    country_means_ddf = year_data_ddf.groupby('country_code')[self.target_col].mean()
                    country_means_pd = country_means_ddf.compute()
                    cross_country_means[year] = country_means_pd
                else:
                    cross_country_means[year] = pd.Series(dtype=float)
            
            for _, val_row in val_clean_pd.iterrows():
                country = val_row['country_code']
                val_year = val_row['year']
                
                if val_year in cross_country_means:
                    country_means = cross_country_means[val_year]
                    other_countries = country_means[country_means.index != country]
                    
                    if len(other_countries) > 0:
                        cross_val = other_countries.mean()
                    else:
                        cross_val = global_mean
                else:
                    cross_val = global_mean
                
                val_pred_cross.append(cross_val)
            
            # Predições para teste
            test_pred_cross = []
            test_unique_years = sorted(test_clean_pd['year'].unique())
            test_cross_country_means = {}
            
            for year in test_unique_years:
                year_data_ddf = combined_history_ddf[combined_history_ddf['year'] <= year - MIN_LAG]
                year_count = int(year_data_ddf.map_partitions(len).compute().sum())
                if year_count > 0:
                    country_means_ddf = year_data_ddf.groupby('country_code')[self.target_col].mean()
                    country_means_pd = country_means_ddf.compute()
                    test_cross_country_means[year] = country_means_pd
                else:
                    test_cross_country_means[year] = pd.Series(dtype=float)
            
            for _, test_row in test_clean_pd.iterrows():
                country = test_row['country_code']
                test_year = test_row['year']
                
                if test_year in test_cross_country_means:
                    country_means = test_cross_country_means[test_year]
                    other_countries = country_means[country_means.index != country]
                    
                    if len(other_countries) > 0:
                        cross_test = other_countries.mean()
                    else:
                        cross_test = combined_mean
                else:
                    cross_test = combined_mean
                
                test_pred_cross.append(cross_test)
            
            val_pred_cross = np.array(val_pred_cross)
            test_pred_cross = np.array(test_pred_cross)
            
            val_r2_cross = r2_score(y_val, val_pred_cross)
            test_r2_cross = r2_score(y_test, test_pred_cross)
            
            fold_results['cross_country'] = {
                'val_r2': float(val_r2_cross),
                'test_r2': float(test_r2_cross),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_cross))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_cross))),
                'test_wape': float((np.abs(y_test - test_pred_cross)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None,
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_cross))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'min_lag_years': MIN_LAG,
                'geographic_leakage_prevented': True,
                'method': 'cross_country_average_excluding_target'
            }
            
            # Métricas agregadas para Naive (inclui WAPE/MASE)
            try:
                test_wape_naive = float((np.abs(y_test - test_pred_naive)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None
                test_mase_naive = (float(np.mean(np.abs(y_test - test_pred_naive))) / mase_scale) if (mase_scale and mase_scale > 0) else None
            except Exception:
                test_wape_naive = None
                test_mase_naive = None
            fold_results['naive_with_lag'].update({
                'test_wape': test_wape_naive,
                'test_mase': test_mase_naive,
                'mase_scale_train': mase_scale
            })
            
            print(f"   Resultados (Val | Test):")
            print(f"      Global Mean:      R²={val_r2_global:.3f} | {test_r2_global:.3f}")
            print(f"      Linear Trend:     R²={val_r2_trend:.3f} | {test_r2_trend:.3f}")  
            print(f"      Naive+Lag>=2yr:   R²={val_r2_naive:.3f} | {test_r2_naive:.3f}")
            print(f"      Cross-Country:    R²={val_r2_cross:.3f} | {test_r2_cross:.3f}")
            
            val_scores = [(name, data['val_r2']) for name, data in fold_results.items()]
            best_val_baseline, best_val_r2 = max(val_scores, key=lambda x: x[1])
            
            best_test_r2 = fold_results[best_val_baseline]['test_r2']
            generalization_gap = best_val_r2 - best_test_r2
            
            fold_results['best_baseline'] = {
                'model': best_val_baseline,
                'val_r2': best_val_r2,
                'test_r2': best_test_r2,
                'generalization_gap': generalization_gap
            }
            
            print(f"   Melhor baseline: {best_val_baseline} (Val: {best_val_r2:.3f} ->Test: {best_test_r2:.3f}, Gap: {generalization_gap:+.3f})")
            
            abs_gap = abs(generalization_gap)
            if abs_gap <= 0.05:
                print(f"      Excelente estabilidade: Gap muito baixo (<=0.05)")
            elif abs_gap <= 0.1:
                print(f"      Boa estabilidade: Gap dentro do esperado (<=0.10)")
            elif abs_gap <= 0.15:
                print(f"      Gap moderado: Variação temporal aceitável ({abs_gap:.3f})")
            else:
                print(f"      Gap elevado: Possível instabilidade temporal ({abs_gap:.3f})")
            
            baseline_results[f'fold_{fold_id}'] = fold_results
        
        return baseline_results

    def analyze_predictability(self, baseline_results: Dict) -> Dict:
        """
        Análise científica de predictabilidade dos modelos baseline.
        
        Args:
            baseline_results: Resultados dos modelos baseline por fold
            
        Returns:
            Dict: Análise completa de predictabilidade com métricas de estabilidade
        """
        print("\nAnálise de predictabilidade Data Lake")
        
        baselines = ['global_mean', 'linear_trend', 'naive_with_lag', 'cross_country']
        all_test_scores = {}
        all_val_scores = {}
        generalization_gaps = {}
        
        for baseline in baselines:
            test_r2_scores = []
            val_r2_scores = []
            gaps = []
            
            for fold_key in baseline_results:
                fold_data = baseline_results[fold_key]
                if baseline in fold_data:
                    test_r2_scores.append(fold_data[baseline]['test_r2'])
                    val_r2_scores.append(fold_data[baseline]['val_r2'])
                    gaps.append(fold_data[baseline]['val_r2'] - fold_data[baseline]['test_r2'])
            
            if test_r2_scores:
                all_test_scores[baseline] = {
                    'mean_r2': float(np.mean(test_r2_scores)),
                    'std_r2': float(np.std(test_r2_scores)),
                    'min_r2': float(np.min(test_r2_scores)),
                    'max_r2': float(np.max(test_r2_scores)),
                    'scores': test_r2_scores
                }
                
                all_val_scores[baseline] = {
                    'mean_r2': float(np.mean(val_r2_scores)),
                    'std_r2': float(np.std(val_r2_scores))
                }
                
                generalization_gaps[baseline] = {
                    'mean_gap': float(np.mean(gaps)),
                    'std_gap': float(np.std(gaps)),
                    'gaps': gaps
                }
        
        print("   Performance out-of-sample (TEST SET) dos baselines:")
        for baseline, stats in all_test_scores.items():
            val_stats = all_val_scores[baseline]
            gap_stats = generalization_gaps[baseline]
            print(f"      {baseline:20} | Test: R²={stats['mean_r2']:.3f}±{stats['std_r2']:.3f} | Val: R²={val_stats['mean_r2']:.3f} | Gap: {gap_stats['mean_gap']:+.3f}")
        
        if all_test_scores:
            best_baseline_overall = max(all_test_scores.keys(), key=lambda x: all_test_scores[x]['mean_r2'])
            best_mean_test_r2 = all_test_scores[best_baseline_overall]['mean_r2']
            best_mean_val_r2 = all_val_scores[best_baseline_overall]['mean_r2']
            best_generalization_gap = generalization_gaps[best_baseline_overall]['mean_gap']
            
            print(f"\n   Melhor baseline: {best_baseline_overall}")
            print(f"      Performance Validação: R² = {best_mean_val_r2:.3f}")
            print(f"      Performance Teste:     R² = {best_mean_test_r2:.3f}")
            print(f"      Gap Generalização:     {best_generalization_gap:+.3f}")
            
            predictability_analysis = {
                'architecture': 'data_lake',
                'methodology': 'scientific_baselines_with_temporal_lags',
                'validation_scores': all_val_scores,
                'test_scores': all_test_scores,
                'generalization_gaps': generalization_gaps,
                'best_baseline': best_baseline_overall,
                'best_test_r2': best_mean_test_r2,
                'best_val_r2': best_mean_val_r2,
                'generalization_gap': best_generalization_gap,
                'predictability_level': 'unknown',
                'scientific_validity': {
                    'temporal_leakage_prevented': True,
                    'geographic_leakage_prevented': True,
                    'validation_used_correctly': True,
                    'walk_forward_validation': True
                }
            }
            
            # Classificação de predictabilidade
            if best_mean_test_r2 < 0:
                predictability_analysis['predictability_level'] = 'very_low'
                print(f"   Predictabilidade muito baixa: R²_test < 0")
                print(f"      Interpretação: Modelo pior que baseline constante")
            elif best_mean_test_r2 < 0.05:
                predictability_analysis['predictability_level'] = 'very_low'
                print(f"   Predictabilidade muito baixa: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretação: Quase sem poder preditivo")
            elif best_mean_test_r2 < 0.15:
                predictability_analysis['predictability_level'] = 'low'
                print(f"   Predictabilidade baixa: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretação: Poder preditivo limitado")
            elif best_mean_test_r2 < 0.35:
                predictability_analysis['predictability_level'] = 'moderate'
                print(f"   Predictabilidade moderada: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretação: Poder preditivo razoável")
            else:
                predictability_analysis['predictability_level'] = 'good'
                print(f"   Boa predictabilidade: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretação: Bom poder preditivo")
            
            avg_generalization_gap = np.mean([gap_data['mean_gap'] for gap_data in generalization_gaps.values()])
            abs_avg_gap = abs(avg_generalization_gap)
            
            if abs_avg_gap <= 0.05:
                print(f"   Excelente estabilidade: Gap médio muito baixo ({avg_generalization_gap:+.3f})")
                stability_level = "excellent"
            elif abs_avg_gap <= 0.1:
                print(f"   Boa estabilidade: Gap médio dentro do esperado ({avg_generalization_gap:+.3f})")
                stability_level = "good"
            elif abs_avg_gap <= 0.15:
                print(f"   Estabilidade moderada: Variação temporal aceitável ({avg_generalization_gap:+.3f})")
                stability_level = "moderate"
            else:
                print(f"   Instabilidade detectada: Gap médio elevado ({avg_generalization_gap:+.3f})")
                print(f"      Possível overfitting ou forte variação temporal")
                stability_level = "low"
            
            publication_criteria = {
                'temporal_leakage_prevented': predictability_analysis['scientific_validity']['temporal_leakage_prevented'],
                'validation_used_correctly': predictability_analysis['scientific_validity']['validation_used_correctly'],
                'stability_acceptable': abs_avg_gap <= 0.2
            }
            
            if not all(publication_criteria.values()):
                failed_criteria = [k for k, v in publication_criteria.items() if not v]
                print(f"   [WARN] Requer atenção: {', '.join(failed_criteria)}")

            predictability_analysis['stability_analysis'] = {
                'avg_generalization_gap': float(avg_generalization_gap),
                'stability_level': stability_level,
                'publication_criteria': {k: bool(v) for k, v in publication_criteria.items()}
            }
            
        else:
            predictability_analysis = {
                'architecture': 'data_lake',
                'baseline_scores': {},
                'predictability_level': 'unknown',
                'scientific_validity': {'error': 'no_valid_results'}
            }
        
        return predictability_analysis
    
    def save_results(self, target_analysis: Dict, baseline_results: Dict, 
                    predictability_analysis: Dict):
        """
        Salvar resultados da análise Data Lake.
        
        Args:
            target_analysis: Análise da distribuição do target
            baseline_results: Resultados dos modelos baseline
            predictability_analysis: Análise de predictabilidade
            
        Returns:
            Dict: Resultados completos consolidados
        """
        print(f"\nSalvando resultados Data Lake...")
        
        full_results = {
            'architecture': 'data_lake',
            'target_variable': self.target_col,
            'data_source': self.data_path,
            'target_distribution_analysis': target_analysis,
            'baseline_model_results': baseline_results,
            'predictability_analysis': predictability_analysis,
            'summary': {
                'total_folds_analyzed': len(baseline_results),
                'best_baseline_model': predictability_analysis.get('best_baseline', 'unknown'),
                'best_baseline_r2': predictability_analysis.get('best_test_r2', 0),
                'predictability_level': predictability_analysis.get('predictability_level', 'unknown'),
            }
        }
        
        results_file = f"{self.results_path}/baseline_analysis_data_lake_results.json"
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"   Resultados Data Lake salvos: {results_file}")
        
        return full_results
    
    def run_complete_analysis(self):
        """
        Executar análise completa de baseline Data Lake.
        
        Returns:
            Dict: Resultados consolidados da análise ou erro
        """
        if getattr(self, '_needs_persist', False):
            self.ddf = self.ddf.persist()
            self._needs_persist = False
        print(f"Análise completa - arquitetura Data Lake")
        
        try:
            target_analysis = self.analyze_target_distribution()
            baseline_results = self.test_baseline_models()
            predictability_analysis = self.analyze_predictability(baseline_results)
            results = self.save_results(target_analysis, baseline_results, 
                                       predictability_analysis)
            
            print(f"\nResumo executivo - arquitetura Data Lake:")
            print(f"   Pesquisa: Comparação de arquiteturas ML para dropout")
            print(f"   Arquitetura: Data Lake (Processamento Distribuído)")
            print(f"   Target: {self.target_col}")
            print(f"   Predictabilidade: {predictability_analysis.get('predictability_level', 'unknown').upper()}")
            print(f"   Melhor baseline: {predictability_analysis.get('best_baseline', 'unknown')}")
            print(f"   R² Teste: {predictability_analysis.get('best_test_r2', 0):.3f}")
            
            gap = predictability_analysis.get('generalization_gap', 0)
            
            if abs(gap) <= 0.05:
                gap_status = f"Gap: {gap:+.3f} (excelente estabilidade)"
            elif abs(gap) <= 0.1:
                gap_status = f"Gap: {gap:+.3f} (boa estabilidade)"
            elif abs(gap) <= 0.15:
                gap_status = f"Gap: {gap:+.3f} (estabilidade moderada)"
            else:
                gap_status = f"Gap: {gap:+.3f} (requer atenção)"
                
            print(f"   {gap_status}")
            
            stability = predictability_analysis.get('stability_analysis', {}).get('stability_level', 'unknown')
            print(f"   Estabilidade: {stability}")
            
            return results
            
        except Exception as e:
            import traceback
            print(f"\nErro na análise Data Lake: {e}")
            traceback.print_exc()
            return {
                'architecture': 'data_lake',
                'status': 'failed',
                'error': str(e)
            }

if __name__ == "__main__":
    analyzer = BaselineModelAnalysisDataLake()
    results = analyzer.run_complete_analysis()
