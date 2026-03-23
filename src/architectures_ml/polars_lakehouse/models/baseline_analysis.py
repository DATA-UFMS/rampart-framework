#!/usr/bin/env python3
"""
Análise de modelos baseline para arquitetura Polars Lakehouse.

Módulo para análise comparativa de modelos baseline usando leitura lazy com Polars
e validação temporal walk-forward com gaps para predição de dropout educacional.

Resumo técnico:
- Leitura lazy via Polars com dados Parquet
- Validação temporal com gaps (mínimo 2 anos) para prevenir vazamento
- Modelos baseline: média global, tendência linear, naive com lag, cross-country
- Métricas: R², RMSE, gaps de generalização
- Usa modelos centralizados de core/models/baseline.py
"""

import pandas as pd
import polars as pl
import numpy as np
import json
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Adicionar path para configuração
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from src.core.config import get_absolute_output_path
from src.core.models.baseline import BaselineModelFactory, BaselineEnsemble
from src.core.scientific_config import RANDOM_SEED, setup_reproducibility

setup_reproducibility()


class BaselineModelAnalysisPolarsLakehouse:
    """
    Análise de modelos baseline para arquitetura Polars Lakehouse.

    Implementa análise científica de modelos baseline com validação temporal
    rigorosa, prevenindo vazamento de dados e utilizando leitura lazy com Polars
    para dados em formato Parquet.

    Attributes:
        data_path (str): Caminho para os dados principais do Polars Lakehouse
        folds_path (str): Caminho para configuração dos folds temporais
        results_path (str): Diretório para salvar resultados
        df_lazy (pl.LazyFrame): DataFrame Polars lazy com os dados
        target_col (str): Nome da coluna target para predição
        folds (list): Lista de configurações dos folds temporais
    """

    def __init__(self):
        """
        Inicializa a análise baseline para arquitetura Polars Lakehouse.

        Docstring em Português conforme padrão do projeto.
        """
        print("Inicializando análise baseline Polars Lakehouse")
        print("=" * 60)
        print("Pesquisa: Comparação de arquiteturas ML para evasão educacional")

        self.data_path = get_absolute_output_path("ml_pipeline/architectures/polars_lakehouse/prep/master_data_polars_lakehouse.parquet")
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/polars_lakehouse/prep/temporal_folds_polars_lakehouse.json")
        self.results_path = get_absolute_output_path("ml_pipeline/architectures/polars_lakehouse/models/baseline_results")

        os.makedirs(self.results_path, exist_ok=True)

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dados Polars Lakehouse não encontrados: {self.data_path}")
        if not os.path.exists(self.folds_path):
            raise FileNotFoundError(f"Folds Polars Lakehouse não encontrados: {self.folds_path}")

        print("Carregando dados Polars Lakehouse com lazy evaluation...")
        self.df_lazy = pl.scan_parquet(self.data_path)
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']

        self.target_col = 'dropout_rate_polars_lakehouse'
        self._load_data_summary()

    def _load_data_summary(self):
        """
        Carrega resumo dos dados usando computação seletiva com Polars.

        Docstring em Português.
        """
        print("[RESUMO] Computando estatísticas dos dados...")

        # Computar estatísticas críticas
        stats_df = self.df_lazy.select([
            pl.len().alias('total_rows'),
            pl.col('year').min().alias('year_min'),
            pl.col('year').max().alias('year_max'),
            pl.col('country_code').n_unique().alias('n_countries'),
            pl.col(self.target_col).mean().alias('target_mean'),
            pl.col(self.target_col).std().alias('target_std'),
            pl.col(self.target_col).min().alias('target_min'),
            pl.col(self.target_col).max().alias('target_max'),
            (pl.col(self.target_col).is_null().sum()).alias('target_missing'),
            (pl.col(self.target_col) < 0).sum().alias('negative_target')
        ]).collect()

        stats = stats_df.to_dicts()[0]

        total_rows = stats['total_rows']
        year_min = stats['year_min']
        year_max = stats['year_max']
        n_countries = stats['n_countries']

        print(f"   Dados carregados: {total_rows} registros, {len(self.df_lazy.collect_schema().names())} colunas")
        print(f"   Período: {year_min}-{year_max}")
        print(f"   Países: {n_countries}")
        print(f"   Target: {self.target_col}")
        print(f"   Folds: {len(self.folds)}")

        if self.target_col not in self.df_lazy.collect_schema().names():
            raise ValueError(f"Target {self.target_col} não encontrado nos dados Polars Lakehouse")

        print(f"   Target stats: μ={stats['target_mean']:.2f}%, σ={stats['target_std']:.2f}%")

        if stats['negative_target'] > 0:
            print(f"   Aviso: {stats['negative_target']} valores negativos no target")
        else:
            print(f"   Target válido: range [{stats['target_min']:.2f}%, {stats['target_max']:.2f}%]")

        self._cached_basic_stats = stats

    def analyze_target_distribution(self) -> Dict:
        """
        Analisar distribuição do target Polars Lakehouse.

        Returns:
            Dict: Estatísticas da distribuição do target

        Docstring em Português.
        """
        print(f"\nAnálise da distribuição do target Polars Lakehouse")
        print("=" * 50)

        analysis = {}

        if hasattr(self, '_cached_basic_stats'):
            stats = self._cached_basic_stats

            target_stats = {
                'architecture': 'polars_lakehouse',
                'target_variable': self.target_col,
                'mean': float(stats['target_mean']),
                'std': float(stats['target_std']),
                'min': float(stats['target_min']),
                'max': float(stats['target_max']),
                'missing_count': int(stats['target_missing']),
                'missing_rate': float(stats['target_missing'] / stats['total_rows'])
            }

            year_min = stats['year_min']
            year_max = stats['year_max']
            unique_years = year_max - year_min + 1
        else:
            # Fallback: computar novamente
            stats_df = self.df_lazy.select([
                pl.col(self.target_col).describe()
            ]).collect()

            target_stats = {
                'architecture': 'polars_lakehouse',
                'target_variable': self.target_col
            }
            unique_years = self.df_lazy.select(pl.col('year').n_unique()).collect()[0, 0]

        print(f"   Target Polars Lakehouse ({self.target_col}):")
        print(f"      Média: {target_stats['mean']:.2f}%")
        print(f"      Desvio: {target_stats['std']:.2f}%")
        print(f"      Range: {target_stats['min']:.2f}% - {target_stats['max']:.2f}%")
        print(f"      Missing: {target_stats['missing_count']} ({target_stats['missing_rate']:.1%})")

        analysis['target_stats'] = target_stats

        # Distribuição temporal
        if unique_years > 1:
            temporal_df = self.df_lazy.group_by('year').agg([
                pl.col(self.target_col).count().alias('count'),
                pl.col(self.target_col).mean().alias('mean'),
                pl.col(self.target_col).std().alias('std'),
                pl.col(self.target_col).min().alias('min'),
                pl.col(self.target_col).max().alias('max')
            ]).sort('year').collect()

            print(f"\n   Evolução temporal Polars Lakehouse:")
            if len(temporal_df) > 0:
                first_year_mean = temporal_df[0, 'mean']
                last_year_mean = temporal_df[-1, 'mean']
                print(f"      Primeiro ano: {first_year_mean:.1f}%")
                print(f"      Último ano: {last_year_mean:.1f}%")
                trend = last_year_mean - first_year_mean
                print(f"      Tendência: {trend:.1f}% em {unique_years} anos")

            analysis['temporal_stats'] = temporal_df.to_dicts()

        # Distribuição por país
        country_df = self.df_lazy.group_by('country_code').agg([
            pl.col(self.target_col).count().alias('count'),
            pl.col(self.target_col).mean().alias('mean'),
            pl.col(self.target_col).std().alias('std'),
            pl.col(self.target_col).min().alias('min'),
            pl.col(self.target_col).max().alias('max')
        ]).sort('mean', descending=True).collect()

        print(f"\n   Variação por país (Polars Lakehouse):")
        if len(country_df) > 0:
            print(f"      Menor dropout: {country_df[-1, 'mean']:.1f}% ({country_df[-1, 'country_code']})")
            print(f"      Maior dropout: {country_df[0, 'mean']:.1f}% ({country_df[0, 'country_code']})")
            country_means = country_df['mean'].to_list()
            print(f"      Variação entre países: {np.std(country_means):.1f}% (std)")

        analysis['country_stats'] = country_df.to_dicts()

        return analysis

    def test_baseline_models(self) -> Dict:
        """
        Testar modelos baseline científicos com validação temporal rigorosa.

        Returns:
            Dict: Resultados dos modelos baseline para todos os folds

        Docstring em Português.
        """
        print(f"\nBaselines com validação temporal")
        print("=" * 60)
        print("Metodologia: Vazamento temporal eliminado com lag >= 2 anos")
        print("Validação: Walk-forward temporal com gaps")

        baseline_results = {}

        for fold_id, fold in enumerate(self.folds):
            print(f"\nFold {fold_id}: Train({fold['train_start']}-{fold['train_end']}) → "
                  f"Val({fold['val_start']}-{fold['val_end']}) → "
                  f"Test({fold['test_start']}-{fold['test_end']})")

            # Filtros lazy
            train_lazy = self.df_lazy.filter(
                (pl.col('year') >= fold['train_start']) & (pl.col('year') <= fold['train_end'])
            )
            val_lazy = self.df_lazy.filter(
                (pl.col('year') >= fold['val_start']) & (pl.col('year') <= fold['val_end'])
            )
            test_lazy = self.df_lazy.filter(
                (pl.col('year') >= fold['test_start']) & (pl.col('year') <= fold['test_end'])
            )

            # Materializar para operações pandas
            train_df = train_lazy.collect().to_pandas()
            val_df = val_lazy.collect().to_pandas()
            test_df = test_lazy.collect().to_pandas()

            train_len = len(train_df)
            val_len = len(val_df)
            test_len = len(test_df)

            print(f"   Dados: Train={train_len}, Val={val_len}, Test={test_len}")
            print(f"   Gaps: Train-Val={fold['val_start']-fold['train_end']-1}yr, "
                  f"Val-Test={fold['test_start']-fold['val_end']-1}yr")

            # Limpeza
            train_clean = train_df.dropna(subset=[self.target_col])
            val_clean = val_df.dropna(subset=[self.target_col])
            test_clean = test_df.dropna(subset=[self.target_col])

            if len(train_clean) == 0 or len(test_clean) == 0:
                print(f"   Fold {fold_id}: Dados insuficientes")
                continue

            y_train = train_clean[self.target_col].values
            y_val = val_clean[self.target_col].values
            y_test = test_clean[self.target_col].values
            global_mean = y_train.mean()

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
                'method': 'global_mean_no_leakage'
            }

            # Baseline 2: Tendência Linear
            X_train_time = train_clean[['year']].values
            X_val_time = val_clean[['year']].values
            X_test_time = test_clean[['year']].values

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
                'slope': float(trend_model.coef_[0]),
                'method': 'linear_trend_no_leakage'
            }

            # Baseline 3: Naive com Lag Científico
            MIN_LAG = 2

            val_pred_naive = []
            for _, row in val_clean.iterrows():
                country = row['country_code']
                val_year = row['year']

                country_train = train_clean[train_clean['country_code'] == country]
                country_hist = country_train[country_train['year'] <= val_year - MIN_LAG]

                if len(country_hist) > 0:
                    naive_val = country_hist.sort_values('year').iloc[-1][self.target_col]
                else:
                    naive_val = global_mean

                val_pred_naive.append(naive_val)

            test_pred_naive = []
            combined_clean = pd.concat([train_clean, val_clean], ignore_index=True)
            combined_mean = combined_clean[self.target_col].mean()

            for _, row in test_clean.iterrows():
                country = row['country_code']
                test_year = row['year']

                country_combined = combined_clean[combined_clean['country_code'] == country]
                country_hist = country_combined[country_combined['year'] <= test_year - MIN_LAG]

                if len(country_hist) > 0:
                    naive_test = country_hist.sort_values('year').iloc[-1][self.target_col]
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
            val_pred_cross = []
            for _, row in val_clean.iterrows():
                country = row['country_code']
                val_year = row['year']

                country_train = train_clean[train_clean['country_code'] == country]
                year_data = train_clean[train_clean['year'] <= val_year - MIN_LAG]

                if len(year_data) > 0:
                    country_means_dict = year_data.groupby('country_code')[self.target_col].mean()
                    other_countries = country_means_dict[country_means_dict.index != country]

                    if len(other_countries) > 0:
                        cross_val = other_countries.mean()
                    else:
                        cross_val = global_mean
                else:
                    cross_val = global_mean

                val_pred_cross.append(cross_val)

            test_pred_cross = []
            for _, row in test_clean.iterrows():
                country = row['country_code']
                test_year = row['year']

                year_data = combined_clean[combined_clean['year'] <= test_year - MIN_LAG]

                if len(year_data) > 0:
                    country_means_dict = year_data.groupby('country_code')[self.target_col].mean()
                    other_countries = country_means_dict[country_means_dict.index != country]

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

            fold_results['cross_country_average'] = {
                'val_r2': float(val_r2_cross),
                'test_r2': float(test_r2_cross),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_cross))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_cross))),
                'method': 'cross_country_average_with_lag'
            }

            # Registrar resultados do fold
            baseline_results[f'fold_{fold_id}'] = fold_results

        return baseline_results

    def run_complete_analysis(self) -> Dict:
        """
        Executar análise baseline completa para arquitetura Polars Lakehouse.

        Returns:
            Dict contendo resultados completos de baseline

        Docstring em Português.
        """
        print("Análise Baseline Completa Polars Lakehouse")
        print("=" * 60)
        print("Arquitetura: Polars Lakehouse para ML com Lazy Evaluation")
        print("Objetivo: Avaliar performance de baselines científicos")
        print("Pattern: Leitura lazy com Polars, sklearn para modelos")
        print("Configuração: Sem vazamento temporal, gaps de 2 anos")

        all_results = {
            'architecture': 'polars_lakehouse',
            'version': 'baseline_analysis',
            'target': self.target_col,
            'target_analysis': self.analyze_target_distribution(),
            'baseline_models': self.test_baseline_models()
        }

        # Análise de performance agregada
        print("\nPERFORMANCE AGREGADA POLARS LAKEHOUSE:")

        for model_name in ['global_mean', 'linear_trend', 'naive_with_lag', 'cross_country_average']:
            test_r2s = []

            for fold_key, fold_results in all_results['baseline_models'].items():
                if model_name in fold_results:
                    test_r2s.append(fold_results[model_name]['test_r2'])

            if len(test_r2s) > 0:
                mean_r2 = np.mean(test_r2s)
                std_r2 = np.std(test_r2s)
                print(f"\n   {model_name}:")
                print(f"      Test R² = {mean_r2:.3f} ± {std_r2:.3f}")

                all_results[f'{model_name}_summary'] = {
                    'mean_test_r2': float(mean_r2),
                    'std_test_r2': float(std_r2)
                }

        # Salvar resultados
        results_file = f"{self.results_path}/baseline_analysis_polars_lakehouse_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResultados Polars Lakehouse salvos: {results_file}")

        return all_results


if __name__ == "__main__":
    model = BaselineModelAnalysisPolarsLakehouse()
    results = model.run_complete_analysis()
    print("\nAnálise baseline Polars Lakehouse concluída!")
