#!/usr/bin/env python3
"""
Análise de modelos baseline para arquitetura Polars DataFrame.

Módulo para análise comparativa de modelos baseline usando leitura lazy com Polars
e validação temporal walk-forward com gaps para predição de dropout educacional.

Resumo técnico:
- Leitura lazy via Polars com dados Parquet
- Validação temporal com gaps (mínimo 2 anos) para prevenir vazamento
- Modelos baseline: média global, tendência linear, naive com lag, cross-country
- Métricas: R², RMSE, gaps de generalização
"""
import time

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
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.*')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from core.config import get_absolute_output_path
from core.prediction_store import PredictionRecorder, predictions_path
from core.scientific_config import SCIENTIFIC_CONFIG, setup_reproducibility

setup_reproducibility()


class BaselineModelAnalysisDataFrameLib:
    """
    Análise de modelos baseline para arquitetura Polars DataFrame.

    Implementa análise científica de modelos baseline com validação temporal,
    prevenindo vazamento de dados e utilizando leitura lazy com Polars para
    dados em formato Parquet.

    Attributes:
        data_path (str): Caminho para os dados principais do Polars DataFrame
        folds_path (str): Caminho para configuração dos folds temporais
        results_path (str): Diretório para salvar resultados
        df_lazy (pl.LazyFrame): DataFrame Polars lazy com os dados
        target_col (str): Nome da coluna target para predição
        folds (list): Lista de configurações dos folds temporais
    """

    def __init__(self):
        self._prediction_recorder = PredictionRecorder('dataframe_lib')
        """
        Inicializa a análise baseline para arquitetura Polars DataFrame.
        """
        print("Inicializando análise baseline Polars")

        self.data_path = get_absolute_output_path("ml_pipeline/architectures/dataframe_lib/prep/master_data_dataframe_lib.parquet")
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/dataframe_lib/prep/temporal_folds_dataframe_lib.json")
        self.results_path = get_absolute_output_path("ml_pipeline/architectures/dataframe_lib/models/baseline_results")

        os.makedirs(self.results_path, exist_ok=True)

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dados Polars DataFrame não encontrados: {self.data_path}")
        if not os.path.exists(self.folds_path):
            raise FileNotFoundError(f"Folds Polars DataFrame não encontrados: {self.folds_path}")

        print("   Carregando dados com LAZY EVALUATION...")
        self.df_lazy = pl.scan_parquet(self.data_path)
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']

        self.target_col = 'dropout_rate_dataframe_lib'
        self._load_data_summary()

    def _load_data_summary(self):
        """
        Carrega resumo dos dados usando computação seletiva com Polars.
        """
        print("   Computando estatísticas...")

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
            raise ValueError(f"Target {self.target_col} não encontrado nos dados Polars DataFrame")

        print(f"   Target stats: mean={stats['target_mean']:.2f}%, std={stats['target_std']:.2f}%")

        if stats['negative_target'] > 0:
            print(f"   Aviso: {stats['negative_target']} valores negativos no target")
        else:
            print(f"   Target válido: range [{stats['target_min']:.2f}%, {stats['target_max']:.2f}%]")

        self._cached_basic_stats = stats

    def analyze_target_distribution(self) -> Dict:
        """
        Analisar distribuição do target Polars DataFrame.

        Returns:
            Dict: Estatísticas da distribuição do target
        """
        print(f"\nAnálise da distribuição do target Polars")

        analysis = {}

        if hasattr(self, '_cached_basic_stats'):
            stats = self._cached_basic_stats

            target_stats = {
                'architecture': 'dataframe_lib',
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
                'architecture': 'dataframe_lib',
                'target_variable': self.target_col
            }
            unique_years = self.df_lazy.select(pl.col('year').n_unique()).collect()[0, 0]

        print(f"   Target Polars ({self.target_col}):")
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

            print(f"\n   Evolução temporal Polars:")
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

        print(f"\n   Variação por país (Polars):")
        if len(country_df) > 0:
            print(f"      Menor dropout: {country_df[-1, 'mean']:.1f}% ({country_df[-1, 'country_code']})")
            print(f"      Maior dropout: {country_df[0, 'mean']:.1f}% ({country_df[0, 'country_code']})")
            country_means = country_df['mean'].to_list()
            print(f"      Variação entre países: {np.std(country_means):.1f}% (std)")

        analysis['country_stats'] = country_df.to_dicts()

        return analysis

    def _write_prediction_artifact(self) -> None:
        """Persist the baseline test prediction vectors of every fold."""
        path = predictions_path('dataframe_lib', 'baseline')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        written = self._prediction_recorder.write(path)
        if written:
            print(f"Prediction vectors written: {written}")

    def test_baseline_models(self) -> Dict:
        """
        Testar modelos baseline científicos com validação temporal.

        Returns:
            Dict: Resultados dos modelos baseline para todos os folds
        """
        print(f"\nBaselines com validação temporal")

        baseline_results = {}

        for fold_id, fold in enumerate(self.folds):
            _fold_t0 = time.perf_counter()
            # Inicializados aqui, e não só na fronteira: no engine SQL a
            # fronteira fica dentro de um try, e depender do fluxo de
            # controle para definir um nome é como se produz NameError.
            # None significa não medido, e não zero, que entraria nas somas.
            _fold_load_s = None
            _fit_t0 = _fold_t0
            print(f"\nFold {fold_id}: Train({fold['train_start']}-{fold['train_end']}) ->"
                  f"Val({fold['val_start']}-{fold['val_end']}) ->"
                  f"Test({fold['test_start']}-{fold['test_end']})")

            # Exclusão de gap years
            train_lazy = self.df_lazy.filter(
                (pl.col('year') >= fold['train_start']) & (pl.col('year') <= fold['train_end'])
            ).filter(
                ~((pl.col('year') >= fold['train_gap_start']) & (pl.col('year') <= fold['train_gap_end']))
            )
            val_lazy = self.df_lazy.filter(
                (pl.col('year') >= fold['val_start']) & (pl.col('year') <= fold['val_end'])
            )
            test_lazy = self.df_lazy.filter(
                (pl.col('year') >= fold['test_start']) & (pl.col('year') <= fold['test_end'])
            ).filter(
                ~((pl.col('year') >= fold['val_gap_start']) & (pl.col('year') <= fold['val_gap_end']))
            )

            # Materializar para operações pandas (ordenação determinística)
            train_df = train_lazy.sort(['country_code', 'year']).collect().to_pandas()
            val_df = val_lazy.sort(['country_code', 'year']).collect().to_pandas()
            test_df = test_lazy.sort(['country_code', 'year']).collect().to_pandas()

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
            # Fronteira da decomposição: acima é materialização do fold, que é
            # do engine; abaixo é o ajuste dos baselines, comum aos três.
            _fold_load_s = time.perf_counter() - _fold_t0
            _fit_t0 = time.perf_counter()

            if len(train_clean) == 0 or len(test_clean) == 0:
                print(f"   Fold {fold_id}: Dados insuficientes")
                continue

            y_train = train_clean[self.target_col].values
            y_val = val_clean[self.target_col].values
            y_test = test_clean[self.target_col].values
            global_mean = y_train.mean()

            # MASE scale from training data
            def _mase_scale_from_train(df):
                try:
                    if df is None or df.empty:
                        return None
                    diffs = []
                    for _, g in df.sort_values(['country_code', 'year']).groupby('country_code'):
                        vals = g[self.target_col].values
                        if len(vals) > 1:
                            diffs.append(np.abs(np.diff(vals)))
                    if not diffs:
                        return None
                    return float(np.mean(np.concatenate(diffs)))
                except Exception:
                    return None

            mase_scale = _mase_scale_from_train(train_clean)

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
                'method': 'global_mean'
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
                'test_wape': float((np.abs(y_test - test_pred_trend)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None,
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_trend))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'slope': float(trend_model.coef_[0]),
                'method': 'linear_trend'
            }

            # Baseline 3: Naive com Lag Científico
            MIN_LAG = int(SCIENTIFIC_CONFIG.get('temporal_gap_years', 2))

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

            fold_results['cross_country'] = {
                'val_r2': float(val_r2_cross),
                'test_r2': float(test_r2_cross),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_cross))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_cross))),
                'test_wape': float((np.abs(y_test - test_pred_cross)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None,
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_cross))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'min_lag_years': MIN_LAG,
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

            # Best baseline por fold
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

            self._prediction_recorder.record(
                fold=fold_id, model='global_mean', y_true=y_test,
                y_pred=test_pred_global, entities=test_clean['country_code'])
            self._prediction_recorder.record(
                fold=fold_id, model='linear_trend', y_true=y_test,
                y_pred=test_pred_trend, entities=test_clean['country_code'])
            self._prediction_recorder.record(
                fold=fold_id, model='naive_with_lag', y_true=y_test,
                y_pred=test_pred_naive, entities=test_clean['country_code'])
            self._prediction_recorder.record(
                fold=fold_id, model='cross_country', y_true=y_test,
                y_pred=test_pred_cross, entities=test_clean['country_code'])

            fold_results['fold_duration_s'] = time.perf_counter() - _fold_t0
            fold_results['fold_load_s'] = _fold_load_s
            fold_results['fit_predict_s'] = time.perf_counter() - _fit_t0
            baseline_results[f'fold_{fold_id}'] = fold_results

        self._write_prediction_artifact()

        return baseline_results

    def analyze_predictability(self, baseline_results: Dict) -> Dict:
        """
        Análise científica de predictabilidade dos modelos baseline.

        Segue o mesmo protocolo de Data Lake e Data Warehouse.

        Args:
            baseline_results: Resultados dos modelos baseline por fold

        Returns:
            Dict: Análise completa de predictabilidade com métricas de estabilidade
        """
        print("\nAnálise de predictabilidade Polars")

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

            predictability_analysis = {
                'architecture': 'dataframe_lib',
                'methodology': 'scientific_baselines_with_temporal_lags',
                'validation_scores': all_val_scores,
                'test_scores': all_test_scores,
                'generalization_gaps': generalization_gaps,
                'best_baseline': best_baseline_overall,
                'best_test_r2': best_mean_test_r2,
                'best_val_r2': best_mean_val_r2,
                'generalization_gap': best_generalization_gap,
                'predictability_level': 'unknown'
            }

            if best_mean_test_r2 < 0:
                predictability_analysis['predictability_level'] = 'very_low'
            elif best_mean_test_r2 < 0.05:
                predictability_analysis['predictability_level'] = 'very_low'
            elif best_mean_test_r2 < 0.15:
                predictability_analysis['predictability_level'] = 'low'
            elif best_mean_test_r2 < 0.35:
                predictability_analysis['predictability_level'] = 'moderate'
            else:
                predictability_analysis['predictability_level'] = 'good'

            avg_generalization_gap = np.mean([gap_data['mean_gap'] for gap_data in generalization_gaps.values()])
            abs_avg_gap = abs(avg_generalization_gap)

            if abs_avg_gap <= 0.05:
                stability_level = "excellent"
            elif abs_avg_gap <= 0.1:
                stability_level = "good"
            elif abs_avg_gap <= 0.15:
                stability_level = "moderate"
            else:
                stability_level = "low"

            predictability_analysis['stability_analysis'] = {
                'avg_generalization_gap': float(avg_generalization_gap),
                'stability_level': stability_level
            }

            print(f"\n   Melhor baseline: {best_baseline_overall}")
            print(f"      Performance Teste: R² = {best_mean_test_r2:.3f}")
            print(f"      Estabilidade: {stability_level} (gap médio: {avg_generalization_gap:+.3f})")

        else:
            predictability_analysis = {
                'architecture': 'dataframe_lib',
                'baseline_scores': {},
                'predictability_level': 'unknown'
            }

        return predictability_analysis

    def save_results(self, target_analysis: Dict, baseline_results: Dict,
                     predictability_analysis: Dict) -> Dict:
        """
        Salvar resultados no formato padronizado (idêntico a DL/DW).

        Args:
            target_analysis: Análise da distribuição do target
            baseline_results: Resultados dos modelos baseline
            predictability_analysis: Análise de predictabilidade

        Returns:
            Dict: Resultados completos consolidados
        """
        print(f"\nSalvando resultados...")

        full_results = {
            'architecture': 'dataframe_lib',
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

        results_file = f"{self.results_path}/baseline_analysis_dataframe_lib_results.json"
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)

        print(f"   Resultados Polars salvos: {results_file}")

        return full_results

    def run_complete_analysis(self):
        """
        Executar análise completa de baseline Polars DataFrame.

        Segue o mesmo protocolo de run_complete_analysis de Data Lake.

        Returns:
            Dict: Resultados consolidados da análise ou erro
        """
        print(f"Análise completa - arquitetura Polars")

        try:
            target_analysis = self.analyze_target_distribution()
            baseline_results = self.test_baseline_models()
            predictability_analysis = self.analyze_predictability(baseline_results)
            results = self.save_results(target_analysis, baseline_results,
                                        predictability_analysis)

            print(f"\nResumo - arquitetura Polars:")
            print(f"   Target: {self.target_col}")
            print(f"   Predictabilidade: {predictability_analysis.get('predictability_level', 'unknown').upper()}")
            print(f"   Melhor baseline: {predictability_analysis.get('best_baseline', 'unknown')}")
            print(f"   R² Teste: {predictability_analysis.get('best_test_r2', 0):.3f}")

            return results

        except Exception as e:
            import traceback
            print(f"\nErro na análise Polars: {e}")
            traceback.print_exc()
            return {
                'architecture': 'dataframe_lib',
                'status': 'failed',
                'error': str(e)
            }


if __name__ == "__main__":
    analyzer = BaselineModelAnalysisDataFrameLib()
    results = analyzer.run_complete_analysis()
    print("\nAnálise baseline Polars concluída!")
