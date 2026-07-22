#!/usr/bin/env python3
"""
Análise de modelos baseline para arquitetura Data Warehouse.

Módulo para análise comparativa usando padrão ML Data Warehouse Consumer
com queries diretas a views e validação temporal walk-forward com gaps.

Resumo técnico:
- Padrão ML Data Warehouse Consumer (Connection Manager + views temporais)
- Validação temporal com gaps (mínimo 2 anos) para prevenir vazamento
- Modelos baseline: média global, tendência linear, naive com lag, cross-country
- Acesso via SQL nativo (DuckDB) sem I/O de arquivos durante treinamento
"""
import time

import pandas as pd
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

core_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core')
core_path = os.path.abspath(core_path)
if core_path not in sys.path:
    sys.path.append(core_path)

project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import get_absolute_output_path
from core.models.hierarchical import (
    write_baseline_predictions as shared_write_baseline_predictions)
from core.prediction_store import PredictionRecorder, predictions_path
from core.scientific_config import SCIENTIFIC_CONFIG, setup_reproducibility

setup_reproducibility()

sql_engine_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'collection', 'sql_engine')
sql_engine_path = os.path.abspath(sql_engine_path)
if sql_engine_path not in sys.path:
    sys.path.append(sql_engine_path)

from connection_manager import DuckDBConnectionManager, SQLProcessingError



def _best_by_val_r2(fold_results: dict):
    """Melhor baseline por R2 de validação, ignorando os indefinidos.

    `max` compara com NaN devolvendo False, então bastava o primeiro item ter
    R2 indefinido para ele ser eleito o melhor -- e `best_test_r2` e
    `generalization_gap` derivavam desse. A escolha passava a depender da ordem
    de inserção no dicionário, não do desempenho.
    """
    import math

    scored = [(name, data['val_r2']) for name, data in fold_results.items()
              if isinstance(data, dict) and 'val_r2' in data
              and data['val_r2'] is not None
              and not math.isnan(float(data['val_r2']))]
    if not scored:
        raise ValueError(
            "Nenhum baseline tem R2 de validação definido neste fold; não há "
            "melhor baseline a reportar."
        )
    return max(scored, key=lambda pair: pair[1])

class BaselineModelAnalysisSqlEngine:
    """
    ML Data Warehouse Consumer - Análise Baseline.
    
    Implementa padrão ML Data Warehouse Consumer para análise científica
    de modelos baseline com validação temporal utilizando:
    - Queries diretas via Connection Manager
    - Consumo de views do Feature Store para treinamento ML
    - Connection pooling para performance em workloads ML
    - Eliminação de file I/O durante treinamento
    
    Attributes:
        folds_path (str): Caminho para configuração dos folds temporais
        results_path (str): Diretório para salvar resultados
        db_path (str): Caminho para o banco DuckDB
        conn_manager (DuckDBConnectionManager): Gerenciador de conexões
        target_col (str): Nome da coluna target para predição
        folds (list): Lista de configurações dos folds temporais
    """
    
    def __init__(self):
        self._prediction_recorder = PredictionRecorder('sql_engine')
        """Inicializa a análise baseline para arquitetura Data Warehouse."""
        print("Inicializando análise baseline DuckDB")
        
        dataset_name = os.environ.get('DATASET_NAME', 'worldbank')
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/prep/temporal_folds_sql_engine.json")
        self.results_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/models")
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
        
        self.target_col = 'dropout_rate_sql_engine'
        self._ensure_target_column()
        
        self._verify_feature_store_views()
        self._load_data_summary_from_views()

    def _ensure_target_column(self) -> None:
        """Garante que a coluna de target exista em analytics_wide.

        Caso ausente, cria e popula como 100 - lower_secondary_completion_rate,
        preservando NULLs/valores inválidos.
        """
        try:
            exists = self.conn_manager.execute_scalar(
                f"SELECT COUNT(*) > 0 FROM information_schema.columns WHERE table_name = 'analytics_wide' AND column_name = '{self.target_col}'"
            )
            if not exists:
                print(f"   Target '{self.target_col}' ausente - criando coluna...")
                self.conn_manager.execute_sql_no_return(
                    f"ALTER TABLE analytics_wide ADD COLUMN IF NOT EXISTS {self.target_col} DOUBLE"
                )
                self.conn_manager.execute_sql_no_return(
                    f"""
                    UPDATE analytics_wide
                    SET {self.target_col} = CASE
                        WHEN lower_secondary_completion_rate IS NULL THEN NULL
                        WHEN lower_secondary_completion_rate < 0 OR lower_secondary_completion_rate > 100 THEN NULL
                        ELSE 100 - lower_secondary_completion_rate
                    END
                    """
                )
                print("   Target criado/populado em analytics_wide")
        except SQLProcessingError as e:
            raise RuntimeError(f"Falha ao garantir coluna target: {e}")
    
    def _verify_feature_store_views(self):
        """Verificar se views temporais existem no Data Warehouse."""
        print("   Verificando views temporais...")
        
        try:
            table_exists = self.conn_manager.execute_scalar(f"""
                SELECT COUNT(*) > 0 
                FROM information_schema.tables 
                WHERE table_name = 'analytics_wide'
            """)
            
            if not table_exists:
                raise RuntimeError(f"Tabela base não encontrada: analytics_wide")
            
            views_found = 0
            for fold_id in range(len(self.folds)):
                for split in ['train', 'val', 'test']:
                    view_name = f"vw_fold_{fold_id}_{split}"
                    view_exists = self.conn_manager.execute_scalar(f"""
                        SELECT COUNT(*) > 0 
                        FROM information_schema.views 
                        WHERE table_name = '{view_name}'
                    """)
                    
                    if view_exists:
                        views_found += 1
            
            if views_found == 0:
                print(f"   Aviso: Nenhuma view temporal encontrada")
                print(f"   Execute o setup.py primeiro para criar as views temporais")
                print(f"   Usando tabela base como fallback")
            else:
                print(f"   Views temporais verificadas: {views_found} views encontradas")
            
        except SQLProcessingError as e:
            raise RuntimeError(f"Erro ao verificar views temporais: {e}")
    
    def _load_data_summary_from_views(self):
        """Carregar resumo dos dados via queries diretas às views."""
        print("   Carregando resumo dos dados via views...")
        
        try:
            # Preferir feature store view (garante presença do target)
            view_base = 'vw_selected_features'
            total_records = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM {view_base} WHERE {self.target_col} IS NOT NULL")
            min_year = self.conn_manager.execute_scalar(f"SELECT MIN(year) FROM {view_base} WHERE {self.target_col} IS NOT NULL")
            max_year = self.conn_manager.execute_scalar(f"SELECT MAX(year) FROM {view_base} WHERE {self.target_col} IS NOT NULL")
            total_countries = self.conn_manager.execute_scalar(f"SELECT COUNT(DISTINCT country_code) FROM {view_base} WHERE {self.target_col} IS NOT NULL")
            target_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM {view_base} WHERE {self.target_col} IS NOT NULL") or 0
            target_std = self.conn_manager.execute_scalar(f"SELECT STDDEV({self.target_col}) FROM {view_base} WHERE {self.target_col} IS NOT NULL") or 0
            target_min = self.conn_manager.execute_scalar(f"SELECT MIN({self.target_col}) FROM {view_base} WHERE {self.target_col} IS NOT NULL") or 0
            target_max = self.conn_manager.execute_scalar(f"SELECT MAX({self.target_col}) FROM {view_base} WHERE {self.target_col} IS NOT NULL") or 0
            negative_target = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM {view_base} WHERE {self.target_col} < 0") or 0
            target_exists = self.conn_manager.execute_scalar(f"SELECT COUNT(*) > 0 FROM information_schema.columns WHERE table_name = '{view_base}' AND column_name = '{self.target_col}'")
            
            print(f"   Dados via views: {total_records} observações")
            print(f"   Período: {min_year}-{max_year}")
            print(f"   Países: {total_countries}")
            print(f"   Target: {self.target_col}")
            print(f"   Folds: {len(self.folds)}")
            
            if not target_exists:
                raise ValueError(f"Target {self.target_col} não encontrado na tabela analytics_wide")
            
            print(f"   Target stats: mean={target_mean:.2f}%, std={target_std:.2f}%")
            
            if negative_target > 0:
                print(f"   Target inválido: {negative_target} valores negativos detectados")
            else:
                print(f"   Target válido: range [{target_min:.2f}%, {target_max:.2f}%]")
            
            self._cached_target_stats = {
                'mean': target_mean,
                'std': target_std,
                'min': target_min,
                'max': target_max,
                'total_records': total_records,
                'min_year': min_year,
                'max_year': max_year,
                'total_countries': total_countries
            }
                
        except SQLProcessingError as e:
            raise RuntimeError(f"Erro ao carregar resumo via views: {e}")
    
    def _load_ml_fold_data(self, fold_id: int, split: str) -> pd.DataFrame:
        """
        Carregar dados do fold via padrão ML Data Warehouse Consumer com Query Dinâmica SQL-First.
        
        Implementa padrão ML Data Warehouse Consumer:
        - Queries diretas às views temporais pós-filtragem de colinearidade
        - Descoberta dinâmica de features via information_schema (100% SQL)
        - Connection pooling para performance em workloads ML
        - Mantém paradigma SQL-first do Data Warehouse
        
        Args:
            fold_id: ID do fold temporal
            split: 'train', 'val', ou 'test'
            
        Returns:
            DataFrame com features ML disponíveis pós-filtragem de colinearidade
        """
        cache_key = f"fold_{fold_id}_{split}"
        if not hasattr(self, '_fold_data_cache'):
            self._fold_data_cache = {}
        
        if cache_key in self._fold_data_cache:
            return self._fold_data_cache[cache_key]
        
        fold = self.folds[fold_id]
        
        if split == 'train':
            year_start, year_end = fold['train_start'], fold['train_end']
        elif split == 'val':
            year_start, year_end = fold['val_start'], fold['val_end']
        elif split == 'test':
            year_start, year_end = fold['test_start'], fold['test_end']
        else:
            raise ValueError(f"Invalid split: {split}")
        
        try:
            view_name = f"vw_fold_{fold_id}_{split}"
            
            view_exists = self.conn_manager.execute_scalar(f"""
                SELECT COUNT(*) > 0
                FROM information_schema.views
                WHERE table_name = '{view_name}'
            """)
            
            if view_exists:
                print(f"      Usando view temporal: {view_name}")
                
                # Descoberta dinâmica de features via SQL
                try:
                    available_features_query = f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = '{view_name}'
                        AND column_name NOT IN ('country_code', 'year', '{self.target_col}')
                        ORDER BY column_name
                    """
                    
                    feature_result = self.conn_manager.execute_sql(available_features_query)
                    available_features = feature_result['column_name'].tolist()
                    
                    if available_features:
                        feature_list = ', '.join(available_features)
                        print(f"      Features descobertas via SQL: {len(available_features)} colunas")
                        
                        # Query dinâmica com features realmente disponíveis (100% SQL)
                        query = f"""
                            SELECT
                                country_code,
                                year,
                                {self.target_col},
                                {feature_list}
                            FROM {view_name}
                            WHERE {self.target_col} IS NOT NULL
                            ORDER BY country_code, year
                        """
                    else:
                        print(f"      Nenhuma feature descoberta, usando colunas básicas")
                        query = f"""
                            SELECT
                                country_code,
                                year,
                                {self.target_col}
                            FROM {view_name}
                            WHERE {self.target_col} IS NOT NULL
                            ORDER BY country_code, year
                        """
                        
                except SQLProcessingError as e:
                    print(f"      Erro na descoberta de features via SQL: {e}")
                    print(f"      Fallback: usando query com colunas básicas")
                    query = f"""
                        SELECT
                            country_code,
                            year,
                            {self.target_col}
                        FROM {view_name}
                        WHERE {self.target_col} IS NOT NULL
                        ORDER BY country_code, year
                    """
                    
            else:
                print(f"      Fallback: View {view_name} não encontrada")
                print(f"      Usando tabela base: analytics_wide (anos {year_start}-{year_end})")
                print(f"      Execute setup.py primeiro para criar views temporais")
                
                # Descoberta de features da tabela base (SQL-first)
                try:
                    base_features_query = f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'analytics_wide'
                        AND column_name NOT IN ('country_code', 'year', '{self.target_col}', 'data_source',
                                               'data_completeness_score', 'etl_batch_id', 'collection_timestamp',
                                               'country_name', 'country_stratum')
                        AND data_type IN ('DOUBLE', 'INTEGER', 'FLOAT', 'DECIMAL', 'NUMERIC')
                        ORDER BY column_name
                    """
                    
                    base_feature_result = self.conn_manager.execute_sql(base_features_query)
                    base_available_features = base_feature_result['column_name'].tolist()
                    
                    if base_available_features:
                        base_feature_list = ', '.join(base_available_features)
                        print(f"      Features descobertas da tabela base: {len(base_available_features)} colunas")
                        
                        query = f"""
                            SELECT
                                country_code,
                                year,
                                {self.target_col},
                                {base_feature_list}
                            FROM analytics_wide
                            WHERE {self.target_col} IS NOT NULL
                              AND year >= {year_start}
                              AND year <= {year_end}
                            ORDER BY country_code, year
                        """
                    else:
                        query = f"""
                            SELECT
                                country_code,
                                year,
                                {self.target_col}
                            FROM analytics_wide
                            WHERE {self.target_col} IS NOT NULL
                              AND year >= {year_start}
                              AND year <= {year_end}
                            ORDER BY country_code, year
                        """
                        
                except SQLProcessingError as e:
                    print(f"      Erro na descoberta de features da tabela base: {e}")
                    query = f"""
                        SELECT
                            country_code,
                            year,
                            {self.target_col}
                        FROM analytics_wide
                        WHERE {self.target_col} IS NOT NULL
                          AND year >= {year_start}
                          AND year <= {year_end}
                        ORDER BY country_code, year
                    """
            
            df = self.conn_manager.execute_sql(query)
            
            if df.empty:
                data_source = view_name if view_exists else f"analytics_wide (years {year_start}-{year_end})"
                raise SQLProcessingError(f"No data returned from {data_source}")
            
            self._fold_data_cache[cache_key] = df
            
            feature_count = len([col for col in df.columns if col not in ['country_code', 'year', self.target_col]])
            
            if view_exists:
                print(f"      Dados da view temporal: {len(df)} registros, {feature_count} features")
            else:
                print(f"      Dados da tabela base: {len(df)} registros, {feature_count} features")
            
            return df
            
        except SQLProcessingError as e:
            raise RuntimeError(f"Erro ao carregar dados do fold {fold_id} split {split}: {e}")
    
    def cleanup(self):
        """Limpar recursos do gerenciador de conexões."""
        if hasattr(self, 'conn_manager') and self.conn_manager:
            try:
                self.conn_manager.close_connection()
                print("   Connection Manager fechado")
            except Exception as e:
                print(f"   Erro ao fechar Connection Manager: {e}")
    
    def analyze_target_distribution(self) -> Dict:
        """
        Analisar distribuição do target via Data Warehouse views.
        
        Returns:
            Dict: Estatísticas da distribuição do target incluindo temporal e por país
        """
        print(f"\nAnálise da distribuição do target DuckDB")
        
        analysis = {}
        
        try:
            if hasattr(self, '_cached_target_stats'):
                cached_stats = self._cached_target_stats
                
                missing_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM analytics_wide WHERE {self.target_col} IS NULL")
                total_count = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
                
                target_stats = {
                    'architecture': 'sql_engine',
                    'target_variable': self.target_col,
                    'mean': cached_stats['mean'],
                    'std': cached_stats['std'],
                    'min': cached_stats['min'],
                    'max': cached_stats['max'],
                    'missing_count': int(missing_count),
                    'missing_rate': float(missing_count / total_count if total_count > 0 else 0)
                }
            else:
                target_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL") or 0
                target_std = self.conn_manager.execute_scalar(f"SELECT STDDEV({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL") or 0
                target_min = self.conn_manager.execute_scalar(f"SELECT MIN({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL") or 0
                target_max = self.conn_manager.execute_scalar(f"SELECT MAX({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL") or 0
                missing_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM analytics_wide WHERE {self.target_col} IS NULL")
                total_count = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
                
                target_stats = {
                    'architecture': 'sql_engine',
                    'target_variable': self.target_col,
                    'mean': float(target_mean),
                    'std': float(target_std),
                    'min': float(target_min),
                    'max': float(target_max),
                    'missing_count': int(missing_count),
                    'missing_rate': float(missing_count / total_count if total_count > 0 else 0)
                }
            
            print(f"   Target ({self.target_col}):")
            print(f"      Média: {target_stats['mean']:.2f}%")
            print(f"      Desvio: {target_stats['std']:.2f}%")
            print(f"      Range: {target_stats['min']:.2f}% - {target_stats['max']:.2f}%")
            print(f"      Missing: {target_stats['missing_count']} ({target_stats['missing_rate']:.1%})")
            
            analysis['target_stats'] = target_stats
            
            temporal_query = f"""
                SELECT 
                    year,
                    COUNT(*) as count,
                    AVG({self.target_col}) as mean,
                    STDDEV({self.target_col}) as std,
                    MIN({self.target_col}) as min,
                    MAX({self.target_col}) as max
                FROM analytics_wide 
                WHERE {self.target_col} IS NOT NULL
                GROUP BY year
                ORDER BY year
            """
            temporal_df = self.conn_manager.execute_sql(temporal_query)
            
            if not temporal_df.empty:
                if hasattr(self, '_cached_target_stats'):
                    first_year = self._cached_target_stats['min_year']
                    last_year = self._cached_target_stats['max_year']
                else:
                    first_year = self.conn_manager.execute_scalar(f"SELECT MIN(year) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
                    last_year = self.conn_manager.execute_scalar(f"SELECT MAX(year) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
                
                first_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM analytics_wide WHERE year = {first_year} AND {self.target_col} IS NOT NULL")
                last_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM analytics_wide WHERE year = {last_year} AND {self.target_col} IS NOT NULL")
                
                print(f"\n   Evolução temporal DuckDB:")
                print(f"      Primeiro ano ({first_year}): {first_mean:.1f}%")
                print(f"      Último ano ({last_year}): {last_mean:.1f}%")
                
                trend = last_mean - first_mean
                print(f"      Tendência: {trend:.1f}% em {last_year - first_year} anos")
                
                analysis['temporal_stats'] = temporal_df.to_dict('records')
            
            country_query = f"""
                SELECT 
                    country_code,
                    COUNT(*) as count,
                    AVG({self.target_col}) as mean,
                    STDDEV({self.target_col}) as std,
                    MIN({self.target_col}) as min,
                    MAX({self.target_col}) as max
                FROM analytics_wide 
                WHERE {self.target_col} IS NOT NULL
                GROUP BY country_code
                ORDER BY mean
            """
            country_df = self.conn_manager.execute_sql(country_query)
            
            if not country_df.empty:
                min_dropout_mean = self.conn_manager.execute_scalar(f"SELECT MIN(avg_dropout) FROM (SELECT country_code, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY country_code)")
                max_dropout_mean = self.conn_manager.execute_scalar(f"SELECT MAX(avg_dropout) FROM (SELECT country_code, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY country_code)")
                min_dropout_country_code = self.conn_manager.execute_scalar(f"SELECT country_code FROM (SELECT country_code, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY country_code ORDER BY avg_dropout LIMIT 1)")
                max_dropout_country_code = self.conn_manager.execute_scalar(f"SELECT country_code FROM (SELECT country_code, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY country_code ORDER BY avg_dropout DESC LIMIT 1)")
                country_variation = self.conn_manager.execute_scalar(f"SELECT STDDEV(avg_dropout) FROM (SELECT country_code, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY country_code)")
                
                print(f"\n   Variação por país:")
                print(f"      Menor dropout: {min_dropout_mean:.1f}% ({min_dropout_country_code})")
                print(f"      Maior dropout: {max_dropout_mean:.1f}% ({max_dropout_country_code})")
                print(f"      Variação entre países: {country_variation:.1f}% (std)")
                
                analysis['country_stats'] = country_df.to_dict('records')
            
            return analysis
            
        except SQLProcessingError as e:
            print(f"   [ERROR] Análise de distribuição: {e}")
            return {
                'architecture': 'sql_engine',
                'error': str(e),
                'target_stats': {}
            }
    
    def _write_prediction_artifact(self) -> None:
        """Delega à implementação compartilhada."""
        shared_write_baseline_predictions(self._prediction_recorder,
                                         architecture='sql_engine')

    def test_baseline_models(self) -> Dict:
        """
        Testar modelos baseline científicos via Data Warehouse.
        
        Correções metodológicas implementadas:
        - Lag mínimo de 2 anos para evitar vazamento temporal
        - Validação walk-forward científica correta
        - Gaps temporais apropriados entre conjuntos
        
        Returns:
            Dict: Resultados dos baselines com validação temporal
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
            print(f"\nFold {fold_id}: Train({fold['train_start']}-{fold['train_end']}) -> Val({fold['val_start']}-{fold['val_end']}) -> Test({fold['test_start']}-{fold['test_end']})")

            try:
                train_data = self._load_ml_fold_data(fold_id, 'train')
                val_data = self._load_ml_fold_data(fold_id, 'val')
                test_data = self._load_ml_fold_data(fold_id, 'test')

                print(f"   Dados: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
                print(f"   Gaps: Train-Val={fold['val_start']-fold['train_end']-1}yr, Val-Test={fold['test_start']-fold['val_end']-1}yr")
                print(f"   Features disponíveis: {len(train_data.columns)} colunas")
                
                train_clean = train_data
                val_clean = val_data
                test_clean = test_data
                # Fronteira da decomposição: acima é materialização do fold, que é
                # do engine; abaixo é o ajuste dos baselines, comum aos três.
                _fold_load_s = time.perf_counter() - _fold_t0
                _fit_t0 = time.perf_counter()
                
            except Exception as e:
                print(f"   Erro ao carregar dados do fold {fold_id}: {e}")
                continue
            
            if len(train_clean) == 0 or len(test_clean) == 0:
                print(f"   Fold {fold_id}: Dados insuficientes")
                continue
            
            y_train = train_clean[self.target_col]
            y_val = val_clean[self.target_col] 
            y_test = test_clean[self.target_col]

            # Escala MASE a partir do treino (diferenças absolutas por país)
            def _mase_scale_from_train(df):
                try:
                    if df is None or len(df) == 0:
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
            mase_scale = _mase_scale_from_train(train_clean)
            
            fold_results = {}
            
            global_mean = y_train.mean()
            
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
            
            X_train_time = train_clean[['year']].values
            trend_model = LinearRegression()  # LinearRegression não aceita random_state
            trend_model.fit(X_train_time, y_train)
            
            X_val_time = val_clean[['year']].values
            X_test_time = test_clean[['year']].values
            
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
            
            MIN_LAG = int(SCIENTIFIC_CONFIG.get('temporal_gap_years', 2))
            
            print(f"      Naive baseline...")
            
            val_pred_naive = []
            for _, val_row in val_clean.iterrows():
                country = val_row['country_code']
                val_year = val_row['year']
                
                country_history = train_clean[
                    (train_clean['country_code'] == country) & 
                    (train_clean['year'] <= val_year - MIN_LAG)
                ].sort_values('year')
                
                if len(country_history) > 0:
                    naive_val = country_history[self.target_col].iloc[-1]
                else:
                    naive_val = global_mean
                
                val_pred_naive.append(naive_val)
            
            test_pred_naive = []
            combined_history = pd.concat([train_clean, val_clean], ignore_index=True)
            
            for _, test_row in test_clean.iterrows():
                country = test_row['country_code']
                test_year = test_row['year']
                
                country_history = combined_history[
                    (combined_history['country_code'] == country) & 
                    (combined_history['year'] <= test_year - MIN_LAG)
                ].sort_values('year')
                
                if len(country_history) > 0:
                    naive_test = country_history[self.target_col].iloc[-1]
                else:
                    naive_test = combined_history[self.target_col].mean()
                
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
            
            print(f"      Cross-Country baseline...")
            
            val_pred_cross = []
            for _, val_row in val_clean.iterrows():
                country = val_row['country_code']
                val_year = val_row['year']

                year_data = train_clean[
                    train_clean['year'] <= val_year - MIN_LAG
                ]

                if len(year_data) > 0:
                    country_means = year_data.groupby('country_code')[self.target_col].mean()
                    other_countries = country_means[country_means.index != country]
                    if len(other_countries) > 0:
                        cross_val = other_countries.mean()
                    else:
                        cross_val = global_mean
                else:
                    cross_val = global_mean

                val_pred_cross.append(cross_val)
            
            test_pred_cross = []
            for _, test_row in test_clean.iterrows():
                country = test_row['country_code']
                test_year = test_row['year']

                year_data = combined_history[
                    combined_history['year'] <= test_year - MIN_LAG
                ]

                if len(year_data) > 0:
                    country_means = year_data.groupby('country_code')[self.target_col].mean()
                    other_countries = country_means[country_means.index != country]
                    if len(other_countries) > 0:
                        cross_test = other_countries.mean()
                    else:
                        cross_test = combined_history[self.target_col].mean()
                else:
                    cross_test = combined_history[self.target_col].mean()

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
            
            # Agregar WAPE/MASE para Naive
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
            
            # Melhor baseline em validação
            best_val_baseline, best_val_r2 = _best_by_val_r2(fold_results)
            
            # Performance do melhor baseline no teste
            best_test_r2 = fold_results[best_val_baseline]['test_r2']
            generalization_gap = best_val_r2 - best_test_r2
            
            fold_results['best_baseline'] = {
                'model': best_val_baseline,
                'val_r2': best_val_r2,
                'test_r2': best_test_r2,
                'generalization_gap': generalization_gap
            }
            
            print(f"   Melhor baseline: {best_val_baseline} (Val: {best_val_r2:.3f} ->Test: {best_test_r2:.3f}, Gap: {generalization_gap:+.3f})")
            
            # Análise mais nuançada do gap de generalização
            abs_gap = abs(generalization_gap)
            if abs_gap <= 0.05:
                print(f"      Excelente estabilidade: Gap muito baixo (<=0.05)")
            elif abs_gap <= 0.1:
                print(f"      Boa estabilidade: Gap dentro do esperado (<=0.10)")
            elif abs_gap <= 0.15:
                print(f"      Gap moderado: Variação temporal aceitável ({abs_gap:.3f})")
            else:
                print(f"      Gap elevado: Possível instabilidade temporal ({abs_gap:.3f})")
            
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
        Analisar predictabilidade científica dos baselines Data Warehouse.
        
        Args:
            baseline_results: Resultados dos baselines com validação temporal
            
        Returns:
            Dict: Análise de predictabilidade agregada com gaps de generalização
        """
        print("\nAnálise de predictabilidade DuckDB")
        
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
                    # Scores de teste (métrica principal)
                    test_r2_scores.append(fold_data[baseline]['test_r2'])
                    # Scores de validação (para comparação)
                    val_r2_scores.append(fold_data[baseline]['val_r2'])
                    # Gap de generalização
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
        
        # Encontrar melhor baseline baseado no TESTE (não validação)
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
                'architecture': 'sql_engine',
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
            
            predictability_analysis['stability_analysis'] = {
                'avg_generalization_gap': float(avg_generalization_gap),
                'stability_level': stability_level
            }
            
        else:
            predictability_analysis = {
                'architecture': 'sql_engine',
                'baseline_scores': {},
                'predictability_level': 'unknown'
            }
        
        return predictability_analysis
    
    def save_results(self, target_analysis: Dict, baseline_results: Dict, 
                    predictability_analysis: Dict):
        """
        Salvar resultados da análise ML Data Warehouse Consumer.
        
        Args:
            target_analysis: Resultados da análise de distribuição do target
            baseline_results: Resultados dos modelos baseline
            predictability_analysis: Análise de predictabilidade
            
        Returns:
            Dict: Resultados completos salvos
        """
        print(f"\nSalvando resultados DuckDB...")
        
        full_results = {
            'architecture': 'sql_engine_consumer',
            'pattern': 'ml_sql_engine_consumer',
            'target_variable': self.target_col,
            'data_source': self.db_path,
            'data_access_method': 'direct_view_queries',
            'target_distribution_analysis': target_analysis,
            'baseline_model_results': baseline_results,
            'predictability_analysis': predictability_analysis,
            'summary': {
                'total_folds_analyzed': len(baseline_results),
                'best_baseline_model': predictability_analysis.get('best_baseline', 'unknown'),
                'best_baseline_r2': predictability_analysis.get('best_test_r2', 0),
                'predictability_level': predictability_analysis.get('predictability_level', 'unknown'),
                'r2_score_identical_tolerance': 0.001
            }
        }
        
        results_file = f"{self.results_path}/baseline_analysis_sql_engine_consumer_results.json"
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"   Resultados salvos: {results_file}")
        
        return full_results
    
    def run_complete_analysis(self):
        """
        Executar análise completa de baseline via ML Data Warehouse Consumer.
        
        Returns:
            Dict: Resultados completos da análise comparativa Data Warehouse vs Data Lake
        """
        print(f"Análise completa - arquitetura DuckDB")
        
        try:
            target_analysis = self.analyze_target_distribution()
            
            # 2. Testar modelos baseline via Feature Store views
            baseline_results = self.test_baseline_models()
            
            # 3. Analisar predictabilidade
            predictability_analysis = self.analyze_predictability(baseline_results)
            
            # 4. Salvar resultados
            results = self.save_results(target_analysis, baseline_results, 
                                       predictability_analysis)
            
            # 5. Resumo final
            print(f"\nResumo - arquitetura DuckDB:")
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
            
            # Verificar se views temporais foram usadas
            views_used = 0
            fallbacks_used = 0
            if hasattr(self, '_fold_data_cache'):
                for cache_key in self._fold_data_cache.keys():
                    fold_id = int(cache_key.split('_')[1])
                    split = cache_key.split('_')[2]
                    view_name = f"vw_fold_{fold_id}_{split}"
                    view_exists = self.conn_manager.execute_scalar(f"""
                        SELECT COUNT(*) > 0 
                        FROM information_schema.views 
                        WHERE table_name = '{view_name}'
                    """)
                    if view_exists:
                        views_used += 1
                    else:
                        fallbacks_used += 1
            
            if views_used > 0 and fallbacks_used == 0:
                print(f"   Temporal Views: {views_used} views usadas")
            elif views_used > 0 and fallbacks_used > 0:
                print(f"   [WARN] Temporal Views: parcial ({views_used} views, {fallbacks_used} fallbacks)")
            else:
                print(f"   [WARN] Temporal Views: nenhuma (apenas fallback)")
                print(f"   Execute setup.py primeiro para criar views temporais")

            return results
            
        except Exception as e:
            # Re-levanta pelo mesmo motivo do modelo hierárquico: um dicionário
            # com status 'failed' atravessa como execução bem-sucedida.
            print(f"\n[ERROR] Análise DuckDB: {e}")
            raise
        finally:
            self.cleanup()

if __name__ == "__main__":
    print("=" * 60)
    analyzer = None
    try:
        analyzer = BaselineModelAnalysisSqlEngine()
        results = analyzer.run_complete_analysis()
        print(f"\nAnálise baseline DuckDB concluída!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        if analyzer:
            analyzer.cleanup()
        raise
    print("=" * 60)
