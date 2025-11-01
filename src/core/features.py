#!/usr/bin/env python3
"""
Módulo centralizado de Feature Engineering.

Consolida lógica de criação e transformação de features para eliminar
duplicação entre arquiteturas, mantendo flexibilidade para customizações.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Any
from datetime import datetime

# Suporte para Dask (opcional)
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


class FeatureEngineer:
    """
    Classe centralizada para feature engineering.
    
    Implementa transformações e criação de features de forma unificada,
    suportando diferentes backends (Pandas, Dask) e preservando a
    lógica científica original das arquiteturas.
    """
    
    @staticmethod
    def create_lag_features(df: Union[pd.DataFrame, 'dd.DataFrame'],
                           columns: List[str],
                           lags: List[int],
                           group_by: Optional[List[str]] = None) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Cria features de lag temporal.
        
        Args:
            df: DataFrame de entrada (Pandas ou Dask)
            columns: Colunas para criar lags
            lags: Lista de períodos de lag
            group_by: Colunas para agrupamento (ex: country_code)
            
        Returns:
            DataFrame com features de lag adicionadas
            
        Example:
            >>> df = create_lag_features(df, ['gdp_per_capita'], [1, 2, 3], ['country_code'])
            Adiciona: gdp_per_capita_lag1, gdp_per_capita_lag2, gdp_per_capita_lag3
        """
        # Detectar tipo de DataFrame
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                lag_col = f"{col}_lag{lag}"
                
                if group_by:
                    if is_dask:
                        # Dask: usar shift com partições
                        df[lag_col] = df.groupby(group_by)[col].shift(lag)
                    else:
                        # Pandas
                        df[lag_col] = df.groupby(group_by)[col].shift(lag)
                else:
                    df[lag_col] = df[col].shift(lag)
        
        return df
    
    @staticmethod
    def create_rolling_features(df: Union[pd.DataFrame, 'dd.DataFrame'],
                               columns: List[str],
                               windows: List[int],
                               functions: List[str] = ['mean', 'std'],
                               group_by: Optional[List[str]] = None,
                               min_periods: int = 1) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Cria features de janela móvel (rolling statistics).
        
        Args:
            df: DataFrame de entrada
            columns: Colunas para estatísticas móveis
            windows: Tamanhos de janela
            functions: Funções estatísticas ('mean', 'std', 'min', 'max')
            group_by: Colunas para agrupamento
            min_periods: Períodos mínimos para cálculo
            
        Returns:
            DataFrame com features de rolling statistics
            
        Example:
            >>> df = create_rolling_features(df, ['inflation_rate'], [3, 5], ['mean', 'std'])
            Adiciona: inflation_rate_rolling3_mean, inflation_rate_rolling3_std, etc.
        """
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for window in windows:
                for func in functions:
                    feature_name = f"{col}_rolling{window}_{func}"
                    
                    if group_by:
                        if is_dask:
                            # Dask: rolling com groupby requer cuidado
                            grouped = df.groupby(group_by)[col]
                            if func == 'mean':
                                df[feature_name] = grouped.rolling(window, min_periods=min_periods).mean()
                            elif func == 'std':
                                df[feature_name] = grouped.rolling(window, min_periods=min_periods).std()
                            elif func == 'min':
                                df[feature_name] = grouped.rolling(window, min_periods=min_periods).min()
                            elif func == 'max':
                                df[feature_name] = grouped.rolling(window, min_periods=min_periods).max()
                        else:
                            # Pandas
                            grouped = df.groupby(group_by)[col]
                            df[feature_name] = grouped.rolling(window, min_periods=min_periods).agg(func).reset_index(0, drop=True)
                    else:
                        if func == 'mean':
                            df[feature_name] = df[col].rolling(window, min_periods=min_periods).mean()
                        elif func == 'std':
                            df[feature_name] = df[col].rolling(window, min_periods=min_periods).std()
                        elif func == 'min':
                            df[feature_name] = df[col].rolling(window, min_periods=min_periods).min()
                        elif func == 'max':
                            df[feature_name] = df[col].rolling(window, min_periods=min_periods).max()
        
        return df
    
    @staticmethod
    def create_difference_features(df: Union[pd.DataFrame, 'dd.DataFrame'],
                                  columns: List[str],
                                  periods: List[int] = [1],
                                  group_by: Optional[List[str]] = None) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Cria features de diferença temporal.
        
        Args:
            df: DataFrame de entrada
            columns: Colunas para calcular diferenças
            periods: Períodos para diferença
            group_by: Colunas para agrupamento
            
        Returns:
            DataFrame com features de diferença
            
        Example:
            >>> df = create_difference_features(df, ['unemployment_rate'], [1, 2])
            Adiciona: unemployment_rate_diff1, unemployment_rate_diff2
        """
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for period in periods:
                diff_col = f"{col}_diff{period}"
                
                if group_by:
                    if is_dask:
                        df[diff_col] = df.groupby(group_by)[col].diff(period)
                    else:
                        df[diff_col] = df.groupby(group_by)[col].diff(period)
                else:
                    df[diff_col] = df[col].diff(period)
        
        return df
    
    @staticmethod
    def create_percentage_change_features(df: Union[pd.DataFrame, 'dd.DataFrame'],
                                         columns: List[str],
                                         periods: List[int] = [1],
                                         group_by: Optional[List[str]] = None) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Cria features de mudança percentual.
        
        Args:
            df: DataFrame de entrada
            columns: Colunas para calcular mudança percentual
            periods: Períodos para cálculo
            group_by: Colunas para agrupamento
            
        Returns:
            DataFrame com features de mudança percentual
        """
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for period in periods:
                pct_col = f"{col}_pct_change{period}"
                
                if group_by:
                    if is_dask:
                        df[pct_col] = df.groupby(group_by)[col].pct_change(period)
                    else:
                        df[pct_col] = df.groupby(group_by)[col].pct_change(period)
                else:
                    df[pct_col] = df[col].pct_change(period)
        
        return df
    
    @staticmethod
    def create_stratum_features(df: Union[pd.DataFrame, 'dd.DataFrame'],
                               columns: List[str],
                               stratum_col: str = 'country_stratum',
                               year_col: str = 'year',
                               operations: List[str] = ['mean', 'std']) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Cria features agregadas por estrato (grupo de países similares).
        
        Args:
            df: DataFrame de entrada
            columns: Colunas para agregação
            stratum_col: Coluna de estrato
            year_col: Coluna de ano
            operations: Operações de agregação
            
        Returns:
            DataFrame com features de estrato
            
        Note:
            Útil para capturar tendências regionais ou por grupo econômico.
        """
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for op in operations:
                feature_name = f"stratum_{col}_{op}"
                
                if is_dask:
                    # Dask: usar transform com agregação
                    if op == 'mean':
                        df[feature_name] = df.groupby([stratum_col, year_col])[col].transform('mean')
                    elif op == 'std':
                        df[feature_name] = df.groupby([stratum_col, year_col])[col].transform('std')
                    elif op == 'min':
                        df[feature_name] = df.groupby([stratum_col, year_col])[col].transform('min')
                    elif op == 'max':
                        df[feature_name] = df.groupby([stratum_col, year_col])[col].transform('max')
                else:
                    # Pandas
                    df[feature_name] = df.groupby([stratum_col, year_col])[col].transform(op)
        
        return df
    
    @staticmethod
    def create_interaction_features(df: Union[pd.DataFrame, 'dd.DataFrame'],
                                   feature_pairs: List[tuple],
                                   operations: List[str] = ['multiply', 'ratio']) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Cria features de interação entre variáveis.
        
        Args:
            df: DataFrame de entrada
            feature_pairs: Pares de features para interação
            operations: Tipos de interação ('multiply', 'ratio', 'diff')
            
        Returns:
            DataFrame com features de interação
            
        Example:
            >>> df = create_interaction_features(df, [('gdp', 'population')], ['ratio'])
            Adiciona: gdp_population_ratio (GDP per capita)
        """
        for feat1, feat2 in feature_pairs:
            if feat1 not in df.columns or feat2 not in df.columns:
                continue
                
            for op in operations:
                if op == 'multiply':
                    feature_name = f"{feat1}_{feat2}_multiply"
                    df[feature_name] = df[feat1] * df[feat2]
                elif op == 'ratio':
                    feature_name = f"{feat1}_{feat2}_ratio"
                    # Evitar divisão por zero
                    df[feature_name] = df[feat1] / df[feat2].replace(0, np.nan)
                elif op == 'diff':
                    feature_name = f"{feat1}_{feat2}_diff"
                    df[feature_name] = df[feat1] - df[feat2]
                elif op == 'sum':
                    feature_name = f"{feat1}_{feat2}_sum"
                    df[feature_name] = df[feat1] + df[feat2]
        
        return df
    
    @staticmethod
    def create_temporal_trend_features(df: Union[pd.DataFrame, 'dd.DataFrame'],
                                      columns: List[str],
                                      year_col: str = 'year',
                                      group_by: Optional[List[str]] = None) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Cria features de tendência temporal.
        
        Args:
            df: DataFrame de entrada
            columns: Colunas para análise de tendência
            year_col: Coluna de ano
            group_by: Colunas para agrupamento
            
        Returns:
            DataFrame com features de tendência temporal
            
        Note:
            Calcula tendência linear e taxa de crescimento média.
        """
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        if is_dask:
            # Para Dask, usar operações mais simples
            for col in columns:
                if col not in df.columns:
                    continue
                    
                # Taxa de crescimento ano a ano
                growth_col = f"{col}_growth_rate"
                if group_by:
                    df[growth_col] = df.groupby(group_by)[col].pct_change()
                else:
                    df[growth_col] = df[col].pct_change()
        else:
            # Para Pandas, podemos calcular tendência linear
            from scipy import stats
            
            def calculate_trend(group, col):
                """Calcula tendência linear para um grupo."""
                valid_data = group[[year_col, col]].dropna()
                if len(valid_data) < 2:
                    return pd.Series([np.nan] * len(group), index=group.index)
                    
                x = valid_data[year_col].values
                y = valid_data[col].values
                
                try:
                    slope, intercept, r_value, _, _ = stats.linregress(x, y)
                    trend = slope * group[year_col] + intercept
                    return pd.Series(trend, index=group.index)
                except:
                    return pd.Series([np.nan] * len(group), index=group.index)
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                trend_col = f"{col}_trend"
                
                if group_by:
                    df[trend_col] = df.groupby(group_by).apply(
                        lambda g: calculate_trend(g, col)
                    ).reset_index(level=0, drop=True)
                else:
                    df[trend_col] = calculate_trend(df, col)
                
                # Taxa de crescimento também
                growth_col = f"{col}_growth_rate"
                if group_by:
                    df[growth_col] = df.groupby(group_by)[col].pct_change()
                else:
                    df[growth_col] = df[col].pct_change()
        
        return df
    
    @staticmethod
    def normalize_features(df: Union[pd.DataFrame, 'dd.DataFrame'],
                          columns: List[str],
                          method: str = 'standard',
                          group_by: Optional[List[str]] = None) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Normaliza features numéricas.
        
        Args:
            df: DataFrame de entrada
            columns: Colunas para normalizar
            method: Método de normalização ('standard', 'minmax', 'robust')
            group_by: Colunas para normalização por grupo
            
        Returns:
            DataFrame com features normalizadas
        """
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            norm_col = f"{col}_normalized"
            
            if method == 'standard':
                # Z-score normalization
                if group_by:
                    if is_dask:
                        mean = df.groupby(group_by)[col].transform('mean')
                        std = df.groupby(group_by)[col].transform('std')
                    else:
                        mean = df.groupby(group_by)[col].transform('mean')
                        std = df.groupby(group_by)[col].transform('std')
                else:
                    mean = df[col].mean()
                    std = df[col].std()
                
                df[norm_col] = (df[col] - mean) / std.replace(0, 1)
                
            elif method == 'minmax':
                # Min-Max normalization [0, 1]
                if group_by:
                    if is_dask:
                        min_val = df.groupby(group_by)[col].transform('min')
                        max_val = df.groupby(group_by)[col].transform('max')
                    else:
                        min_val = df.groupby(group_by)[col].transform('min')
                        max_val = df.groupby(group_by)[col].transform('max')
                else:
                    min_val = df[col].min()
                    max_val = df[col].max()
                
                range_val = max_val - min_val
                if is_dask:
                    df[norm_col] = (df[col] - min_val) / range_val.replace(0, 1)
                else:
                    df[norm_col] = (df[col] - min_val) / range_val.replace(0, 1) if isinstance(range_val, pd.Series) else (df[col] - min_val) / (range_val if range_val != 0 else 1)
                    
            elif method == 'robust':
                # Robust normalization using median and IQR
                if group_by:
                    if is_dask:
                        median = df.groupby(group_by)[col].transform('median')
                        q1 = df.groupby(group_by)[col].transform(lambda x: x.quantile(0.25))
                        q3 = df.groupby(group_by)[col].transform(lambda x: x.quantile(0.75))
                    else:
                        median = df.groupby(group_by)[col].transform('median')
                        q1 = df.groupby(group_by)[col].transform(lambda x: x.quantile(0.25))
                        q3 = df.groupby(group_by)[col].transform(lambda x: x.quantile(0.75))
                else:
                    median = df[col].median()
                    q1 = df[col].quantile(0.25) if not is_dask else df[col].quantile(0.25).compute()
                    q3 = df[col].quantile(0.75) if not is_dask else df[col].quantile(0.75).compute()
                
                iqr = q3 - q1
                if is_dask:
                    df[norm_col] = (df[col] - median) / iqr.replace(0, 1)
                else:
                    df[norm_col] = (df[col] - median) / iqr.replace(0, 1) if isinstance(iqr, pd.Series) else (df[col] - median) / (iqr if iqr != 0 else 1)
        
        return df
    
    @staticmethod
    def handle_missing_values(df: Union[pd.DataFrame, 'dd.DataFrame'],
                            columns: List[str],
                            method: str = 'forward_fill',
                            group_by: Optional[List[str]] = None,
                            limit: Optional[int] = None) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Trata valores ausentes de forma científica.
        
        Args:
            df: DataFrame de entrada
            columns: Colunas para tratamento
            method: Método de imputação ('forward_fill', 'backward_fill', 'interpolate', 'mean', 'median')
            group_by: Colunas para tratamento por grupo
            limit: Limite de preenchimentos consecutivos
            
        Returns:
            DataFrame com valores ausentes tratados
            
        Note:
            Preserva gaps temporais científicos quando necessário.
        """
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'forward_fill':
                if group_by:
                    if is_dask:
                        df[col] = df.groupby(group_by)[col].fillna(method='ffill', limit=limit)
                    else:
                        df[col] = df.groupby(group_by)[col].fillna(method='ffill', limit=limit)
                else:
                    df[col] = df[col].fillna(method='ffill', limit=limit)
                    
            elif method == 'backward_fill':
                if group_by:
                    if is_dask:
                        df[col] = df.groupby(group_by)[col].fillna(method='bfill', limit=limit)
                    else:
                        df[col] = df.groupby(group_by)[col].fillna(method='bfill', limit=limit)
                else:
                    df[col] = df[col].fillna(method='bfill', limit=limit)
                    
            elif method == 'interpolate':
                if not is_dask:
                    if group_by:
                        df[col] = df.groupby(group_by)[col].apply(
                            lambda x: x.interpolate(method='linear', limit=limit)
                        )
                    else:
                        df[col] = df[col].interpolate(method='linear', limit=limit)
                        
            elif method == 'mean':
                if group_by:
                    if is_dask:
                        mean_val = df.groupby(group_by)[col].transform('mean')
                        df[col] = df[col].fillna(mean_val)
                    else:
                        df[col] = df.groupby(group_by)[col].transform(
                            lambda x: x.fillna(x.mean())
                        )
                else:
                    df[col] = df[col].fillna(df[col].mean())
                    
            elif method == 'median':
                if group_by:
                    if is_dask:
                        median_val = df.groupby(group_by)[col].transform('median')
                        df[col] = df[col].fillna(median_val)
                    else:
                        df[col] = df.groupby(group_by)[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                else:
                    median_val = df[col].median() if not is_dask else df[col].median().compute()
                    df[col] = df[col].fillna(median_val)
        
        return df
    
    @staticmethod
    def validate_feature_quality(df: Union[pd.DataFrame, 'dd.DataFrame'],
                                columns: List[str],
                                min_non_null_ratio: float = 0.5,
                                max_constant_ratio: float = 0.95) -> Dict[str, Dict[str, Any]]:
        """
        Valida qualidade das features criadas.
        
        Args:
            df: DataFrame com features
            columns: Colunas para validar
            min_non_null_ratio: Proporção mínima de valores não-nulos
            max_constant_ratio: Proporção máxima de valores constantes
            
        Returns:
            Dicionário com estatísticas de qualidade por feature
        """
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        quality_stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if is_dask:
                # Dask: computar estatísticas básicas
                total_count = df[col].count().compute()
                null_count = df[col].isna().sum().compute()
                unique_count = df[col].nunique().compute()
                
                # Para verificar valores constantes, usar value_counts
                value_counts = df[col].value_counts().compute()
                if len(value_counts) > 0:
                    most_common_ratio = value_counts.iloc[0] / total_count
                else:
                    most_common_ratio = 0
            else:
                # Pandas
                total_count = len(df)
                null_count = df[col].isna().sum()
                unique_count = df[col].nunique()
                
                value_counts = df[col].value_counts()
                if len(value_counts) > 0:
                    most_common_ratio = value_counts.iloc[0] / total_count
                else:
                    most_common_ratio = 0
            
            non_null_ratio = (total_count - null_count) / total_count if total_count > 0 else 0
            
            quality_stats[col] = {
                'total_count': int(total_count),
                'null_count': int(null_count),
                'non_null_ratio': float(non_null_ratio),
                'unique_count': int(unique_count),
                'most_common_ratio': float(most_common_ratio),
                'is_valid': non_null_ratio >= min_non_null_ratio and most_common_ratio <= max_constant_ratio,
                'validation_timestamp': datetime.now().isoformat()
            }
        
        return quality_stats