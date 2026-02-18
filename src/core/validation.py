#!/usr/bin/env python3
"""
Módulo centralizado de validação para arquiteturas ML.

Centraliza toda lógica de validação temporal, integridade de dados e
métricas científicas, eliminando duplicação entre arquiteturas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime


class TemporalValidator:
    """
    Validador temporal para prevenção de vazamento em séries temporais.
    
    Implementa validação rigorosa de splits temporais com gaps obrigatórios
    para garantir validade científica em previsão de dropout educacional.
    """
    
    def __init__(self, min_gap_years: int = 2):
        """
        Inicializa validador temporal.
        
        Args:
            min_gap_years: Gap mínimo em anos entre splits (default: 2)
        """
        self.min_gap_years = min_gap_years
    
    def validate_fold_integrity(self, fold: Dict) -> Tuple[bool, List[str]]:
        """
        Valida integridade completa de um fold temporal.
        
        Args:
            fold: Dicionário com configuração do fold
            
        Returns:
            Tupla (is_valid, lista_de_erros)
        """
        errors = []
        
        # Verificar campos obrigatórios
        required_fields = [
            'train_start', 'train_end', 'val_start', 'val_end',
            'test_start', 'test_end'
        ]
        
        for field in required_fields:
            if field not in fold:
                errors.append(f"Campo obrigatório ausente: {field}")
        
        if errors:
            return False, errors
        
        # Verificar ordem cronológica
        if fold['train_start'] > fold['train_end']:
            errors.append(f"Train: início ({fold['train_start']}) > fim ({fold['train_end']})")
        
        if fold['val_start'] > fold['val_end']:
            errors.append(f"Val: início ({fold['val_start']}) > fim ({fold['val_end']})")
        
        if fold['test_start'] > fold['test_end']:
            errors.append(f"Test: início ({fold['test_start']}) > fim ({fold['test_end']})")
        
        # Verificar sequência temporal
        if fold['train_end'] >= fold['val_start']:
            errors.append(f"Sobreposição train-val: train_end={fold['train_end']}, val_start={fold['val_start']}")
        
        if fold['val_end'] >= fold['test_start']:
            errors.append(f"Sobreposição val-test: val_end={fold['val_end']}, test_start={fold['test_start']}")
        
        # Verificar gaps mínimos
        train_val_gap = fold['val_start'] - fold['train_end'] - 1
        val_test_gap = fold['test_start'] - fold['val_end'] - 1
        
        if train_val_gap < self.min_gap_years:
            errors.append(f"Gap train-val insuficiente: {train_val_gap} < {self.min_gap_years}")
        
        if val_test_gap < self.min_gap_years:
            errors.append(f"Gap val-test insuficiente: {val_test_gap} < {self.min_gap_years}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_walk_forward(self, folds: List[Dict]) -> Tuple[bool, Dict]:
        """
        Valida estrutura walk-forward de múltiplos folds.
        
        Args:
            folds: Lista de folds para validação
            
        Returns:
            Tupla (is_valid, relatório_detalhado)
        """
        report = {
            'total_folds': len(folds),
            'valid_folds': 0,
            'invalid_folds': 0,
            'fold_errors': {},
            'walk_forward_valid': True,
            'expanding_window': True
        }
        
        for i, fold in enumerate(folds):
            is_valid, errors = self.validate_fold_integrity(fold)
            
            if is_valid:
                report['valid_folds'] += 1
            else:
                report['invalid_folds'] += 1
                report['fold_errors'][f'fold_{i}'] = errors
        
        # Verificar se é walk-forward expansivo
        if len(folds) > 1:
            for i in range(1, len(folds)):
                # Train deve expandir ou manter
                if folds[i]['train_end'] < folds[i-1]['train_end']:
                    report['expanding_window'] = False
                    report['walk_forward_valid'] = False
                    break
        
        report['all_valid'] = report['invalid_folds'] == 0

        return report['all_valid'], report

    def enforce_walk_forward(self, folds: List[Dict]) -> None:
        """
        Valida estrutura walk-forward e interrompe execução em caso de violação.

        Raises:
            ValueError: Se qualquer fold violar integridade temporal
        """
        all_valid, report = self.validate_walk_forward(folds)
        if not all_valid:
            errors = report.get('fold_errors', {})
            raise ValueError(
                f"Anti-leakage violation: {report['invalid_folds']} of "
                f"{report['total_folds']} folds failed temporal integrity. "
                f"Errors: {errors}"
            )
    
    def validate_gap(self, start_year: int, end_year: int) -> bool:
        """
        Valida se o gap entre dois anos é suficiente.
        
        Args:
            start_year: Ano inicial
            end_year: Ano final
            
        Returns:
            True se gap é suficiente, False caso contrário
        """
        gap = end_year - start_year
        return gap >= self.min_gap_years
    
    def check_temporal_leakage(self, train_years: Tuple[int, int],
                              test_years: Tuple[int, int]) -> bool:
        """
        Verifica se há vazamento temporal entre treino e teste.
        
        Args:
            train_years: Tupla (início, fim) do treino
            test_years: Tupla (início, fim) do teste
            
        Returns:
            True se há vazamento, False caso contrário
        """
        # Há vazamento se test começar antes do treino terminar
        return test_years[0] <= train_years[1]
    
    def validate_temporal_split(self, train_data: pd.DataFrame,
                              val_data: pd.DataFrame,
                              test_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Valida split temporal usando DataFrames.
        
        Args:
            train_data: Dados de treino
            val_data: Dados de validação
            test_data: Dados de teste
            
        Returns:
            Tupla (is_valid, message)
        """
        # Extrair anos dos dados
        if 'year' in train_data.columns:
            train_years = (train_data['year'].min(), train_data['year'].max())
            val_years = (val_data['year'].min(), val_data['year'].max())
            test_years = (test_data['year'].min(), test_data['year'].max())
            
            # Verificar ordem temporal
            if train_years[1] >= val_years[0]:
                return False, f"Sobreposição train-val: {train_years[1]} >= {val_years[0]}"
            
            if val_years[1] >= test_years[0]:
                return False, f"Sobreposição val-test: {val_years[1]} >= {test_years[0]}"
            
            # Verificar gaps
            train_val_gap = val_years[0] - train_years[1] - 1
            val_test_gap = test_years[0] - val_years[1] - 1
            
            if train_val_gap < self.min_gap_years:
                return False, f"Gap train-val insuficiente: {train_val_gap} < {self.min_gap_years}"
            
            if val_test_gap < self.min_gap_years:
                return False, f"Gap val-test insuficiente: {val_test_gap} < {self.min_gap_years}"
            
            return True, "Split temporal válido"
        
        return True, "Coluna 'year' não encontrada para validação"


class DataIntegrityValidator:
    """
    Validador de integridade de dados para ML.
    
    Verifica qualidade, completude e consistência dos dados
    antes do treinamento de modelos.
    """
    
    def __init__(self):
        """Inicializa validador de integridade."""
        self.validation_results = {}
    
    def validate_target_distribution(self, target_values: np.ndarray,
                                    expected_range: Tuple[float, float] = (0, 100),
                                    name: str = "target") -> Dict:
        """
        Valida distribuição da variável target.
        
        Args:
            target_values: Valores do target
            expected_range: Range esperado (min, max)
            name: Nome da variável para relatório
            
        Returns:
            Dicionário com análise da distribuição
        """
        # Remover NaN para análise
        clean_values = target_values[~np.isnan(target_values)]
        
        validation = {
            'variable': name,
            'total_observations': len(target_values),
            'valid_observations': len(clean_values),
            'missing_count': len(target_values) - len(clean_values),
            'missing_rate': (len(target_values) - len(clean_values)) / len(target_values) * 100
        }
        
        if len(clean_values) > 0:
            validation.update({
                'mean': float(np.mean(clean_values)),
                'std': float(np.std(clean_values)),
                'min': float(np.min(clean_values)),
                'max': float(np.max(clean_values)),
                'median': float(np.median(clean_values)),
                'q25': float(np.percentile(clean_values, 25)),
                'q75': float(np.percentile(clean_values, 75))
            })
            
            # Verificar range
            out_of_range = np.sum((clean_values < expected_range[0]) | 
                                 (clean_values > expected_range[1]))
            validation['out_of_range_count'] = int(out_of_range)
            validation['out_of_range_rate'] = float(out_of_range / len(clean_values) * 100)
            
            # Verificar valores negativos (não deveria existir para dropout)
            negative_count = np.sum(clean_values < 0)
            validation['negative_values'] = int(negative_count)
            
            # Alertas
            validation['warnings'] = []
            
            if validation['missing_rate'] > 20:
                validation['warnings'].append(f"Alta taxa de missing: {validation['missing_rate']:.1f}%")
            
            if validation['out_of_range_rate'] > 5:
                validation['warnings'].append(f"Valores fora do range: {validation['out_of_range_rate']:.1f}%")
            
            if negative_count > 0:
                validation['warnings'].append(f"Valores negativos detectados: {negative_count}")
            
            if validation['std'] < 1:
                validation['warnings'].append(f"Baixa variabilidade: std={validation['std']:.2f}")
        else:
            validation['warnings'] = ["Sem dados válidos para análise"]
        
        validation['is_valid'] = len(validation.get('warnings', [])) == 0
        
        return validation
    
    def validate_feature_quality(self, features_df: pd.DataFrame,
                                max_missing_rate: float = 0.5,
                                min_variance: float = 0.01) -> Dict:
        """
        Valida qualidade das features para ML.
        
        Args:
            features_df: DataFrame com features
            max_missing_rate: Taxa máxima de missing aceitável
            min_variance: Variância mínima para feature ser útil
            
        Returns:
            Dicionário com análise de qualidade
        """
        quality_report = {
            'total_features': len(features_df.columns),
            'features_analyzed': [],
            'high_missing_features': [],
            'low_variance_features': [],
            'constant_features': [],
            'recommended_to_remove': []
        }
        
        for col in features_df.columns:
            feature_analysis = {
                'name': col,
                'dtype': str(features_df[col].dtype),
                'missing_rate': features_df[col].isna().mean(),
                'unique_values': features_df[col].nunique(),
                'variance': features_df[col].var() if features_df[col].dtype in ['float64', 'int64'] else None
            }
            
            # Verificar problemas
            if feature_analysis['missing_rate'] > max_missing_rate:
                quality_report['high_missing_features'].append(col)
                quality_report['recommended_to_remove'].append(col)
            
            if feature_analysis['unique_values'] == 1:
                quality_report['constant_features'].append(col)
                quality_report['recommended_to_remove'].append(col)
            
            elif feature_analysis['variance'] is not None and feature_analysis['variance'] < min_variance:
                quality_report['low_variance_features'].append(col)
            
            quality_report['features_analyzed'].append(feature_analysis)
        
        # Remover duplicatas das recomendações
        quality_report['recommended_to_remove'] = list(set(quality_report['recommended_to_remove']))
        
        quality_report['quality_score'] = 1.0 - (len(quality_report['recommended_to_remove']) / 
                                                 quality_report['total_features'])
        
        return quality_report
    
    def validate_dataframe(self, df: pd.DataFrame,
                          target_col: str = None,
                          check_completeness: bool = True) -> Tuple[bool, Dict]:
        """
        Valida integridade completa de um DataFrame.
        
        Args:
            df: DataFrame para validar
            target_col: Nome da coluna target (opcional)
            check_completeness: Se deve verificar completude
            
        Returns:
            Tupla (is_valid, validation_report)
        """
        validation_report = {
            'is_valid': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_data': {},
            'warnings': [],
            'errors': []
        }
        
        # Verificar se DataFrame está vazio
        if df.empty:
            validation_report['is_valid'] = False
            validation_report['errors'].append("DataFrame está vazio")
            return False, validation_report
        
        # Verificar dados faltantes
        missing_counts = df.isnull().sum()
        missing_rates = (missing_counts / len(df)) * 100
        
        for col in df.columns:
            if missing_counts[col] > 0:
                validation_report['missing_data'][col] = {
                    'count': int(missing_counts[col]),
                    'rate': float(missing_rates[col])
                }
                
                # Se completude é necessária
                if check_completeness and missing_rates[col] > 50:
                    validation_report['warnings'].append(
                        f"Coluna '{col}' tem {missing_rates[col]:.1f}% de dados faltantes"
                    )
        
        # Validar target se especificado
        if target_col and target_col in df.columns:
            target_validation = self.validate_target_distribution(
                df[target_col].values,
                name=target_col
            )
            validation_report['target_validation'] = target_validation
            
            if not target_validation['is_valid']:
                validation_report['warnings'].extend(target_validation.get('warnings', []))
        
        # Verificar colunas numéricas com variância zero
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].var() == 0:
                validation_report['warnings'].append(f"Coluna '{col}' tem variância zero")
        
        # Verificar valores infinitos
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        for col, count in inf_counts.items():
            if count > 0:
                validation_report['warnings'].append(f"Coluna '{col}' tem {count} valores infinitos")
                validation_report['is_valid'] = False
        
        # Verificar duplicatas
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_report['warnings'].append(f"DataFrame tem {duplicates} linhas duplicadas")
        
        # Determinar validade final
        if validation_report['errors']:
            validation_report['is_valid'] = False
        
        # Se há muitos warnings, pode indicar problemas
        if len(validation_report['warnings']) > 5:
            validation_report['is_valid'] = False
            validation_report['errors'].append("Muitos problemas de qualidade detectados")
        
        return validation_report['is_valid'], validation_report
    
    def check_data_leakage(self, train_data: Any, test_data: Any,
                          id_columns: List[str] = None) -> Dict:
        """
        Verifica possível vazamento de dados entre train e test.
        
        Args:
            train_data: Dados de treino
            test_data: Dados de teste
            id_columns: Colunas de identificação para verificar duplicatas
            
        Returns:
            Dicionário com análise de vazamento
        """
        leakage_report = {
            'leakage_detected': False,
            'duplicate_rows': 0,
            'duplicate_ids': [],
            'temporal_overlap': False
        }
        
        # Verificar duplicatas de IDs se fornecidos
        if id_columns and hasattr(train_data, '__getitem__'):
            train_ids = set()
            test_ids = set()
            
            for col in id_columns:
                if col in train_data and col in test_data:
                    train_ids.update(train_data[col])
                    test_ids.update(test_data[col])
            
            overlap = train_ids.intersection(test_ids)
            if overlap:
                leakage_report['leakage_detected'] = True
                leakage_report['duplicate_ids'] = list(overlap)[:10]  # Primeiros 10
                leakage_report['duplicate_count'] = len(overlap)
        
        # Verificar overlap temporal se houver coluna 'year'
        if hasattr(train_data, '__getitem__') and 'year' in train_data and 'year' in test_data:
            train_years = set(train_data['year'])
            test_years = set(test_data['year'])
            
            if train_years.intersection(test_years):
                leakage_report['temporal_overlap'] = True
                leakage_report['leakage_detected'] = True
                leakage_report['overlapping_years'] = sorted(list(train_years.intersection(test_years)))
        
        return leakage_report


class MetricsValidator:
    """
    Validador de métricas para garantir confiabilidade dos resultados.
    """
    
    def __init__(self):
        """Inicializa validador de métricas."""
        self.metrics_history = []
    
    def validate_metrics_consistency(self, metrics: Dict,
                                    expected_ranges: Dict = None) -> Dict:
        """
        Valida consistência e razoabilidade das métricas.
        
        Args:
            metrics: Dicionário com métricas calculadas
            expected_ranges: Ranges esperados para cada métrica
            
        Returns:
            Dicionário com validação das métricas
        """
        if expected_ranges is None:
            # Ranges padrão para regressão de dropout
            expected_ranges = {
                'r2': (-1, 1),
                'mape': (0, 100),
                'rmse': (0, 50),
                'mae': (0, 50)
            }
        
        validation = {
            'metrics_validated': [],
            'out_of_range': [],
            'suspicious_values': [],
            'is_valid': True
        }
        
        for metric_name, value in metrics.items():
            if value is None or np.isnan(value):
                validation['suspicious_values'].append({
                    'metric': metric_name,
                    'issue': 'null_or_nan'
                })
                validation['is_valid'] = False
                continue
            
            # Verificar range
            if metric_name in expected_ranges:
                min_val, max_val = expected_ranges[metric_name]
                if value < min_val or value > max_val:
                    validation['out_of_range'].append({
                        'metric': metric_name,
                        'value': value,
                        'expected_range': expected_ranges[metric_name]
                    })
                    validation['is_valid'] = False
            
            # Verificações específicas
            if metric_name == 'r2' and value < -0.5:
                validation['suspicious_values'].append({
                    'metric': 'r2',
                    'value': value,
                    'issue': 'muito_negativo_indica_modelo_muito_ruim'
                })
            
            if metric_name == 'mape' and value > 50:
                validation['suspicious_values'].append({
                    'metric': 'mape',
                    'value': value,
                    'issue': 'erro_percentual_muito_alto'
                })
            
            validation['metrics_validated'].append({
                'metric': metric_name,
                'value': value,
                'valid': True
            })
        
        # Armazenar no histórico
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'validation': validation
        })
        
        return validation
    
    def check_overfitting(self, train_metrics: Dict, test_metrics: Dict,
                         threshold: float = 0.15) -> Dict:
        """
        Verifica sinais de overfitting comparando métricas train vs test.
        
        Args:
            train_metrics: Métricas no conjunto de treino
            test_metrics: Métricas no conjunto de teste
            threshold: Diferença máxima aceitável
            
        Returns:
            Dicionário com análise de overfitting
        """
        overfitting_analysis = {
            'overfitting_detected': False,
            'gaps': {},
            'severity': 'none'
        }
        
        # Calcular gaps para métricas comuns
        common_metrics = set(train_metrics.keys()).intersection(set(test_metrics.keys()))
        
        for metric in common_metrics:
            if train_metrics[metric] is not None and test_metrics[metric] is not None:
                # Para R2, maior é melhor
                if metric == 'r2':
                    gap = train_metrics[metric] - test_metrics[metric]
                # Para erros, menor é melhor
                else:
                    gap = test_metrics[metric] - train_metrics[metric]
                
                overfitting_analysis['gaps'][metric] = gap
                
                if abs(gap) > threshold:
                    overfitting_analysis['overfitting_detected'] = True
        
        # Classificar severidade
        if overfitting_analysis['overfitting_detected']:
            max_gap = max(abs(g) for g in overfitting_analysis['gaps'].values())
            
            if max_gap > 0.3:
                overfitting_analysis['severity'] = 'high'
            elif max_gap > 0.2:
                overfitting_analysis['severity'] = 'medium'
            else:
                overfitting_analysis['severity'] = 'low'
        
        return overfitting_analysis


# Funções auxiliares para uso direto
def validate_temporal_splits(train_years: Tuple[int, int],
                            val_years: Tuple[int, int],
                            test_years: Tuple[int, int],
                            min_gap: int = 2) -> bool:
    """
    Função conveniente para validação rápida de splits temporais.
    
    Args:
        train_years: (início, fim) do período de treino
        val_years: (início, fim) do período de validação
        test_years: (início, fim) do período de teste
        min_gap: Gap mínimo entre períodos
        
    Returns:
        True se splits são válidos, False caso contrário
    """
    validator = TemporalValidator(min_gap_years=min_gap)
    
    fold = {
        'train_start': train_years[0],
        'train_end': train_years[1],
        'val_start': val_years[0],
        'val_end': val_years[1],
        'test_start': test_years[0],
        'test_end': test_years[1]
    }
    
    is_valid, errors = validator.validate_fold_integrity(fold)

    if not is_valid:
        raise ValueError(
            f"Anti-leakage violation in temporal split: {errors}"
        )

    return is_valid


def generate_validation_report(data: pd.DataFrame, 
                              target_col: str,
                              feature_cols: List[str]) -> Dict:
    """
    Gera relatório completo de validação dos dados.
    
    Args:
        data: DataFrame com os dados
        target_col: Nome da coluna target
        feature_cols: Lista de colunas de features
        
    Returns:
        Dicionário com relatório completo de validação
    """
    data_validator = DataIntegrityValidator()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': data.shape,
        'target_analysis': {},
        'features_analysis': {},
        'recommendations': []
    }
    
    # Validar target
    if target_col in data.columns:
        report['target_analysis'] = data_validator.validate_target_distribution(
            data[target_col].values,
            name=target_col
        )
    
    # Validar features
    if feature_cols:
        features_df = data[feature_cols]
        report['features_analysis'] = data_validator.validate_feature_quality(features_df)
        
        # Adicionar recomendações
        if report['features_analysis']['recommended_to_remove']:
            report['recommendations'].append(
                f"Remover {len(report['features_analysis']['recommended_to_remove'])} features problemáticas"
            )
    
    # Adicionar warnings do target
    if 'warnings' in report['target_analysis']:
        report['recommendations'].extend(report['target_analysis']['warnings'])
    
    return report