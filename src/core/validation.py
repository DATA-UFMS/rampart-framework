#!/usr/bin/env python3
"""
Módulo centralizado de validação para arquiteturas ML.

Centraliza toda lógica de validação temporal, integridade de dados e
métricas científicas, eliminando duplicação entre arquiteturas.

Protocolo anti-leakage (P1-P5):
    P1 — Ordenação temporal: train_end < val_start < val_end < test_start.
    P2 — Gap mínimo: N anos entre splits (default 2), configurável via
         temporal_gap_years. Embargo opcional para dados sub-anuais.
    P3 — Separação de features: lista de exclusão (derivadas do target,
         metadados) + detecção de proxy (|r| > 0.95 com target).
    P4 — Escopo de seleção: feature selection restrita ao período de
         treino do primeiro fold (Kapoor & Narayanan, 2023).
    P5 — Escopo de preprocessing: scaling e imputação ajustados
         exclusivamente nos dados de treino (Kaufman et al. 2012).

HPO: grid search no conjunto de validação; modelo final retreinado
no treino completo. Previne leakage por otimização no teste (Kapoor & Narayanan, 2023).

Enforcement: violações de P1/P2 geram ValueError via enforce_walk_forward().
Violações de P3/P4 geram ValueError em run_feature_selection().
P5 é enforced por contrato (docstring + testes unitários).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime


class TemporalValidator:
    """
    Validador temporal para prevenção de vazamento em séries temporais.

    Implementa validação de splits temporais com gaps obrigatórios
    e embargo configurável para garantir validade científica em previsão
    de dropout educacional.

    O protocolo combina dois mecanismos complementares:
      - **Gap temporal**: período mínimo entre splits consecutivos,
        impedindo que informação futura influencie o treino.
      - **Embargo**: observações adjacentes ao limite de cada split
        são excluídas do treino para prevenir leakage por
        autocorrelação residual (López de Prado, 2018).

    Nota sobre purging (López de Prado 2018):
        Purging remove observações de treino cujos labels sobrepõem
        temporalmente o período de teste. Em dados com granularidade
        anual (um ponto por país/ano), não há sobreposição de labels
        entre splits — cada observação é um ponto discreto. Portanto,
        purging é desnecessário neste contexto. O gap temporal de N
        anos já subsume o efeito do embargo para dados anuais, pois
        não existem observações sub-anuais intermediárias a excluir.
        O parâmetro embargo_years existe para uso em adaptações do
        framework a dados de maior frequência (mensal, diário).
    """

    def __init__(self, min_gap_years: int = 2, embargo_years: int = 0):
        """
        Inicializa validador temporal.

        Args:
            min_gap_years: Gap mínimo em anos entre splits (default: 2).
                Controla a separação temporal obrigatória entre períodos.
            embargo_years: Período adicional de embargo em anos (default: 0).
                Quando > 0, observações no intervalo [train_end+1,
                train_end+embargo] são excluídas do treino, mesmo que
                já estejam fora do split de treino. Previne leakage
                por autocorrelação residual em dados com dependência
                temporal (lagged features, médias móveis).
        """
        self.min_gap_years = min_gap_years
        self.embargo_years = embargo_years
    
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

        # Verificar embargo: o gap efetivo deve cobrir também o embargo
        if self.embargo_years > 0:
            effective_gap_tv = train_val_gap - self.embargo_years
            effective_gap_vt = val_test_gap - self.embargo_years
            if effective_gap_tv < 0:
                errors.append(
                    f"Embargo train-val violado: gap={train_val_gap} < "
                    f"embargo={self.embargo_years}"
                )
            if effective_gap_vt < 0:
                errors.append(
                    f"Embargo val-test violado: gap={val_test_gap} < "
                    f"embargo={self.embargo_years}"
                )

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
    


class DataIntegrityValidator:
    """
    Validador de integridade de dados para ML.
    
    Verifica qualidade, completude e consistência dos dados
    antes do treinamento de modelos.
    """
    
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
        
        if df.empty:
            validation_report['is_valid'] = False
            validation_report['errors'].append("DataFrame está vazio")
            return False, validation_report
        
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
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].var() == 0:
                validation_report['warnings'].append(f"Coluna '{col}' tem variância zero")
        
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        for col, count in inf_counts.items():
            if count > 0:
                validation_report['warnings'].append(f"Coluna '{col}' tem {count} valores infinitos")
                validation_report['is_valid'] = False
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_report['warnings'].append(f"DataFrame tem {duplicates} linhas duplicadas")
        
        # Determinar validade final
        if validation_report['errors']:
            validation_report['is_valid'] = False
        
        # Heurística: mais de MAX_TOLERABLE_WARNINGS indica dataset degradado
        MAX_TOLERABLE_WARNINGS = 5
        if len(validation_report['warnings']) > MAX_TOLERABLE_WARNINGS:
            validation_report['is_valid'] = False
            validation_report['errors'].append(
                f"Número de warnings ({len(validation_report['warnings'])}) "
                f"excede o limite tolerável ({MAX_TOLERABLE_WARNINGS})"
            )
        
        return validation_report['is_valid'], validation_report
    
