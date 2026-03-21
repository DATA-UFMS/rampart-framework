#!/usr/bin/env python3
"""
Módulo de Validação de Equivalência Científica para o Benchmark.

Este módulo contém a classe `BenchmarkValidator`, responsável por
comparar os resultados e artefatos gerados pelos pipelines
Data Warehouse e Data Lake para garantir que, apesar das diferenças
arquiteturais, a lógica científica seja estatisticamente equivalente.
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple

from core.scientific_config import SCIENTIFIC_CONFIG


class BenchmarkValidator:
    """
    Valida a equivalência científica entre os resultados de duas arquiteturas.

    Carrega os artefatos de saída (estatísticas, seleções de features, etc.)
    de cada pipeline e compara-os usando as tolerâncias definidas no
    `SCIENTIFIC_CONFIG`.
    """

    def __init__(self, dw_results_meta: Dict, dl_results_meta: Dict):
        """
        Inicializa o validador com os metadados dos resultados.

        Args:
            dw_results_meta: Dicionário com metadados do Data Warehouse.
                             Deve conter 'output_base_path'.
            dl_results_meta: Dicionário com metadados do Data Lake.
                             Deve conter 'output_base_path'.
        """
        self.config = SCIENTIFIC_CONFIG
        self.dw_path = dw_results_meta['output_base_path']
        self.dl_path = dl_results_meta['output_base_path']
        
        self.dw_artifacts = self._load_artifacts(self.dw_path, 'data_warehouse')
        self.dl_artifacts = self._load_artifacts(self.dl_path, 'data_lake')

        self.report = {
            'validation_summary': {},
            'acceptable_differences': [],
            'critical_divergences': []
        }

    def _load_artifacts(self, base_path: str, arch_name: str) -> Dict[str, Any]:
        """Carrega os artefatos JSON de um pipeline para validação."""
        artifacts = {}
        try:
            with open(f"{base_path}/prep/target_statistics.json", 'r') as f:
                artifacts['target_stats'] = json.load(f)
            
            with open(f"{base_path}/prep/feature_selection_{arch_name}.json", 'r') as f:
                artifacts['feature_selection'] = json.load(f)

            with open(f"{base_path}/prep/temporal_folds_{arch_name}.json", 'r') as f:
                artifacts['folds_config'] = json.load(f)

        except FileNotFoundError as e:
            print(f"AVISO: Artefato não encontrado para {arch_name}: {e}")
        return artifacts

    def validate_all(self) -> Dict:
        """Executa todas as validações de equivalência."""
        print("Executando validação de equivalência científica completa...")
        
        self.validate_target_statistics()
        self.validate_feature_selection_overlap()
        self.validate_feature_correlations()
        self.validate_fold_integrity()
        
        # Resumir resultados
        summary = {
            'overall_status': 'PASSED' if not self.report['critical_divergences'] else 'FAILED',
            'total_critical_divergences': len(self.report['critical_divergences']),
            'total_acceptable_differences': len(self.report['acceptable_differences'])
        }
        self.report['validation_summary'] = summary
        
        print(f"   -> Status Final: {summary['overall_status']}")

        if self.report['critical_divergences']:
            raise ValueError(
                f"Cross-architecture validation FAILED: "
                f"{len(self.report['critical_divergences'])} critical divergences. "
                f"Details: {self.report['critical_divergences']}"
            )

        return self.report

    def validate_target_statistics(self) -> bool:
        """Valida se as estatísticas do target são equivalentes."""
        dw_stats = self.dw_artifacts.get('target_stats', {})
        dl_stats = self.dl_artifacts.get('target_stats', {})
        max_diff = self.config['target_stats_max_diff']
        is_equivalent = True

        for stat in ['mean', 'std', 'min', 'max']:
            dw_val = dw_stats.get(stat, 0)
            dl_val = dl_stats.get(stat, 0)
            diff = abs(dw_val - dl_val) / (abs(dw_val) + 1e-9)
            
            if diff > max_diff:
                is_equivalent = False
                self.report['critical_divergences'].append({
                    'check': 'target_statistics',
                    'metric': stat,
                    'dw_value': dw_val,
                    'dl_value': dl_val,
                    'relative_difference': diff,
                    'threshold': max_diff
                })
        
        print(f"   - Validação de Estatísticas do Target: {'OK' if is_equivalent else 'FALHOU'}")
        return is_equivalent

    def validate_feature_selection_overlap(self) -> bool:
        """Valida se o overlap das features selecionadas é aceitável."""
        dw_feats = set(self.dw_artifacts.get('feature_selection', {}).get('selected_features', []))
        dl_feats = set(self.dl_artifacts.get('feature_selection', {}).get('selected_features', []))
        min_overlap = self.config['features_overlap_min_pct']
        
        intersection = len(dw_feats.intersection(dl_feats))
        union = len(dw_feats.union(dl_feats))
        overlap_pct = intersection / union if union > 0 else 0

        is_equivalent = overlap_pct >= min_overlap
        if not is_equivalent:
            self.report['critical_divergences'].append({
                'check': 'feature_selection_overlap',
                'overlap_percentage': overlap_pct,
                'threshold': min_overlap,
                'features_only_in_dw': sorted(list(dw_feats - dl_feats)),
                'features_only_in_dl': sorted(list(dl_feats - dw_feats))
            })

        print(f"   - Validação de Overlap de Features: {'OK' if is_equivalent else 'FALHOU'} ({overlap_pct:.2%})")
        return is_equivalent

    def validate_feature_correlations(self) -> bool:
        """Valida se as correlações das features com o target são equivalentes."""
        dw_corrs = self.dw_artifacts.get('feature_selection', {}).get('target_correlations', {})
        dl_corrs = self.dl_artifacts.get('feature_selection', {}).get('target_correlations', {})
        max_mae = self.config['correlations_max_mae']
        
        common_features = set(dw_corrs.keys()).intersection(set(dl_corrs.keys()))
        errors = [abs(dw_corrs[f] - dl_corrs[f]) for f in common_features]
        mae = np.mean(errors) if errors else 0

        is_equivalent = mae <= max_mae
        if not is_equivalent:
            self.report['critical_divergences'].append({
                'check': 'feature_correlations',
                'mean_absolute_error': mae,
                'threshold': max_mae
            })
        
        # Documentar diferenças de ponto flutuante esperadas
        self.report_acceptable_differences(
            'feature_correlations_floating_point',
            f"MAE de {mae:.5f} nas correlações é esperado devido a diferenças de precisão entre SQL e Dask."
        )

        print(f"   - Validação de Correlações (MAE): {'OK' if is_equivalent else 'FALHOU'} ({mae:.5f})")
        return is_equivalent

    def validate_fold_integrity(self) -> bool:
        """Valida se os tamanhos dos folds são equivalentes."""
        dw_folds = self.dw_artifacts.get('folds_config', {}).get('folds', [])
        dl_folds = self.dl_artifacts.get('folds_config', {}).get('folds', [])
        max_diff_pct = self.config['fold_sizes_max_diff_pct']
        is_equivalent = True

        if len(dw_folds) != len(dl_folds):
            self.report['critical_divergences'].append({
                'metric': 'fold_count',
                'dw_value': len(dw_folds),
                'dl_value': len(dl_folds),
                'message': f'Fold count mismatch: DW={len(dw_folds)} vs DL={len(dl_folds)}'
            })
            return False

        for i, (dw_fold, dl_fold) in enumerate(zip(dw_folds, dl_folds)):
            
            for split in ['train_count', 'val_count', 'test_count']:
                dw_count = dw_fold.get(split, 0)
                dl_count = dl_fold.get(split, 0)
                diff = abs(dw_count - dl_count) / (dw_count + 1e-9)

                if diff > max_diff_pct:
                    is_equivalent = False
                    self.report['critical_divergences'].append({
                        'check': 'fold_integrity',
                        'fold_id': i,
                        'split': split,
                        'dw_count': dw_count,
                        'dl_count': dl_count,
                        'relative_difference': diff,
                        'threshold': max_diff_pct
                    })

        print(f"   - Validação de Integridade dos Folds: {'OK' if is_equivalent else 'FALHOU'}")
        return is_equivalent

    def report_acceptable_differences(self, context: str, message: str):
        """Documenta diferenças esperadas que não invalidam o benchmark."""
        self.report['acceptable_differences'].append({
            'context': context,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })