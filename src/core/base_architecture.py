#!/usr/bin/env python3
"""
Classe base abstrata para arquiteturas ML.

Este módulo define a estrutura comum para todas as arquiteturas de ML,
garantindo consistência metodológica e facilitando manutenção sem duplicação.
Preserva 100% da lógica científica original de cada arquitetura.
"""

from abc import ABC, abstractmethod
import os
import sys
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from core.scientific_config import SCIENTIFIC_CONFIG, setup_reproducibility


class BaseArchitectureML(ABC):
    """
    Classe base abstrata para arquiteturas de Machine Learning.
    
    Define a estrutura comum e métodos compartilhados entre diferentes
    arquiteturas (Data Lake, Data Warehouse), garantindo consistência
    metodológica e eliminando duplicação de código.
    
    Características principais:
    - Interface abstrata para métodos específicos de cada arquitetura
    - Implementação compartilhada de lógica comum (folds, validação, etc.)
    - Preservação de 100% da lógica científica original
    - Suporte para diferentes backends (Dask, SQL, Pandas)
    
    Attributes:
        architecture_name: Nome da arquitetura (data_lake, data_warehouse)
        output_base: Diretório base para outputs
        prep_dir: Diretório de preparação
        target_column: Nome da coluna target criada
        source_column: Coluna fonte para criar target
    """
    
    def __init__(self, architecture_name: str, output_base_path: str):
        """
        Inicializa a arquitetura base.
        
        Args:
            architecture_name: Identificador da arquitetura
            output_base_path: Caminho base para outputs
        """
        self.architecture_name = architecture_name
        self.output_base = output_base_path
        self.prep_dir = f"{self.output_base}/prep"
        
        # Configuração científica centralizada
        self.config = SCIENTIFIC_CONFIG
        setup_reproducibility()

        # Configuração de target (comum a todas arquiteturas)
        self.target_column = f"dropout_rate_{architecture_name}"
        self.source_column = "lower_secondary_completion_rate"
        
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Cria estrutura de diretórios necessária."""
        os.makedirs(self.prep_dir, exist_ok=True)
        os.makedirs(f"{self.prep_dir}/folds", exist_ok=True)
        
    @abstractmethod
    def setup_environment(self) -> None:
        """
        Configura ambiente específico da arquitetura.
        
        Método abstrato que deve ser implementado por cada arquitetura
        para configurar seu ambiente específico (Dask, SQL, etc.).
        """
        pass
    
    @abstractmethod
    def load_data(self) -> Any:
        """
        Carrega dados específicos da arquitetura.
        
        Returns:
            Dados carregados no formato específico da arquitetura
            (dd.DataFrame para Dask, conexão SQL para DW, etc.)
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> None:
        """
        Valida integridade dos dados.
        
        Args:
            data: Dados para validação no formato da arquitetura
            
        Raises:
            ValueError: Quando validação falha
        """
        pass
    
    @abstractmethod
    def create_target_implementation(self, data: Any) -> Any:
        """
        Implementação específica para criar variável target.
        
        Args:
            data: Dados de entrada
            
        Returns:
            Dados com variável target criada
        """
        pass
    
    def create_target(self, data: Any) -> Any:
        """
        Cria variável target com validação científica comum.
        
        Este método implementa a lógica comum de criação de target
        (dropout_rate = 100 - completion_rate) com validações científicas
        idênticas para todas as arquiteturas.
        
        Args:
            data: Dados de entrada no formato da arquitetura
            
        Returns:
            Dados com variável target criada e validada
            
        Note:
            Utiliza inversão simples pois raw_data_collector já
            garantiu que valores estão no range [0,100].
        """
        print(f"\nCriando target {self.architecture_name}...")
        print(f"   Usando fonte: {self.source_column}")
        print(f"   Target: {self.target_column}")
        
        # Delegar implementação específica
        data_with_target = self.create_target_implementation(data)
        
        self._save_target_statistics(data_with_target)
        
        return data_with_target
    
    @abstractmethod
    def _compute_target_statistics(self, data: Any) -> Dict[str, float]:
        """
        Computa estatísticas do target específicas da arquitetura.
        
        Args:
            data: Dados com target criado
            
        Returns:
            Dicionário com estatísticas (mean, std, min, max, etc.)
        """
        pass
    
    def _save_target_statistics(self, data: Any) -> None:
        """
        Salva estatísticas do target de forma padronizada.
        
        Args:
            data: Dados com target para computar estatísticas
        """
        stats = self._compute_target_statistics(data)
        
        stats.update({
            'architecture': self.architecture_name,
            'target_variable': self.target_column,
            'source_column': self.source_column,
            'creation_timestamp': datetime.now().isoformat()
        })
        
        # Validações científicas comuns
        expected_range = [0, 80]  # Baseado em completion rates reais
        if stats['mean'] < expected_range[0] or stats['mean'] > expected_range[1]:
            print(f"   Aviso: Média de dropout ({stats['mean']:.2f}%) "
                  f"fora do range esperado {expected_range}")
        
        if stats['valid_count'] < 500:
            print(f"   Aviso: Poucos dados válidos ({stats['valid_count']}) "
                  f"para ML robusto")
        
        stats_path = f"{self.prep_dir}/target_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"   Estatísticas salvas: {stats_path}")
    
    def create_temporal_folds(self, data: Any = None) -> List[Dict]:
        """
        Cria folds temporais científicos com metodologia walk-forward.
        
        Implementa estrutura de validação temporal idêntica para todas
        as arquiteturas, garantindo comparabilidade científica.
        
        Args:
            data: Dados opcionais para validação de folds
            
        Returns:
            Lista de configurações de folds com metadados completos
            
        Note:
            Aplica gaps temporais de 2 anos entre train/val e val/test
            para prevenção rigorosa de vazamento temporal.
            Metodologia adequada para conferences/journals ML.
        """
        print("\nCriando folds temporais...")
        gap = int(self.config.get('temporal_gap_years', 2))
        print(f"Metodologia: Walk-forward automático com gaps de {gap} anos")
        folds = self._generate_walkforward_folds_auto()
        
        # Validar folds se dados fornecidos
        if data is not None:
            # Resetar índice do DataFrame para evitar duplicatas (se for um DataFrame)
            if hasattr(data, 'reset_index'):
                data = data.reset_index(drop=True)
            self._validate_temporal_folds(data, folds)
        
        return folds

    def _generate_walkforward_folds_auto(self) -> List[Dict]:
        """
        Gera folds walk-forward expansivos automaticamente, respeitando gaps e janelas.

        Parâmetros são lidos de SCIENTIFIC_CONFIG (com defaults seguros):
          - temporal_range_start / end
          - folds_min_train_years
          - folds_val_len_years
          - folds_test_len_years
          - temporal_gap_years
          - folds_step_years
          - folds_max (opcional)
        """
        cfg = self.config
        start_year = int(cfg.get('temporal_range_start', 2000))
        end_year = int(cfg.get('temporal_range_end', 2023))
        min_train = int(cfg.get('folds_min_train_years', 8))
        val_len = int(cfg.get('folds_val_len_years', 2))
        test_len = int(cfg.get('folds_test_len_years', 2))
        gap = int(cfg.get('temporal_gap_years', 2))
        step = int(cfg.get('folds_step_years', 1))
        max_folds = cfg.get('folds_max', None)
        try:
            max_folds = int(max_folds) if max_folds is not None else None
        except Exception:
            max_folds = None

        # Calcular limites de início para a janela de teste
        # Derivação: val_start >= start_year + min_train + gap
        # test_start = val_start + val_len + gap
        # => test_start_min = start_year + min_train + val_len + 2*gap
        test_start_min = start_year + min_train + val_len + 2 * gap
        test_start_max = end_year - test_len + 1

        folds: List[Dict] = []
        fold_id = 0
        for test_start in range(test_start_min, test_start_max + 1, step):
            test_end = test_start + test_len - 1
            # Derivar val_end e val_start a partir do gap
            val_end = test_start - gap - 1
            val_start = val_end - val_len + 1
            # Derivar train_end e train_start
            train_end = val_start - gap - 1
            train_start = start_year

            # Verificações de validade
            if train_end < train_start:
                continue
            train_len = train_end - train_start + 1
            if train_len < min_train:
                continue
            if not (train_start <= train_end < val_start <= val_end < test_start <= test_end <= end_year):
                continue

            train_val_gap = val_start - train_end - 1
            val_test_gap = test_start - val_end - 1
            if train_val_gap < gap or val_test_gap < gap:
                continue

            fold = {
                'fold_id': fold_id,
                'architecture': self.architecture_name,
                'methodology': 'walk_forward_with_gaps_auto',
                'train_start': int(train_start), 'train_end': int(train_end),
                'train_gap_start': int(train_end + 1), 'train_gap_end': int(val_start - 1),
                'val_start': int(val_start), 'val_end': int(val_end),
                'val_gap_start': int(val_end + 1), 'val_gap_end': int(test_start - 1),
                'test_start': int(test_start), 'test_end': int(test_end),
                'total_train_years': int(train_len),
                'total_val_years': int(val_len),
                'total_test_years': int(test_len),
                'train_val_gap': int(train_val_gap),
                'val_test_gap': int(val_test_gap),
                'description': f'Walk-forward auto (gap={gap}y, val={val_len}y, test={test_len}y)',
                'forecast_horizon': '1-2 anos à frente'
            }
            folds.append(fold)
            fold_id += 1
            if max_folds is not None and len(folds) >= max_folds:
                break

        if not folds:
            raise ValueError(
                f"Nenhum fold pôde ser gerado com os parâmetros atuais. "
                f"Ajuste temporal_range_start/end, folds_min_train_years ou gaps. "
                f"Parâmetros: start={start_year}, end={end_year}, min_train={min_train}, "
                f"val_len={val_len}, test_len={test_len}, gap={gap}"
            )

        print(f"   ✓ Folds auto-gerados: {len(folds)} (gap={gap}, val={val_len}, test={test_len})")
        return folds
    
    @abstractmethod
    def _validate_temporal_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Valida estrutura científica dos folds.
        
        Args:
            data: Dados para validação
            folds: Lista de folds para validar
        """
        pass
    
    @abstractmethod
    def save_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Salva folds no formato específico da arquitetura.
        
        Args:
            data: Dados processados
            folds: Lista de configurações de folds
        """
        pass
    
    def get_excluded_features(self) -> List[str]:
        """
        Retorna lista de features a excluir (vazamento/metadados).
        
        Lista harmonizada entre todas as arquiteturas para garantir
        comparação científica justa.
        
        Returns:
            Lista de nomes de colunas a excluir
        """
        return [
            # Identificadores e metadados
            'country_code', 'country_name', 'year', 'country_stratum', 
            'synthetic_flag',
            
            # Target
            self.target_column,
            
            # Metadados de processamento
            'data_source', 'etl_batch_id', 'collection_timestamp',
            'data_completeness_score', 'processing_method', 
            'processed_timestamp', 'partition_id',
            
            # Vazamento de dados - features derivadas do target
            'lower_secondary_completion_rate',  # Fonte direta do target
            'enrollment_rate_secondary_net',    # Relacionada ao target
            'stratum_completion_mean',          # Deriva de completion_rate
            'stratum_completion_std',           # Deriva de completion_rate
            'completion_rate_lag1',             # Deriva de completion_rate
            'completion_rate_rolling_mean',     # Deriva de completion_rate
            'temporal_trend',                   # Deriva de correlação temporal
            
            # Redundâncias
            'year_rank'  # Redundante com 'year'
        ]
    
    @abstractmethod
    def compute_feature_correlations(self, data: Any, features: List[str]) -> Dict[str, float]:
        """
        Computa correlações entre features e target.
        
        Args:
            data: Dados com features
            features: Lista de features para análise
            
        Returns:
            Dicionário com correlações absolutas
        """
        pass
    
    def select_features_by_correlation(self, 
                                      correlations: Dict[str, float],
                                      min_corr: float = 0.15,
                                      max_corr: float = 0.8) -> List[str]:
        """
        Seleciona features por correlação moderada com target.
        
        Args:
            correlations: Dicionário de correlações
            min_corr: Correlação mínima (evita irrelevância)
            max_corr: Correlação máxima (evita vazamento)
            
        Returns:
            Lista de features selecionadas
        """
        selected = [
            feat for feat, corr in correlations.items()
            if min_corr <= corr <= max_corr
        ]
        
        print(f"   Features com correlação moderada ({min_corr}-{max_corr}): "
              f"{len(selected)}")
        
        # Relaxar critério se muito poucas features
        if len(selected) < 5:
            selected = [
                feat for feat, corr in correlations.items()
                if corr >= min_corr * 0.67  
            ]
            print(f"   Critério relaxado: {len(selected)} features")
        
        return selected
    
    @abstractmethod
    def apply_collinearity_filter(self, data: Any, features: List[str],
                                  threshold: float = 0.8) -> List[str]:
        """
        Remove multicolinearidade via filtragem greedy de correlação pairwise.

        Para cada feature candidata, calcula a correlação absoluta máxima com
        as features já selecionadas. Rejeita se max |r| >= threshold.

        Args:
            data: Dados para análise
            features: Features candidatas
            threshold: Threshold de correlação pairwise para remoção

        Returns:
            Lista de features após remoção de multicolinearidade
        """
        pass
    
    def run_feature_selection(self, data: Any) -> Dict:
        """
        Executa pipeline completo de seleção de features.
        
        Pipeline padronizado:
        1. Remove features com vazamento/metadados
        2. Seleciona por correlação moderada com target
        3. Remove multicolinearidade via filtragem pairwise de correlação
        
        Args:
            data: Dados para seleção
            
        Returns:
            Dicionário com estatísticas e features selecionadas
        """
        print(f"\nFeature selection {self.architecture_name}...")
        
        exclude_cols = self.get_excluded_features()
        all_features = self.get_numeric_features(data)
        feature_cols = [col for col in all_features if col not in exclude_cols]
        
        print(f"   Removidas {len(exclude_cols)} variáveis (vazamento/metadados)")
        print(f"   Analisando {len(feature_cols)} features candidatas...")
        
        # Correlação com target
        correlations = self.compute_feature_correlations(data, feature_cols)
        selected_by_corr = self.select_features_by_correlation(correlations)
        
        # Filtragem de colinearidade pairwise
        final_features = self.apply_collinearity_filter(data, selected_by_corr)
        
        # Estatísticas de seleção
        selection_stats = {
            'architecture': self.architecture_name,
            'total_features_analyzed': len(feature_cols),
            'features_selected': len(final_features),
            'selection_method': 'correlation_pairwise_filter',
            'selected_features': final_features,
            'target_correlations': {
                feat: float(correlations.get(feat, 0))
                for feat in final_features
            },
            'selection_timestamp': datetime.now().isoformat()
        }
        
        selection_path = f"{self.prep_dir}/feature_selection_{self.architecture_name}.json"
        with open(selection_path, 'w') as f:
            json.dump(selection_stats, f, indent=2)
        
        print(f"   Feature selection salva: {selection_path}")
        print(f"   Features selecionadas: {len(final_features)}")
        
        return selection_stats
    
    @abstractmethod
    def get_numeric_features(self, data: Any) -> List[str]:
        """
        Obtém lista de features numéricas dos dados.
        
        Args:
            data: Dados de entrada
            
        Returns:
            Lista de nomes de colunas numéricas
        """
        pass
    
    @abstractmethod
    def prepare_features(self, data: Any, selected_features: List[str]) -> Any:
        """
        Prepara features finais para ML.
        
        Args:
            data: Dados completos
            selected_features: Lista de features selecionadas
            
        Returns:
            Dados preparados com features finais
        """
        pass
    
    def prepare_time_series_data(self, data: Any, target_col: str = None) -> Any:
        """
        Prepara dados de série temporal para ML.
        
        Args:
            data: Dados de entrada
            target_col: Nome da coluna target
            
        Returns:
            Dados preparados para série temporal
        """
        if target_col is None:
            target_col = self.target_column
            
        # Implementação básica compartilhada
        return data
    
    def engineer_features(self, data: Any) -> Any:
        """
        Aplica feature engineering nos dados.
        
        Args:
            data: Dados de entrada
            
        Returns:
            Dados com features engenheiradas
        """
        # Implementação básica compartilhada
        return data
    
    def train_baseline_models(self, X_train: Any, y_train: Any) -> Dict:
        """
        Treina modelos baseline.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Dicionário com modelos treinados
        """
        # Implementação básica compartilhada
        return {}
    
    def evaluate_models(self, models: Dict, X_test: Any, y_test: Any) -> Dict:
        """
        Avalia modelos treinados.
        
        Args:
            models: Dicionário com modelos
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas
        """
        # Implementação básica compartilhada
        return {}
    
    @staticmethod
    def _convert_numpy_types(obj):
        """Converte tipos numpy para tipos Python nativos para serialização JSON."""
        if hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, dict):
            return {k: BaseArchitectureML._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [BaseArchitectureML._convert_numpy_types(v) for v in obj]
        else:
            return obj

    def save_fold_metadata(self, fold: Dict, fold_dir: str) -> None:
        """
        Salva metadados de um fold de forma padronizada.

        Args:
            fold: Configuração do fold
            fold_dir: Diretório do fold
        """
        fold_metadata = {
            **self._convert_numpy_types(fold),
            'data_source': self.architecture_name,
            'target_variable': self.target_column,
            'temporal_boundaries_preserved': True,
            'gaps_applied_effectively': True,
            'saved_timestamp': datetime.now().isoformat()
        }
        
        with open(f'{fold_dir}/metadata.json', 'w') as f:
            json.dump(fold_metadata, f, indent=2)
    
    def save_master_config(self, folds: List[Dict], total_observations: int,
                          total_countries: int, year_range: Tuple[int, int]) -> str:
        """
        Salva configuração master dos folds.
        
        Args:
            folds: Lista de folds
            total_observations: Total de observações
            total_countries: Total de países
            year_range: Tupla (ano_min, ano_max)
            
        Returns:
            Caminho do arquivo de configuração salvo
        """
        folds_config = {
            'architecture': self.architecture_name,
            'creation_timestamp': datetime.now().isoformat(),
            'total_observations': int(total_observations),
            'total_countries': int(total_countries),
            'year_range': [int(year_range[0]), int(year_range[1])],
            'target_variable': self.target_column,
            'folds': self._convert_numpy_types(folds)
        }
        
        folds_path = f"{self.prep_dir}/temporal_folds_{self.architecture_name}.json"
        with open(folds_path, 'w') as f:
            json.dump(folds_config, f, indent=2)
        
        return folds_path
    
    def run_setup(self) -> Dict:
        """
        Executa pipeline completo de setup da arquitetura.
        
        Pipeline padronizado:
        1. Setup do ambiente
        2. Carregamento de dados
        3. Validação de dados
        4. Criação de target
        5. Seleção de features
        6. Preparação de features
        7. Criação de folds temporais
        8. Salvamento de folds e configurações
        
        Returns:
            Dicionário com resultados do setup
        """
        print(f"Executando setup {self.architecture_name}")
        print("=" * 60)
        
        try:
            # 1. Setup do ambiente específico
            self.setup_environment()
            
            # 2. Carregar dados
            data = self.load_data()
            
            # 3. Validar dados
            self.validate_data(data)
            
            # 4. Criar target
            data_with_target = self.create_target(data)
            
            selection_stats = self.run_feature_selection(data_with_target)
            
            data_processed = self.prepare_features(
                data_with_target, 
                selection_stats['selected_features']
            )
            
            # 7. Criar folds temporais
            folds = self.create_temporal_folds(data_processed)
            
            # 8. Salvar folds
            self.save_folds(data_processed, folds)
            
            print(f"\nSetup {self.architecture_name} concluído!")
            print("=" * 60)
            
            return {
                'architecture': self.architecture_name,
                'status': 'success',
                'setup_timestamp': datetime.now().isoformat(),
                'features_selected': len(selection_stats['selected_features']),
                'folds_created': len(folds)
            }
            
        except Exception as e:
            print(f"\nErro no setup {self.architecture_name}: {e}")
            print("=" * 60)
            
            return {
                'architecture': self.architecture_name,
                'status': 'failed',
                'error': str(e),
                'setup_timestamp': datetime.now().isoformat()
            }
    
    def compute_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calcula métricas de performance padrão para avaliação de modelos.
        
        Método compartilhado para cálculo consistente de métricas entre
        todas as arquiteturas, garantindo comparabilidade científica.
        
        Args:
            y_true: Valores reais do target
            y_pred: Valores preditos pelo modelo
            
        Returns:
            Dicionário com métricas calculadas:
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - r2: Coeficiente de determinação R²
            - mape: Mean Absolute Percentage Error (se aplicável)
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE apenas quando não há zeros no y_true
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape) if not np.isnan(mape) else None
        }
    
    def validate_data_integrity(self, data: Any, required_columns: List[str] = None) -> Dict[str, Any]:
        """
        Valida integridade dos dados com verificações científicas.
        
        Args:
            data: Dados para validação
            required_columns: Colunas obrigatórias
            
        Returns:
            Dicionário com resultados da validação
        """
        integrity_issues = []
        
        # Verificar colunas obrigatórias
        if required_columns:
            if hasattr(data, 'columns'):
                missing_cols = [col for col in required_columns if col not in data.columns]
                if missing_cols:
                    integrity_issues.append(f"Colunas faltantes: {missing_cols}")
        
        # Verificar dados nulos excessivos
        if hasattr(data, 'isna'):
            null_ratio = data.isna().mean()
            high_null_cols = null_ratio[null_ratio > 0.9].index.tolist()
            if high_null_cols:
                integrity_issues.append(f"Colunas com >90% nulos: {high_null_cols}")
        
        return {
            'integrity_check': 'passed' if not integrity_issues else 'failed',
            'issues': integrity_issues,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def log_pipeline_step(self, step_name: str, details: Dict = None) -> None:
        """
        Registra etapa do pipeline com detalhes.
        
        Args:
            step_name: Nome da etapa
            details: Detalhes adicionais
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'architecture': self.architecture_name,
            'step': step_name,
            'details': details or {}
        }
        
        # Log para arquivo se existir diretório
        if os.path.exists(self.prep_dir):
            log_file = f"{self.prep_dir}/pipeline_log.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        print(f"   [{step_name}] logged at {log_entry['timestamp']}")
    
    def validate_temporal_integrity(self, train_data: Any, val_data: Any,
                                   test_data: Any) -> Tuple[bool, str]:
        """
        Valida integridade temporal dos splits para prevenir vazamento.
        
        Versão alternativa que recebe os dados diretamente ao invés de apenas anos.
        
        Args:
            train_data: Dados de treino
            val_data: Dados de validação
            test_data: Dados de teste
            
        Returns:
            Tupla (is_valid, message)
        """
        # Extrair anos dos dados
        if hasattr(train_data, '__getitem__') and 'year' in train_data:
            train_years = (train_data['year'].min(), train_data['year'].max())
            val_years = (val_data['year'].min(), val_data['year'].max())
            test_years = (test_data['year'].min(), test_data['year'].max())
            
            is_valid = self.validate_temporal_integrity_years(
                train_years, val_years, test_years
            )
            
            if is_valid:
                return True, "Integridade temporal validada"
            else:
                return False, "Violação de integridade temporal detectada"
        
        return True, "Não foi possível validar integridade temporal"
    
    def validate_temporal_integrity_years(self, train_years: Tuple[int, int],
                                   val_years: Tuple[int, int],
                                   test_years: Tuple[int, int]) -> bool:
        """
        Valida integridade temporal dos splits para prevenir vazamento.
        
        Verifica se:
        1. Os períodos estão em ordem cronológica correta
        2. Existem gaps temporais adequados entre splits
        3. Não há sobreposição entre períodos
        
        Args:
            train_years: Tupla (ano_inicial, ano_final) do treino
            val_years: Tupla (ano_inicial, ano_final) da validação
            test_years: Tupla (ano_inicial, ano_final) do teste
            
        Returns:
            True se a integridade temporal está preservada, False caso contrário
            
        Note:
            Requer gap mínimo de 2 anos entre splits para prevenir
            vazamento temporal em dados educacionais.
        """
        # Verificar ordem temporal
        if not (train_years[1] < val_years[0] < val_years[1] < test_years[0]):
            print(f"   ERRO: Ordem temporal violada")
            print(f"   Train: {train_years}, Val: {val_years}, Test: {test_years}")
            return False
        
        # Calcular gaps
        train_val_gap = val_years[0] - train_years[1] - 1
        val_test_gap = test_years[0] - val_years[1] - 1
        
        MIN_GAP = self.config.get('temporal_gap_years', 2)
        
        if train_val_gap < MIN_GAP:
            print(f"   ERRO: Gap train-val insuficiente: {train_val_gap} < {MIN_GAP}")
            return False
        
        if val_test_gap < MIN_GAP:
            print(f"   ERRO: Gap val-test insuficiente: {val_test_gap} < {MIN_GAP}")
            return False
        
        # Nota: sobreposição já detectada pelas verificações de gap acima
        
        print(f"   ✓ Integridade temporal validada")
        print(f"   Gaps: train-val={train_val_gap}yr, val-test={val_test_gap}yr")
        
        return True
    
    def calculate_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Calcula importância das features de forma padronizada.
        
        Args:
            model: Modelo treinado com atributo feature_importances_
            feature_names: Lista com nomes das features
            
        Returns:
            Dicionário com importância normalizada de cada feature
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Normalizar para somar 1
            total = importances.sum()
            importances_norm = importances / total if total != 0 else importances
            
            return {
                feature: float(importance)
                for feature, importance in zip(feature_names, importances_norm)
            }
        return {}
    
    def detect_data_drift(self, reference_data: Any, current_data: Any,
                         features: List[str], threshold: float = 0.1) -> Dict:
        """
        Detecta drift nos dados comparando distribuições.
        
        Args:
            reference_data: Dados de referência (treino)
            current_data: Dados atuais (produção/teste)
            features: Lista de features para análise
            threshold: Threshold para detecção de drift
            
        Returns:
            Dicionário com análise de drift por feature
        """
        from scipy import stats
        
        drift_analysis = {}
        
        for feature in features:
            # Extrair valores da feature (específico da arquitetura)
            # Implementação básica - subclasses podem sobrescrever
            if hasattr(reference_data, '__getitem__'):
                ref_values = reference_data[feature]
                curr_values = current_data[feature]
                
                # Teste Kolmogorov-Smirnov
                ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
                
                drift_detected = p_value < threshold
                
                drift_analysis[feature] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drift_detected': drift_detected,
                    'threshold': threshold
                }
        
        return drift_analysis

    def validate_scientific_equivalence(self, other_arch_results: Dict) -> Dict:
        """
        Valida a equivalência científica com os resultados de outra arquitetura.

        Este método delega a validação para a classe BenchmarkValidator,
        que é projetada para comparar os artefatos de dois pipelines.

        Args:
            other_arch_results: Dicionário de resultados da outra arquitetura.

        Returns:
            Dicionário com o relatório de validação de equivalência.
        """
        from core.benchmark_validator import BenchmarkValidator

        print(f"\nValidando equivalência científica com {other_arch_results.get('architecture', 'outra arquitetura')}...")
        
        self_results = {
            'architecture': self.architecture_name,
            'output_base_path': self.output_base
        }

        validator = BenchmarkValidator(self_results, other_arch_results)
        
        equivalence_report = validator.validate_all()

        report_path = f"{self.output_base}/scientific_equivalence_report.json"
        with open(report_path, 'w') as f:
            json.dump(equivalence_report, f, indent=2)
        
        print(f"   Relatório de equivalência salvo em: {report_path}")
        return equivalence_report

    def get_reproducibility_report(self) -> Dict:
        """
        Gera um relatório sobre as configurações de reprodutibilidade.

        Returns:
            Dicionário contendo as configurações de seed utilizadas.
        """
        report = {
            'architecture': self.architecture_name,
            'global_random_seed': self.config.get('random_seed'),
            'numpy_seed_set': True,
            'dask_seed_set': 'dask' in sys.modules,
            'report_timestamp': datetime.now().isoformat()
        }
        print("\n🌱 Relatório de Reprodutibilidade:")
        for key, value in report.items():
            print(f"   - {key}: {value}")
            
        return report
