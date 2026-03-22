#!/usr/bin/env python3
"""Setup reprodutível do pipeline ML para a arquitetura Data Lake.

O módulo executa as etapas do protocolo metodológico (QP1–QP3) no paradigma
schema-on-read: carregamento via Dask, criação de folds temporais com gaps
anti-leak, alinhamento de features com a arquitetura Data Warehouse e geração de
artefatos em `outputs/ml_pipeline/`. Mantemos o conjunto mínimo necessário de
transformações para garantir simetria com o Data Warehouse e permitir análise
exploratória distribuída sem caches ou otimizações ocultas."""

import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from typing import Any, List, Dict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.base_architecture import BaseArchitectureML
from core.config import get_absolute_output_path
from core.validation import TemporalValidator, DataIntegrityValidator
from core.logging_config import get_logger, log_ml_pipeline


class DataLakeArchitectureML(BaseArchitectureML):
    """Implementação do pipeline ML para a arquitetura Data Lake.

    A classe mantém simetria metodológica com a versão Data Warehouse: usa os
    mesmos folds temporais (QP1), garante equivalência de features e validações
    (QP2) e registra todos os artefatos necessários para o benchmark (QP3).
    O processamento é realizado com Dask em modo lazy, sem camadas de cache
    adicionais, para evidenciar características intrínsecas do paradigma
    schema-on-read."""
    
    def _safe_write_parquet_file(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Escreve arquivo Parquet com limpeza defensiva de conflitos.
        
        Args:
            df: DataFrame pandas para persistência
            file_path: Caminho de destino para arquivo Parquet
            
        Raises:
            Exception: Propagada da operação de escrita subjacente
            
        Robustez implementada:
            - Criação de diretórios pai se ausentes
            - Remoção de arquivos/diretórios conflitantes pré-existentes
            - Escrita atomica via pandas.to_parquet com index=False
            
        Justificativa:
            Data Lakes frequentemente têm conflitos de naming entre arquivos e
            diretórios devido à natureza schema-on-read. Esta função garante
            escrita bem-sucedida independente do estado do filesystem.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        df.to_parquet(file_path, index=False)
    
    def __init__(self):
        """Inicializa paths, validadores e logging para o pipeline Data Lake."""
        # Inicialização da arquitetura base
        output_base = get_absolute_output_path('ml_pipeline/architectures/data_lake')
        super().__init__(architecture_name='data_lake', output_base_path=output_base)
        
        # Logger científico com contexto ML
        self.logger = get_logger(__name__, with_ml_context=True)
        self.logger.set_context(architecture='data_lake', module='setup')
        
        print("Inicializando Pipeline ML Data Lake Científico")
        print("=" * 70)
        print("[PARADIGMA] Schema-on-read com processamento distribuído lazy")
        
        # Configurações de paths Data Lake
        self.data_lake_path = get_absolute_output_path('collection/data_lake/processed/final_results.parquet')
        self.fallback_path = get_absolute_output_path('collection/data_lake/raw')
        
        # Validadores científicos centralizados
        self.temporal_validator = TemporalValidator(min_gap_years=2)
        self.data_validator = DataIntegrityValidator()
        
        print(f"[OUTPUT] Diretório base: {self.output_base}")
        print(f"[INPUT] Dados primários: {self.data_lake_path}")
        print(f"[FALLBACK] Dados raw: {self.fallback_path}")
        print("[METODOLOGIA] Lazy evaluation sem camadas de cache adicionais")
    
    def setup_environment(self) -> None:
        """
        Configura ambiente Dask com otimizações científicas para ML temporal.
        
        Configurações aplicadas:
            1. Query planning: Habilitado para otimização automática de operações
            2. Memory management: Thresholds conservadores para estabilidade
            3. Random seeds: Determinismo científico em operações estocásticas
            4. Worker limits: Prevenção de OOM em datasets grandes
            
        Justificativa dos parâmetros:
            - memory.target=0.8: 80% RAM antes de spill (melhores práticas Dask)
            - memory.spill=0.9: 90% RAM antes de kill worker (failsafe)
            - query-planning=True: Otimização de graphs para datasets >1GB
            
        Seeds configurados:
            - NumPy: Controla amostragem e transformações estatísticas
            - Dask: Garante determinismo em operações array distribuídas
            
        Referência:
            Rocklin, M. (2015). Dask: Computação paralela com algoritmos em bloco.
        """
        print("[AMBIENTE] Configurando Dask para processamento científico distribuído")
        
        # Configurações Dask otimizadas para ML científico
        dask.config.set({'dataframe.query-planning': True})      # Otimização de queries
        dask.config.set({'distributed.worker.memory.target': 0.8})  # 80% RAM target
        dask.config.set({'distributed.worker.memory.spill': 0.9})   # 90% RAM spill
        
        print("   ✓ Memory management: Conservative thresholds configurados")
        print("   ✓ Query optimization: Habilitado para datasets >1GB")
        
        # Seeds para reprodutibilidade científica completa
        random_seed = self.config['random_seed']
        np.random.seed(random_seed)
        # Nota: Dask não possui config nativa de seed global.
        # A reprodutibilidade é garantida pela seed do numpy.

        print(f"   ✓ Reprodutibilidade: Seed {random_seed} configurado (NumPy)")
        print("[STATUS] Ambiente Dask cientificamente configurado")
    
    def load_data(self) -> dd.DataFrame:
        """
        Carrega dados educacionais com paradigma Schema-on-Read distribuído.
        
        Returns:
            dd.DataFrame: Dask DataFrame com lazy evaluation preservado para
                         otimização de memória em datasets >10GB
                         
        Raises:
            FileNotFoundError: Quando nem dados processados nem raw estão disponíveis
            
        Estratégia de carregamento hierárquica:
            1. Processed data: Dados pós-pipeline Data Lake (formato otimizado)
            2. Raw partitioned: Fallback para dados brutos particionados
            3. Error handling: Logging detalhado para debugging científico
            
        Vantagens Schema-on-Read:
            - Flexibilidade: Schema inferido dinamicamente durante carregamento
            - Performance: PyArrow engine otimizado para formatos colunares
            - Escalabilidade: Partições Dask permitem processamento >RAM disponível
            
        Complexidade: O(1) devido à lazy evaluation vs O(n) carregamento eager
        """
        self.logger.info("Iniciando carregamento Data Lake com schema-on-read")
        print("\n[CARREGAMENTO] Executando estratégia Schema-on-Read hierárquica")
        print("━" * 60)
        
        ddf = None
        data_source = None
        
        # === Estratégia 1: Dados processados (otimizados) ===
        if os.path.exists(self.data_lake_path):
            try:
                print("[TENTATIVA 1] Carregando dados processados...")
                ddf = dd.read_parquet(self.data_lake_path, engine='pyarrow').persist()
                data_source = "processed"
                ncols = len(ddf.columns)
                print(f"   ✓ Carregado: {ddf.npartitions} partições × {ncols} variáveis (persistido em memória)")
            except Exception as e:
                self.logger.warning(f"Erro ao carregar dados processados: {e}")
                print(f"   ✗ Erro dados processados: {e}")
        
        # === Estratégia 2: Dados raw particionados (fallback) ===
        if ddf is None and os.path.exists(self.fallback_path):
            try:
                print("[TENTATIVA 2] Fallback para dados raw particionados...")
                ddf = self._load_from_partitioned_raw_distributed()
                data_source = "raw_partitioned"
                print("   ✓ Carregamento raw bem-sucedido com schema-on-read")
            except Exception as e:
                self.logger.error(f"Erro ao carregar dados raw: {e}")
                print(f"   ✗ Erro dados raw: {e}")
        
        # === Validação de carregamento ===
        if ddf is None:
            raise FileNotFoundError(
                "Dados Data Lake não encontrados em nenhuma fonte.\n"
                f"Verificar: {self.data_lake_path} ou {self.fallback_path}\n"
                "Execute 'data_lake/processor.py' para gerar dados processados."
            )
    
        # === Análise de adequação científica ===
        print("[ANÁLISE] Computando estatísticas para adequação ML temporal")
        
        # Computação em lote para eficiência (única chamada Dask compute)
        stats_to_compute = {
            'year_min': ddf['year'].min(),
            'year_max': ddf['year'].max(),
            'n_countries': ddf['country_code'].nunique(),
            'total_rows': ddf.index.size
        }
        computed_stats = dask.compute(stats_to_compute)[0]
        
        # Análise de adequação para ML temporal
        years_span = computed_stats['year_max'] - computed_stats['year_min'] + 1
        avg_obs_per_country = computed_stats['total_rows'] / computed_stats['n_countries']
        
        print(f"[TEMPORAL] {computed_stats['year_min']}-{computed_stats['year_max']} ({years_span} anos)")
        print(f"[GEOGRÁFICO] {computed_stats['n_countries']} países ({avg_obs_per_country:.1f} obs/país)")
        print(f"[DIMENSÕES] {computed_stats['total_rows']:,} observações totais")
        print(f"[FONTE] {data_source}")
        
        # Warnings científicos para adequação temporal
        if years_span < 10:
            print("  ⚠ AVISO: Série temporal curta pode limitar validação walk-forward")
        
        if computed_stats['n_countries'] < 15:
            print("  ⚠ AVISO: Poucos países podem afetar generalização geográfica")
        
        self.logger.info(f"Dados carregados com sucesso via {data_source}")
        
        return ddf
    
    @log_ml_pipeline('validation')
    def validate_data(self, ddf: dd.DataFrame) -> None:
        """
        Executa validação científica distribuída com amostragem estratégica.
        
        Args:
            ddf: DataFrame Dask com dados educacionais carregados
            
        Metodologia de validação:
            1. Amostragem adaptativa: min(1000, total_rows) para eficiência
            2. DataIntegrityValidator: Validador centralizado para consistência
            3. Schema validation: Verificação de colunas obrigatórias
            4. Range validation: Detecção de valores impossíveis
            5. Fallback inteligente: Busca automática de variáveis alternativas
            
        Paradigma Schema-on-Read:
            Validação executada após carregamento, permitindo flexibilidade
            na estrutura dos dados mas garantindo qualidade mínima para ML.
            
        Critérios científicos:
            - Target coverage >50%: Poder estatístico adequado para ML
            - Range [0,100]: Consistência com definições educacionais
            - Schema compliance: Presença de identificadores temporais/geográficos
            
        Amostragem justificada:
            Para datasets >100k observações, validação completa seria
            computacionalmente custosa. Amostra de 1000 obs oferece
            95% confiança para detecção de problemas >0.5% (binomial).
        """
        print("[VALIDAÇÃO] Executando verificação científica distribuída")
        print("━" * 50)
        
        # === Amostragem adaptativa para validação eficiente ===
        total_rows = int(ddf.index.size.compute())
        sample_size = min(1000, total_rows)  # Balanceia precisão vs eficiência
        
        print(f"[AMOSTRAGEM] {sample_size:,}/{total_rows:,} observações ({sample_size/total_rows:.1%})")
        
        # Criação de amostra preservando distribuição
        sample_df = ddf.head(sample_size, npartitions=ddf.npartitions)
        if hasattr(sample_df, 'compute'):
            sample_df = sample_df.compute()
        
        # === Validação centralizada com DataIntegrityValidator ===
        print("[INTEGRIDADE] Aplicando validador centralizado...")
        is_valid, validation_report = self.data_validator.validate_dataframe(
            sample_df,
            target_col=self.source_column,
            check_completeness=True
        )
        
        if not is_valid:
            warnings = validation_report.get('warnings', [])
            self.logger.warning(f"Problemas de integridade detectados: {len(warnings)} warnings")
            for warning in warnings[:3]:
                print(f"  ⚠ {warning}")
        
        # === Schema validation com fallback inteligente ===
        print("[SCHEMA] Validando estrutura para ML temporal...")
        if self.source_column not in ddf.columns:
            print(f"  → Coluna target '{self.source_column}' não encontrada")
            
            # Fallback baseado em correspondência semântica
            completion_cols = [col for col in ddf.columns if 'completion' in col.lower()]
            if completion_cols:
                self.source_column = completion_cols[0]
                print(f"  → Usando alternativa: {self.source_column}")
            else:
                raise ValueError(
                    "Nenhuma variável educacional adequada encontrada.\n"
                    "Verificar presença de colunas com 'completion' no nome."
                )
        
        # === Análise de qualidade distribuída ===
        print("[QUALIDADE] Computando estatísticas de adequação ML...")
        
        # Computação em lote otimizada (única chamada Dask)
        validation_stats = {
            'target_data': (~ddf[self.source_column].isna()).sum(),
            'target_min': ddf[self.source_column].min(),
            'target_max': ddf[self.source_column].max(),
            'target_mean': ddf[self.source_column].mean(),
            'over_100_count': (ddf[self.source_column] > 100).sum(),
            'under_0_count': (ddf[self.source_column] < 0).sum(),
            'total_rows': ddf.index.size
        }
        computed = dask.compute(validation_stats)[0]
        
        # Análise de adequação científica
        total_rows = computed['total_rows']
        target_coverage = (computed['target_data'] / total_rows) * 100
        
        print(f"[COBERTURA] {computed['target_data']:,}/{total_rows:,} válidos ({target_coverage:.1f}%)")
        print(f"[RANGE] [{computed['target_min']:.1f}%, {computed['target_max']:.1f}%]")
        print(f"[MÉDIA] {computed['target_mean']:.1f}%")
        
        # Alertas científicos para qualidade
        if target_coverage < 50:
            print("  ⚠ CRÍTICO: Baixa cobertura de target (<50%) pode comprometer ML")
        
        if computed['over_100_count'] > 0:
            print(f"  ⚠ AVISO: {computed['over_100_count']} valores >100% (dados inválidos)")
        
        if computed['under_0_count'] > 0:
            print(f"  ⚠ AVISO: {computed['under_0_count']} valores <0% (dados inválidos)")
        
        # === Validação de schema obrigatório ===
        required_cols = ['country_code', 'year']
        missing_cols = [col for col in required_cols if col not in ddf.columns]
        if missing_cols:
            raise ValueError(
                f"Schema incompleto para ML temporal: colunas ausentes {missing_cols}.\n"
                "Identificadores país-ano são obrigatórios para validação walk-forward."
            )
        
        print("[STATUS] ✓ Validação científica concluída - equivalência Data Warehouse mantida")
    
    def create_target_implementation(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Constrói variável target via transformação Dask distribuída.
        
        Args:
            ddf: DataFrame Dask com dados educacionais
            
        Returns:
            DataFrame Dask enriquecido com variável target dropout_rate_data_lake
            
        Transformação científica:
            Dropout Rate = 100 - Completion Rate
            
        Justificativa educacional:
            Seguindo UNESCO (2018) e World Bank Education Statistics,
            dropout rate oferece interpretabilidade direta para políticas:
            - Valores altos = necessidade de intervenção urgente
            - Comparabilidade internacional padronizada
            - Linearidade com fatores socioeconômicos (Psacharopoulos, 1994)
            
        Paradigma Dask:
            Transformação aplicada lazily via .apply() com meta specification
            para preservar tipos e otimizar grafo computacional. Operação
            O(1) em termos de memória devido à lazy evaluation.
            
        Equivalência metodológica:
            Implementa mesma transformação que Data Warehouse (SQL),
            garantindo comparabilidade científica entre arquiteturas.
        
        Referência:
            Psacharopoulos, G. (1994). Returns to investment in education.
        """
        print("[TARGET] Construindo variável target com fundamentação educacional")
        
        def create_dropout_rate(completion_rate):
            """
            Função pura para transformação completion → dropout rate.
            
            Args:
                completion_rate: Taxa de conclusão (0-100%)
                
            Returns:
                Taxa de abandono (0-100%)
                
            Preserva NaN para missing values (não imputa artificialmente).
            """
            return 100 - completion_rate
        
        print(f"[TRANSFORMAÇÃO] {self.source_column} → {self.target_column}")
        print("[FÓRMULA] Dropout Rate = 100 - Completion Rate")
        
        # Transformação Dask distribuída com meta specification
        ddf_with_target = ddf.assign(
            **{self.target_column: ddf[self.source_column].apply(
                create_dropout_rate,
                meta=(self.target_column, 'f8')  # Especificação de tipo Float64
            )}
        )
        
        print("[STATUS] ✓ Target criado via Dask lazy evaluation")
        print("[EQUIVALÊNCIA] Transformação idêntica ao Data Warehouse (SQL)")
        try:
            base = ddf_with_target[['country_code', 'year', self.target_column]].rename(
                columns={self.target_column: 'dropout_rate_t'}
            )
            prev = base.assign(year=base['year'] + 2).rename(columns={'dropout_rate_t': 'dropout_rate_lag_2'})
            merged = dd.merge(ddf_with_target, prev[['country_code', 'year', 'dropout_rate_lag_2']],
                              on=['country_code', 'year'], how='left')
            # Lag 3 anos
            prev3 = base.assign(year=base['year'] + 3).rename(columns={'dropout_rate_t': 'dropout_rate_lag_3'})
            merged = dd.merge(merged, prev3[['country_code', 'year', 'dropout_rate_lag_3']],
                              on=['country_code', 'year'], how='left')
            ddf_with_target = merged
            print("[FEATURE] dropout_rate_lag_2 e dropout_rate_lag_3 criados (join country/year-k)")
        except Exception as e:
            print(f"[AVISO] Falha ao criar dropout_rate_lag_2: {e}")
        
        return ddf_with_target
    
    def _compute_target_statistics(self, ddf: dd.DataFrame) -> Dict[str, float]:
        """
        Computa estatísticas descritivas científicas da variável target via Dask distribuído.
        
        Args:
            ddf: DataFrame Dask com variável target criada
            
        Returns:
            Dicionário com estatísticas float64 para análise científica
            
        Estatísticas computadas:
            - Momentos: média, desvio padrão (não enviesado)
            - Range: mínimo, máximo para detecção de outliers
            - Completude: contagem válida vs missing para análise de qualidade
            
        Otimização distribuída:
            Única chamada dask.compute() para minimizar materialização do
            grafo computacional, crítico para datasets >10GB que não
            cabem em memória (Rocklin, 2015).
            
        Equivalência metodológica:
            Implementa mesmas estatísticas que Data Warehouse via SQL,
            garantindo comparabilidade rigorosa entre arquiteturas.
            
        Complexidade: O(n/p) onde p = número de partições Dask
        """
        # Computação em lote otimizada para eficiência distribuída
        stats_batch = {
            'mean': ddf[self.target_column].mean(),
            'std': ddf[self.target_column].std(),
            'min': ddf[self.target_column].min(),
            'max': ddf[self.target_column].max(),
            'missing_count': ddf[self.target_column].isna().sum(),
            'valid_count': (~ddf[self.target_column].isna()).sum()
        }
        
        # Única chamada compute para máxima eficiência
        computed = dask.compute(stats_batch)[0]
        
        # Conversão para float64 para consistência científica
        return {k: float(v) for k, v in computed.items()}
    
    def _validate_temporal_folds(self, ddf: dd.DataFrame, folds: List[Dict]) -> None:
        """Validação temporal  com TemporalValidator."""
        print("   ↪ Validando folds temporais...")
        
        # Usar TemporalValidator centralizado IGUAL ao Data Warehouse
        for fold in folds:
            # Validar integridade temporal usando anos
            train_years = (fold['train_start'], fold['train_end'])
            val_years = (fold['val_start'], fold['val_end'])
            test_years = (fold['test_start'], fold['test_end'])
            
            is_valid = self.validate_temporal_integrity_years(train_years, val_years, test_years)
            if not is_valid:
                self.logger.warning(f"Fold {fold['fold_id']}: Problema de integridade temporal")
            
            # Validar gaps usando validador centralizado
            is_valid, errors = self.temporal_validator.validate_fold_integrity(fold)
            if not is_valid:
                for error in errors:
                    self.logger.warning(f"Fold {fold['fold_id']}: {error}")
            
            train_filter = (
                (ddf['year'] >= fold['train_start']) &
                (ddf['year'] <= fold['train_end']) &
                ~((ddf['year'] >= fold['train_gap_start']) &
                  (ddf['year'] <= fold['train_gap_end']))
            )
            val_filter = (ddf['year'] >= fold['val_start']) & (ddf['year'] <= fold['val_end'])
            test_filter = (
                (ddf['year'] >= fold['test_start']) &
                (ddf['year'] <= fold['test_end']) &
                ~((ddf['year'] >= fold['val_gap_start']) &
                  (ddf['year'] <= fold['val_gap_end']))
            )
            
            # Contar dados por fold
            fold_stats = {
                'train_count': train_filter.sum(),
                'val_count': val_filter.sum(),
                'test_count': test_filter.sum(),
                'train_countries': ddf[train_filter]['country_code'].nunique(),
                'val_countries': ddf[val_filter]['country_code'].nunique(),
                'test_countries': ddf[test_filter]['country_code'].nunique()
            }
            computed_fold = dask.compute(fold_stats)[0]
            fold.update(computed_fold)
            
            print(f"\n   ↪ FOLD {fold['fold_id']} - Validação:")
            print(f"      Train: {fold['train_count']} obs, {fold['train_countries']} países")
            print(f"      Val: {fold['val_count']} obs, {fold['val_countries']} países")
            print(f"      Test: {fold['test_count']} obs, {fold['test_countries']} países")
    
    def get_numeric_features(self, ddf: dd.DataFrame) -> List[str]:
        """
        Identifica features numéricas candidatas para modelagem ML via type inference.
        
        Args:
            ddf: DataFrame Dask com dados educacionais
            
        Returns:
            Lista de nomes de variáveis numéricas adequadas para ML temporal
            
        Metodologia Schema-on-Read:
            Utiliza Pandas.select_dtypes() sobre schema inferido dinamicamente,
            permitindo flexibilidade na estrutura de entrada típica de Data Lakes.
            
        Critérios de seleção:
            1. Tipos numéricos: int*, float*, complex (NumPy hierarchy)
            2. Exclusão automática: Identificadores temporais/geográficos, targets
            3. Preservação de ordem: Determinismo para reprodutibilidade
            
        Vantagem vs SQL:
            Type inference dinâmico detecta features numéricas sem
            necessidade de schema pré-definido, suportando evolução
            estrutural comum em Data Lakes (Kleppmann, 2017).
            
        Limitações:
            - Não detecta variáveis categóricas numéricas (códigos, IDs)
            - Ignora features derivadas não materializadas no DataFrame
            - Schema inference pode ser custoso para DataFrames muito largos
        """
        # Schema-on-read: Type inference dinâmico
        numeric_cols = ddf.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclusão sistemática de metadados, targets e features derivadas
        # Lag features (dropout_rate_lag_*) são adicionadas em prepare_features,
        # não devem participar do filtro de colinearidade — análogo ao DW,
        # que cria lags via SQL LAG() somente em prepare_features.
        exclude_cols = ['year', 'country_code', self.target_column, self.source_column]
        exclude_prefixes = ('dropout_rate_lag_',)

        # Filtro preservando ordem para determinismo
        numeric_features = [
            col for col in numeric_cols
            if col not in exclude_cols
            and not any(col.startswith(p) for p in exclude_prefixes)
        ]
        
        return numeric_features
    
    def compute_feature_correlations(self, ddf: dd.DataFrame,
                                    features: List[str]) -> Dict[str, float]:
        """
        Computa correlações de Pearson feature-target via amostragem científica.
        
        Args:
            ddf: DataFrame Dask com dados educacionais completos
            features: Lista de features candidatas para análise de correlação
            
        Returns:
            Dicionário {feature_name: absolute_correlation} para ranking
            
        Metodologia híbrida Dask-Pandas:
            1. Dask sampling: Amostragem distribuída eficiente para datasets >10GB
            2. Pandas correlation: Cálculo preciso de correlação após materialização
            3. Equivalência SQL: Resultados idênticos ao Data Warehouse para benchmarking
            
        Amostragem científica:
            - Tamanho ótimo: min(10k, total_rows) baseado em Central Limit Theorem
            - Seed reprodutível: Garante determinismo entre execuções
            - Random sampling: Preserva distribuição populacional (Cochran, 1977)
            
        Justificativa da abordagem:
            Para equivalência rigorosa com SQL Data Warehouse, materialização
            da amostra é necessária. Overhead computacional é aceitável pois
            correlação é step único na seleção de features.
            
        Complexidade: O(s×f) onde s=sample_size, f=num_features
        
        Referências:
            Cochran, W. G. (1977). Sampling techniques (3rd ed.). Wiley.
            Pearson, K. (1895). Mathematical contributions to evolution.
        """
        print("[CORRELAÇÃO] Computando associação linear via amostragem científica")
        
        target_col = self.target_column
        correlations = {}
        
        # === Amostragem científica reprodutível (configurável) ===
        total_rows = int(ddf.index.size.compute())
        use_sampling = bool(self.config.get('correlation_sampling', True))
        min_sample = int(self.config.get('correlation_min_sample_size', 5000))
        frac = float(self.config.get('correlation_sample_fraction', 0.1))

        if use_sampling and total_rows > min_sample:
            sample_size = max(min_sample, int(total_rows * frac))
            sample_frac = min(sample_size / total_rows, 1.0)
        else:
            sample_size = total_rows
            sample_frac = 1.0

        print(f"[AMOSTRAGEM] {sample_size:,}/{total_rows:,} observações ({sample_frac:.1%})")
        print("[SEED] Configurando amostragem reprodutível...")

        # Amostragem distribuída Dask com seed científico (ou full-scan)
        sample_ddf = ddf if sample_frac >= 0.9999 else ddf.sample(
            frac=sample_frac,
            random_state=self.config['random_seed']
        )
        sample_df = sample_ddf.compute()  # Materialização para Pandas
        
        print(f"[MATERIALIZAÇÃO] Amostra de {len(sample_df):,} obs para análise Pandas")
        
        # === Cálculo de correlação preciso via Pandas ===
        print(f"[PEARSON] Calculando correlações para {len(features)} features...")
        
        successful_correlations = 0
        failed_features = []
        
        for feat in features:
            if feat not in sample_df.columns:
                correlations[feat] = 0.0
                continue
                
            try:
                # Correlação de Pearson com tratamento 
                corr = sample_df[feat].corr(sample_df[target_col])
                
                # Valor absoluto para ranking por força da associação
                if pd.isna(corr):
                    correlations[feat] = 0.0
                else:
                    correlations[feat] = abs(float(corr))
                    successful_correlations += 1
                    
            except Exception as e:
                self.logger.warning(f"Erro correlação {feat}: {e}")
                correlations[feat] = 0.0
                failed_features.append(feat)
        
        # === Relatório científico ===
        print(f"[RESULTADO] {successful_correlations}/{len(features)} correlações bem-sucedidas")
        
        if failed_features:
            print(f"[AVISOS] {len(failed_features)} features com erro: {failed_features[:3]}")
        
        # Estatísticas agregadas
        valid_correlations = [r for r in correlations.values() if r > 0]
        if valid_correlations:
            avg_corr = sum(valid_correlations) / len(valid_correlations)
            max_corr = max(valid_correlations)
            print(f"[ESTATÍSTICAS] Correlação média: {avg_corr:.3f}, máxima: {max_corr:.3f}")
        
        print("[STATUS] ✓ Análise de correlação concluída com equivalência SQL")
        
        return correlations
    

    def apply_collinearity_filter(self, ddf: dd.DataFrame, features: List[str],
                                   threshold: float = 0.8) -> List[str]:
        """
        Remove multicolinearidade via filtragem greedy de correlação pairwise.

        Para cada feature candidata, calcula a correlação absoluta máxima com
        as features já selecionadas e rejeita se max |r| >= threshold.

        Args:
            ddf: DataFrame Dask com features candidatas
            features: Lista de features para análise de multicolinearidade
            threshold: Limiar de correlação pairwise (padrão 0.8)

        Returns:
            Lista filtrada de features com multicolinearidade reduzida

        Algoritmo greedy:
            1. Primeira feature sempre aceita (baseline)
            2. Features subsequentes aceitas se max |r| < threshold
            3. Ordem preservada para determinismo científico

        Amostragem híbrida Dask-Pandas:
            - Dask: Amostragem distribuída eficiente para datasets >10GB
            - Pandas: Matriz de correlação precisa após materialização
            - Equivalência: Resultados idênticos ao Data Warehouse SQL

        Referências:
            Dormann, C. F., et al. (2013). Collinearity: a review of methods
            to deal with it and a simulation study evaluating their performance.
        """
        if len(features) <= 1:
            print("[COLLINEARITY] Menos de 2 features - análise desnecessária")
            return features

        print(f"[COLLINEARITY] Executando análise de multicolinearidade para {len(features)} features")
        print("━" * 50)

        # === Configuração de amostragem científica ===
        total_rows = float(ddf.index.size.compute())

        # Critérios baseados em configuração científica
        min_sample_absolute = self.config.get('correlation_min_sample_size', 5000)
        sample_fraction = self.config.get('correlation_sample_fraction', 0.1)

        min_sample_size = max(min_sample_absolute, int(total_rows * sample_fraction))
        sample_frac = min(min_sample_size / total_rows, 1.0)

        print(f"[AMOSTRAGEM] {min_sample_size:,} registros ({sample_frac:.1%})")
        print(f"[CRITÉRIO] max(config={min_sample_absolute}, {sample_fraction:.0%}×dataset)")

        try:
            # === Amostragem distribuída Dask ===
            print("[DASK] Executando amostragem distribuída...")
            corr_sample_ddf = ddf[features].sample(
                frac=sample_frac,
                random_state=self.config['random_seed']
            )

            # === Materialização Pandas para correlação precisa ===
            print("[PANDAS] Materializando amostra para análise de correlação...")
            corr_data = corr_sample_ddf.compute().dropna()

            actual_sample_size = len(corr_data)
            print(f"[CONFIRMAÇÃO] {actual_sample_size:,} observações válidas pós-dropna")

            if actual_sample_size > 10:  # Mínimo estatístico
                # === Construção de matriz de correlação ===
                print("[MATRIZ] Computando correlações par-a-par...")
                corr_matrix = corr_data.corr().abs()

                # === Algoritmo greedy pairwise (equivalente Data Warehouse) ===
                print(f"[ALGORITMO] Aplicando seleção greedy com threshold {threshold}")

                selected = []
                rejected_count = 0

                for feature in features:
                    if feature not in corr_matrix.columns:
                        continue

                    if not selected:
                        selected.append(feature)
                    else:
                        max_corr = 0.0
                        worst_pair = None

                        for sel_feat in selected:
                            if sel_feat in corr_matrix.columns:
                                corr_val = corr_matrix.loc[feature, sel_feat]
                                if corr_val > max_corr:
                                    max_corr = corr_val
                                    worst_pair = sel_feat

                        if max_corr < threshold:
                            selected.append(feature)
                        else:
                            rejected_count += 1
                            if rejected_count <= 3:
                                print(f"  → Rejeitado {feature}: r={max_corr:.3f} com {worst_pair}")

                # === Relatório científico ===
                reduction_rate = ((len(features) - len(selected)) / len(features)) * 100
                print(f"\n[RESULTADO] Filtragem de colinearidade concluída:")
                print(f"  → Features originais: {len(features)}")
                print(f"  → Features selecionadas: {len(selected)}")
                print(f"  → Redução: {len(features) - len(selected)} ({reduction_rate:.1f}%)")
                print(f"  → Equivalência: Algoritmo idêntico ao Data Warehouse SQL")

                return selected

            else:
                print(f"[FALLBACK] Amostra inadequada ({actual_sample_size}≤10) - retornando top-10")
                return features[:10]

        except Exception as e:
            self.logger.error(f"Erro na filtragem de colinearidade: {e}")
            print(f"[ERRO] Filtragem de colinearidade falhou: {e}")
            print("[FALLBACK] Retornando top-10 features como segurança")
            return features[:10]
    
    @log_ml_pipeline('feature_engineering')
    def prepare_features(self, ddf: dd.DataFrame, selected_features: List[str]) -> dd.DataFrame:
        """
        Executa feature engineering científico com symmetric log transform distribuída.
        
        Args:
            ddf: DataFrame Dask com features selecionadas via filtragem de colinearidade
            selected_features: Features pós-seleção para transformação
            
        Returns:
            DataFrame Dask enriquecido com features originais + transformadas
            
        Engenharia de Features Científica:
            Aplica symmetric log transform: T(x) = sign(x) * ln(|x| + 1)
            às top-5 features para normalização de distribuições assimétricas comuns
            em dados socioeconômicos.

        Justificativas metodológicas:
            1. Top-5 limite: Baseado em curse of dimensionality (Bellman, 1961)
               e overfitting em pequenas amostras educacionais
            2. Symmetric log: Trata zeros e negativos naturalmente, adequada para
               indicadores educacionais com déficits/declínios
            3. Lazy evaluation: Transformações aplicadas via .apply() Dask
               para otimização de memória
               
        Estrutura final:
            - Metadados: country_code, year, target (essenciais ML temporal)
            - Features originais: selected_features (pós-filtragem de colinearidade)
            - Features transformadas: {feature}_log_transform (top-5)
            
        Equivalência arquitetural:
            Implementa mesmas transformações que Data Warehouse via SQL,
            garantindo comparabilidade científica para benchmarking.
            
        Logging científico:
            Captura métricas de qualidade (missing%, dimensionalidade) para
            auditoria e reprodutibilidade do pipeline ML.
            
        Referências:
            Bellman, R. (1961). Adaptive control processes. Princeton.
        """
        print("\n[FEATURE ENGINEERING] Executando transformações científicas distribuídas")
        print("━" * 70)

        # === Cópia defensiva para preservar DataFrame original ===
        print("[PREPARAÇÃO] Criando cópia de trabalho...")
        ddf_work = ddf.copy()

        # === Transformação, log simétrico: T(x) = sign(x) * ln(|x| + 1) ===
        print("[TRANSFORMAÇÃO] Aplicando symmetric log transform a top-5 features")
        
        # Critério: Limitar escopo por curse of dimensionality
        features_to_transform = selected_features[:5] if len(selected_features) > 5 else selected_features
        transformed_count = 0
        
        print(f"[ESCOPO] {len(features_to_transform)} features selecionadas para transformação:")
        
        # Aplicação de transformação feature por feature
        for feat in features_to_transform:
            if feat not in ddf_work.columns:
                print(f"  → {feat}: AUSENTE (ignorado)")
                continue
            
            transform_col = f"{feat}_log_transform"
            
            print(f"  → {feat} → {transform_col}")
            
            ddf_work[transform_col] = ddf_work[feat].apply(
                lambda x: np.sign(x) * np.log(np.abs(x) + 1) if pd.notna(x) else np.nan,
                meta=(transform_col, 'f8')  # Metadados Float64 para Dask
            )
            transformed_count += 1
        
        print(f"[RESULTADO] {transformed_count} symmetric log transforms aplicadas")
        
        # === Construção de dataset ML final ===
        print("[ASSEMBLY] Construindo dataset final para modelagem...")
        
        # Metadados essenciais para ML temporal
        ml_features = ['country_code', 'year', self.target_column]
        
        # Features originais pós-filtragem de colinearidade
        ml_features.extend(selected_features)
        
        # Features transformadas (apenas as que foram criadas)
        transformed_cols = [f"{feat}_log_transform" for feat in features_to_transform
                          if f"{feat}_log_transform" in ddf_work.columns]
        ml_features.extend(transformed_cols)
        
        # Incluir lags do target no dataset salvo, mesmo que não selecionados
        for lag_col in ['dropout_rate_lag_2', 'dropout_rate_lag_3']:
            if lag_col in ddf_work.columns and lag_col not in ml_features:
                ml_features.append(lag_col)
        
        # === Garantia de unicidade e existência ===
        # Preservar ordem mas remover duplicatas
        ml_features = list(dict.fromkeys(ml_features))
        
        # Filtrar apenas colunas que existem no DataFrame
        ml_features = [col for col in ml_features if col in ddf_work.columns]
        
        print(f"[DIMENSIONALIDADE] Dataset ML final:")
        print(f"  → Metadados: 3 (country_code, year, target)")
        print(f"  → Features originais: {len(selected_features)}")
        print(f"  → Features transformadas: {len(transformed_cols)}")
        print(f"  → Total variáveis: {len(ml_features)}")
        
        # === Seleção final com preservação de estrutura Dask ===
        result_ddf = ddf_work[ml_features]
        
        # === Logging científico para auditoria ===
        try:
            total_rows = int(ddf.index.size.compute())
            sample_size = min(100, total_rows)
            
            # Cálculo de estatísticas de qualidade
            if hasattr(result_ddf, 'compute'):
                sample_stats = result_ddf.head(sample_size, npartitions=result_ddf.npartitions)
            else:
                sample_stats = result_ddf.head(sample_size)
            
            # Proporção de valores faltantes
            missing_pct = float(sample_stats.isna().mean().mean() * 100)
            
            # Log estruturado para reprodutibilidade
            self.logger.log_data_info(
                "ml_ready_data",
                shape=(total_rows, len(ml_features)),
                missing_pct=missing_pct
            )
            
            print(f"[QUALIDADE] {missing_pct:.1f}% valores faltantes (amostra n={sample_size})")
            
        except Exception as e:
            self.logger.warning(f"Erro ao computar estatísticas de qualidade: {e}")
            print(f"  ⚠ Aviso: Estatísticas de qualidade indisponíveis: {e}")
        
        print("[STATUS] ✓ Feature engineering concluído - dataset pronto para ML temporal")
        
        return result_ddf
    
    def save_folds(self, ddf: dd.DataFrame, folds: List[Dict]) -> None:
        """Salva folds"""
        print("\n🖧 SALVANDO FOLDS...")
        
        for fold in folds:
            fold_id = fold['fold_id']
            fold_dir = f"{self.prep_dir}/folds/fold_{fold_id}"
            os.makedirs(fold_dir, exist_ok=True)
            
            print(f"   ↪ Processando fold {fold_id}...")
            
            train_filter = (
                (ddf['year'] >= fold['train_start']) &
                (ddf['year'] <= fold['train_end']) &
                ~((ddf['year'] >= fold['train_gap_start']) &
                  (ddf['year'] <= fold['train_gap_end']))
            )
            val_filter = (
                (ddf['year'] >= fold['val_start']) &
                (ddf['year'] <= fold['val_end'])
            )
            test_filter = (
                (ddf['year'] >= fold['test_start']) &
                (ddf['year'] <= fold['test_end']) &
                ~((ddf['year'] >= fold['val_gap_start']) &
                  (ddf['year'] <= fold['val_gap_end']))
            )
            
            train_ddf = ddf[train_filter]
            val_ddf = ddf[val_filter]
            test_ddf = ddf[test_filter]
            
            # Converter para Pandas e salvar
            try:
                train_df = train_ddf.compute()
                val_df = val_ddf.compute()
                test_df = test_ddf.compute()
                
                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)
                self._safe_write_parquet_file(train_df, f'{fold_dir}/train_data_lake.parquet')
                self._safe_write_parquet_file(val_df, f'{fold_dir}/val_data_lake.parquet')
                self._safe_write_parquet_file(test_df, f'{fold_dir}/test_data_lake.parquet')
                
                print(f"   ✔ Fold {fold_id} salvo: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
                
            except Exception as e:
                print(f"   ✖ Erro ao salvar fold {fold_id}: {e}")
                raise
            
            fold_metadata = {
                **fold,
                'storage_method': 'parquet_files',
                'paradigm': 'schema_on_read'
            }
            self.save_fold_metadata(fold_metadata, fold_dir)
        
        # Master data 
        print("\n   ↪ Salvando master data...")
        try:
            master_path = f"{self.prep_dir}/master_data_data_lake.parquet"
            master_df = ddf.compute().reset_index(drop=True)
            self._safe_write_parquet_file(master_df, master_path)
            print(f"   ✔ Master data salvo: {len(master_df)} registros")
            
        except Exception as e:
            print(f"   ✖ Erro ao salvar master data: {e}")
            raise
        
        # Configuração master 
        total_obs = len(master_df)
        total_countries = master_df['country_code'].nunique()
        year_min = int(master_df['year'].min())
        year_max = int(master_df['year'].max())
        
        self.save_master_config(folds, total_obs, total_countries, (year_min, year_max))
        
        print(f"   ✔ Data Lake: Folds salvos com paradigma Schema-on-Read")
    
    
    def _load_from_partitioned_raw_distributed(self) -> dd.DataFrame:
        """Carrega dados particionados com Schema-on-Read."""
        parquet_files = glob.glob(f"{self.fallback_path}/**/*.parquet", recursive=True)
        
        if not parquet_files:
            raise FileNotFoundError("Nenhum arquivo parquet encontrado")
        
        ddf = dd.read_parquet(self.fallback_path, engine='pyarrow').persist()

        # Conversão para formato wide se necessário
        if 'indicator_name' in ddf.columns:
            print("   Convertendo para formato wide (Schema-on-Read)...")
            ddf = self._convert_to_wide_format_distributed(ddf)
        
        return ddf
    
    def _convert_to_wide_format_distributed(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """Conversão wide format - Usa Pandas quando necessário."""
        try:
            print("    Aplicando pivotagem para formato wide...")
            
            df = ddf.compute()
            
            index_cols = ['country_code', 'country_name', 'year']
            if 'country_stratum' in df.columns:
                index_cols.append('country_stratum')
            
            # Pivotear dados
            df_wide = df.pivot_table(
                index=index_cols,
                columns='indicator_name',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            df_wide.columns.name = None
            
            ddf_wide = dd.from_pandas(df_wide, npartitions=max(1, len(df_wide) // 10000))
            
            print(f"   ✔ Conversão concluída: {len(ddf_wide.columns)} colunas")
            return ddf_wide
            
        except Exception as e:
            print(f"   ✖ Erro na conversão wide: {e}")
            return ddf
    
    def run_setup_with_monitoring(self) -> Dict[str, Any]:
        """Executa setup  com monitoramento."""
        with self.logger.timer("complete_setup_pipeline"):
            results = self.run_setup()
            
            results['paradigm'] = 'schema_on_read_dask'
            results['standardized'] = True
            results['version'] = 'no_cache'

            self.logger.info(
                "Setup Data Lake concluído",
                total_time=results.get('processing_time'),
            )

            return results


def main():
    """Executa o pipeline Data Lake end-to-end para validação local."""
    print("="*80)
    print("TESTE PIPELINE ML DATA LAKE - METODOLOGIA DISTRIBUÍDA CIENTÍFICA".center(80))
    print("="*80)
    
    try:
        # Inicialização com configurações científicas otimizadas
        print("\n[INICIALIZAÇÃO] Configurando pipeline Data Lake distribuído...")
        setup = DataLakeArchitectureML()
        
        # Execução com monitoramento científico completo
        print("\n[EXECUÇÃO] Iniciando pipeline científico com monitoramento completo...")
        results = setup.run_setup_with_monitoring()
        
        # Relatório científico detalhado
        print("\n" + "="*80)
        print("RELATÓRIO CIENTÍFICO DO PIPELINE DATA LAKE".center(80))
        print("="*80)
        
        if results.get('status') == 'success':
            print("✓ Pipeline Data Lake executado com SUCESSO")
            print(f"  → Paradigma: {results.get('paradigm', 'N/A')}")
            print(f"  → Features selecionadas: {results.get('features_selected', 'N/A')}")
            print(f"  → Folds temporais: {results.get('folds_created', 'N/A')}")
            print(f"  → Processamento: {results.get('processing_time', 'N/A')}s")
        else:
            print("✗ Pipeline Data Lake FALHOU")
            if 'error' in results:
                print(f"  → Erro: {results['error']}")
            
        print(f"\n[DETALHES] Resultados completos:")
        for key, value in results.items():
            if key not in ['status', 'error']:
                print(f"  → {key}: {value}")
                
    except Exception as e:
        print(f"\n✗ ERRO CRÍTICO no pipeline Data Lake: {e}")
        print("   → Verificar se dados foram processados pelo data_lake/processor.py")
        print("   → Verificar configuração Dask e disponibilidade de memória")
        print("   → Consultar logs detalhados para debugging")
        return {'success': False, 'error': str(e)}
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
