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

    PARADIGM_META = {
        'name': 'data_lake',
        'label': 'Data Lake com Dask',
        'processor_module': 'collection.data_lake.processor',
        'processor_class': 'DataLakeProcessor',
        'processor_run_method': 'run_data_lake_processing',
        'baseline_module': 'architectures_ml.data_lake.models.baseline_analysis',
        'baseline_class': 'BaselineModelAnalysisDataLake',
        'hierarchical_module': 'architectures_ml.data_lake.models.hierarchical_model',
        'hierarchical_class': 'HierarchicalModelDataLake',
        'setup_script': 'src/architectures_ml/data_lake/setup.py',
        'processor_script': 'src/collection/data_lake/processor.py',
        'baseline_script': 'src/architectures_ml/data_lake/models/baseline_analysis.py',
        'hierarchical_script': 'src/architectures_ml/data_lake/models/hierarchical_model.py',
    }

    def _safe_write_parquet_file(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Escreve arquivo Parquet com limpeza defensiva de conflitos.
        
        Args:
            df: DataFrame pandas para persistência
            file_path: Caminho de destino para arquivo Parquet
            
        Raises:
            Exception: Propagada da operação de escrita subjacente
            
        Tratamento de conflitos:
            - Criação de diretórios pai se ausentes
            - Remoção de arquivos/diretórios conflitantes pré-existentes
            - Escrita atomica via pandas.to_parquet com index=False

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
        
        self.logger = get_logger(__name__, with_ml_context=True)
        self.logger.set_context(architecture='data_lake', module='setup')
        
        print("Inicializando Pipeline ML Data Lake")
        print("Schema-on-read com processamento distribuido lazy")
        
        # Configurações de paths Data Lake
        self.data_lake_path = get_absolute_output_path('collection/data_lake/processed/final_results.parquet')
        self.fallback_path = get_absolute_output_path('collection/data_lake/raw')
        
        self.temporal_validator = TemporalValidator(min_gap_years=2)
        self.data_validator = DataIntegrityValidator()
        
        print(f"  Diretorio base: {self.output_base}")
        print(f"  Dados primarios: {self.data_lake_path}")
        print(f"  Dados raw (fallback): {self.fallback_path}")
        print("  Lazy evaluation sem camadas de cache adicionais")
    
    def setup_environment(self) -> None:
        """
        Configura ambiente Dask com otimizações para ML temporal.
        
        Configurações aplicadas:
            1. Query planning: Habilitado para otimização automática de operações
            2. Memory management: Thresholds conservadores para estabilidade
            3. Random seeds: Determinismo em operações estocásticas
            4. Worker limits: Prevenção de OOM em datasets grandes
            
        Justificativa dos parâmetros:
            - memory.target=0.8: 80% RAM antes de spill (melhores práticas Dask)
            - memory.spill=0.9: 90% RAM antes de kill worker (failsafe)
            - query-planning=True: Otimização de graphs para datasets >1GB
            
        Seeds configurados:
            - NumPy: Controla amostragem e transformações estatísticas
            - Dask: Garante determinismo em operações array distribuídas
            
        """
        print("Configurando Dask")
        
        dask.config.set({'dataframe.query-planning': True})      # Otimização de queries
        dask.config.set({'distributed.worker.memory.target': 0.8})  # 80% RAM target
        dask.config.set({'distributed.worker.memory.spill': 0.9})   # 90% RAM spill
        
        print("  Memory management: conservative thresholds")
        print("  Query optimization: habilitado para datasets >1GB")

        # Dask não possui config nativa de seed global.
        # A reprodutibilidade é garantida pela seed do numpy.
        random_seed = self.config['random_seed']
        np.random.seed(random_seed)

        print(f"  Seed {random_seed} configurado (NumPy)")
    
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
            3. Error handling: Logging detalhado para debugging
            
        Vantagens Schema-on-Read:
            - Flexibilidade: Schema inferido dinamicamente durante carregamento
            - Performance: PyArrow engine otimizado para formatos colunares
            - Escalabilidade: Partições Dask permitem processamento >RAM disponível
            
        Leitura lazy -- materialização sob demanda.
        """
        self.logger.info("Iniciando carregamento Data Lake com schema-on-read")
        print("\nCarregando dados (schema-on-read)")
        
        ddf = None
        data_source = None
        
        # Estratégia 1: Dados processados (otimizados)
        if os.path.exists(self.data_lake_path):
            try:
                ddf = dd.read_parquet(self.data_lake_path, engine='pyarrow').persist()
                data_source = "processed"
                ncols = len(ddf.columns)
                print(f"  Carregado: {ddf.npartitions} particoes x {ncols} variaveis")
            except Exception as e:
                self.logger.warning(f"Erro ao carregar dados processados: {e}")
                print(f"  [ERROR] Dados processados: {e}")
        
        # Estratégia 2: Dados raw particionados (fallback)
        if ddf is None and os.path.exists(self.fallback_path):
            try:
                print("  Fallback para dados raw particionados...")
                ddf = self._load_from_partitioned_raw_distributed()
                data_source = "raw_partitioned"
                print("  Carregamento raw ok")
            except Exception as e:
                self.logger.error(f"Erro ao carregar dados raw: {e}")
                print(f"  [ERROR] Dados raw: {e}")
        
        # Validação de carregamento
        if ddf is None:
            raise FileNotFoundError(
                "Dados Data Lake não encontrados em nenhuma fonte.\n"
                f"Verificar: {self.data_lake_path} ou {self.fallback_path}\n"
                "Execute 'data_lake/processor.py' para gerar dados processados."
            )
    
        # Análise de adequação
        
        # Computação em lote para eficiência (única chamada Dask compute)
        stats_to_compute = {
            'year_min': ddf['year'].min(),
            'year_max': ddf['year'].max(),
            'n_countries': ddf['country_code'].nunique(),
            'total_rows': ddf.index.size
        }
        computed_stats = dask.compute(stats_to_compute)[0]
        
        years_span = computed_stats['year_max'] - computed_stats['year_min'] + 1
        avg_obs_per_country = computed_stats['total_rows'] / computed_stats['n_countries']

        print(f"  {computed_stats['year_min']}-{computed_stats['year_max']} ({years_span} anos)")
        print(f"  {computed_stats['n_countries']} paises ({avg_obs_per_country:.1f} obs/pais)")
        print(f"  {computed_stats['total_rows']:,} observacoes totais")
        print(f"  Fonte: {data_source}")

        if years_span < 10:
            print("  [WARN] Serie temporal curta pode limitar validacao walk-forward")

        if computed_stats['n_countries'] < 15:
            print("  [WARN] Poucos paises podem afetar generalizacao geografica")
        
        self.logger.info(f"Dados carregados com sucesso via {data_source}")
        
        return ddf
    
    @log_ml_pipeline('validation')
    def validate_data(self, ddf: dd.DataFrame) -> None:
        """
        Executa validação distribuída com amostragem estratégica.
        
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
            
        Critérios:
            - Target coverage >50%: Poder estatístico adequado para ML
            - Range [0,100]: Consistência com definições educacionais
            - Schema compliance: Presença de identificadores temporais/geográficos
            
        Amostragem:
            Para eficiência, amostra de min(1000, total_rows) observações.
        """
        print("Validando dados")
        
        # Amostragem adaptativa para validação eficiente
        total_rows = int(ddf.index.size.compute())
        sample_size = min(1000, total_rows)  # Balanceia precisão vs eficiência
        
        print(f"  Amostragem: {sample_size:,}/{total_rows:,} ({sample_size/total_rows:.1%})")

        # Criação de amostra preservando distribuição
        sample_df = ddf.head(sample_size, npartitions=ddf.npartitions)
        if hasattr(sample_df, 'compute'):
            sample_df = sample_df.compute()
        
        # Validação centralizada com DataIntegrityValidator
        is_valid, validation_report = self.data_validator.validate_dataframe(
            sample_df,
            target_col=self.source_column,
            check_completeness=True
        )
        
        if not is_valid:
            warnings = validation_report.get('warnings', [])
            self.logger.warning(f"Problemas de integridade detectados: {len(warnings)} warnings")
            for warning in warnings[:3]:
                print(f"  [WARN] {warning}")

        # Schema validation com fallback
        if self.source_column not in ddf.columns:
            print(f"  [WARN] Coluna target '{self.source_column}' nao encontrada")

            completion_cols = [col for col in ddf.columns if 'completion' in col.lower()]
            if completion_cols:
                self.source_column = completion_cols[0]
                print(f"    Usando alternativa: {self.source_column}")
            else:
                raise ValueError(
                    "Nenhuma variável educacional adequada encontrada.\n"
                    "Verificar presença de colunas com 'completion' no nome."
                )
        
        # Análise de qualidade distribuída
        
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
        
        total_rows = computed['total_rows']
        target_coverage = (computed['target_data'] / total_rows) * 100

        print(f"  Cobertura: {computed['target_data']:,}/{total_rows:,} validos ({target_coverage:.1f}%)")
        print(f"  Range: [{computed['target_min']:.1f}%, {computed['target_max']:.1f}%]")
        print(f"  Media: {computed['target_mean']:.1f}%")

        if target_coverage < 50:
            print("  [WARN] Baixa cobertura de target (<50%) pode comprometer ML")

        if computed['over_100_count'] > 0:
            print(f"  [WARN] {computed['over_100_count']} valores >100% (dados invalidos)")

        if computed['under_0_count'] > 0:
            print(f"  [WARN] {computed['under_0_count']} valores <0% (dados invalidos)")
        
        # Validação de schema obrigatório
        required_cols = ['country_code', 'year']
        missing_cols = [col for col in required_cols if col not in ddf.columns]
        if missing_cols:
            raise ValueError(
                f"Schema incompleto para ML temporal: colunas ausentes {missing_cols}.\n"
                "Identificadores país-ano são obrigatórios para validação walk-forward."
            )
        
        print("  Validacao concluida")
    
    def create_target_implementation(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Constrói variável target via transformação Dask distribuída.
        
        Args:
            ddf: DataFrame Dask com dados educacionais
            
        Returns:
            DataFrame Dask enriquecido com variável target dropout_rate_data_lake
            
        Transformação:
            Dropout Rate = 100 - Completion Rate
            
        Justificativa educacional:
            Seguindo UNESCO (2018) e World Bank Education Statistics,
            dropout rate oferece interpretabilidade direta para políticas:
            - Valores altos = necessidade de intervenção urgente
            - Comparabilidade internacional padronizada
            - Comparabilidade internacional padronizada

        Paradigma Dask:
            Transformação aplicada lazily via .apply() com meta specification
            para preservar tipos e otimizar grafo computacional.
        """
        print("Construindo variavel target")
        
        def create_dropout_rate(completion_rate):
            """
            Funcao pura para transformacao completion -> dropout rate.
            
            Args:
                completion_rate: Taxa de conclusão (0-100%)
                
            Returns:
                Taxa de abandono (0-100%)
                
            Preserva NaN para missing values (não imputa artificialmente).
            """
            return 100 - completion_rate
        
        print(f"  {self.source_column} -> {self.target_column}")
        print("  Dropout Rate = 100 - Completion Rate")
        
        # Transformação Dask distribuída com meta specification
        ddf_with_target = ddf.assign(
            **{self.target_column: ddf[self.source_column].apply(
                create_dropout_rate,
                meta=(self.target_column, 'f8')  # Especificação de tipo Float64
            )}
        )
        
        print("  Target criado via Dask lazy evaluation")
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
            print("  dropout_rate_lag_2 e dropout_rate_lag_3 criados (join country/year-k)")
        except Exception as e:
            print(f"  [WARN] Falha ao criar dropout_rate_lag_2: {e}")
        
        return ddf_with_target
    
    def _compute_target_statistics(self, ddf: dd.DataFrame) -> Dict[str, float]:
        """
        Computa estatísticas descritivas da variável target via Dask distribuído.
        
        Args:
            ddf: DataFrame Dask com variável target criada
            
        Returns:
            Dicionário com estatísticas float64 para análise
            
        Estatísticas computadas:
            - Momentos: média, desvio padrão (não enviesado)
            - Range: mínimo, máximo para detecção de outliers
            - Completude: contagem válida vs missing para análise de qualidade
            
        Otimização distribuída:
            Única chamada dask.compute() para minimizar materialização do
            grafo computacional.
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
        
        # Conversão para float64 para consistência
        return {k: float(v) for k, v in computed.items()}
    
    def _validate_temporal_folds(self, ddf: dd.DataFrame, folds: List[Dict]) -> None:
        """Validação temporal  com TemporalValidator."""
        print("Validando folds temporais")

        # Validação via TemporalValidator centralizado
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
            
            print(f"\n  Fold {fold['fold_id']}:")
            print(f"    Train: {fold['train_count']} obs, {fold['train_countries']} paises")
            print(f"    Val: {fold['val_count']} obs, {fold['val_countries']} paises")
            print(f"    Test: {fold['test_count']} obs, {fold['test_countries']} paises")
    
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
        # que cria lags via self-join temporal somente em prepare_features.
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
        Computa correlações de Pearson feature-target via amostragem.
        
        Args:
            ddf: DataFrame Dask com dados educacionais completos
            features: Lista de features candidatas para análise de correlação
            
        Returns:
            Dicionário {feature_name: absolute_correlation} para ranking
            
        Metodologia híbrida Dask-Pandas:
            1. Dask sampling: Amostragem distribuída eficiente para datasets >10GB
            2. Pandas correlation: Cálculo preciso de correlação após materialização
            3. Equivalência SQL: Resultados idênticos ao Data Warehouse para benchmarking
            
        Amostragem:
            - Tamanho ótimo: min(10k, total_rows) baseado em Central Limit Theorem
            - Seed reprodutível: Garante determinismo entre execuções
            - Random sampling: Preserva distribuição populacional (Cochran, 1977)
            
        Justificativa da abordagem:
            Para equivalência com SQL Data Warehouse, materialização
            da amostra é necessária. Overhead computacional é aceitável pois
            correlação é step único na seleção de features.
            
        """
        print("Analisando correlacoes feature-target")
        
        target_col = self.target_column
        correlations = {}
        
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

        print(f"  Amostragem: {sample_size:,}/{total_rows:,} ({sample_frac:.1%})")

        sample_ddf = ddf if sample_frac >= 0.9999 else ddf.sample(
            frac=sample_frac,
            random_state=self.config['random_seed']
        )
        sample_df = sample_ddf.compute()  # Materialização para Pandas
        
        print(f"  Amostra materializada: {len(sample_df):,} obs, {len(features)} features")
        
        successful_correlations = 0
        failed_features = []
        
        for feat in features:
            if feat not in sample_df.columns:
                correlations[feat] = 0.0
                continue
                
            try:
                corr = sample_df[feat].corr(sample_df[target_col])
                
                if pd.isna(corr):
                    correlations[feat] = 0.0
                else:
                    correlations[feat] = abs(float(corr))
                    successful_correlations += 1
                    
            except Exception as e:
                self.logger.warning(f"Erro correlação {feat}: {e}")
                correlations[feat] = 0.0
                failed_features.append(feat)
        
        print(f"  {successful_correlations}/{len(features)} correlacoes calculadas")

        if failed_features:
            print(f"  [WARN] {len(failed_features)} features com erro: {failed_features[:3]}")

        valid_correlations = [r for r in correlations.values() if r > 0]
        if valid_correlations:
            avg_corr = sum(valid_correlations) / len(valid_correlations)
            max_corr = max(valid_correlations)
            print(f"  Correlacao media: {avg_corr:.3f}, maxima: {max_corr:.3f}")
        
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
            3. Ordem preservada para determinismo

        Amostragem híbrida Dask-Pandas:
            - Dask: Amostragem distribuída eficiente para datasets >10GB
            - Pandas: Matriz de correlação precisa após materialização
            - Equivalência: Resultados idênticos ao Data Warehouse SQL

        """
        if len(features) <= 1:
            print("  Menos de 2 features - colinearidade desnecessaria")
            return features

        print(f"Filtrando colinearidade: {len(features)} features")

        # Configuração de amostragem
        total_rows = float(ddf.index.size.compute())

        min_sample_absolute = self.config.get('correlation_min_sample_size', 5000)
        sample_fraction = self.config.get('correlation_sample_fraction', 0.1)

        min_sample_size = max(min_sample_absolute, int(total_rows * sample_fraction))
        sample_frac = min(min_sample_size / total_rows, 1.0)

        print(f"  Amostragem: {min_sample_size:,} registros ({sample_frac:.1%})")

        try:
            # Amostragem distribuída Dask
            corr_sample_ddf = ddf[features].sample(
                frac=sample_frac,
                random_state=self.config['random_seed']
            )

            corr_data = corr_sample_ddf.compute().dropna()

            actual_sample_size = len(corr_data)
            print(f"  {actual_sample_size:,} observacoes validas pos-dropna")

            if actual_sample_size > 10:  # Mínimo estatístico
                corr_matrix = corr_data.corr().abs()

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
                                print(f"    Rejeitado {feature}: r={max_corr:.3f} com {worst_pair}")

                reduction_rate = ((len(features) - len(selected)) / len(features)) * 100
                print(f"  Originais: {len(features)}, selecionadas: {len(selected)}, "
                      f"removidas: {len(features) - len(selected)} ({reduction_rate:.1f}%)")

                return selected

            else:
                print(f"  Amostra inadequada ({actual_sample_size}<=10) - fallback top-10")
                return features[:10]

        except Exception as e:
            self.logger.error(f"Erro na filtragem de colinearidade: {e}")
            print(f"[ERROR] Filtragem de colinearidade falhou: {e}")
            print("  Fallback: retornando top-10 features")
            return features[:10]
    
    @log_ml_pipeline('feature_engineering')
    def prepare_features(self, ddf: dd.DataFrame, selected_features: List[str]) -> dd.DataFrame:
        """
        Executa feature engineering com symmetric log transform distribuída.
        
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
            garantindo comparabilidade para benchmarking.
            
        Logging:
            Captura métricas de qualidade (missing%, dimensionalidade) para
            auditoria e reprodutibilidade do pipeline ML.
            
        """
        print("\nFeature engineering")

        # Cópia para preservar DataFrame original
        ddf_work = ddf.copy()

        # Transformação log simétrico: T(x) = sign(x) * ln(|x| + 1)
        
        # Critério: Limitar escopo por curse of dimensionality
        features_to_transform = selected_features[:5] if len(selected_features) > 5 else selected_features
        transformed_count = 0
        
        print(f"  Transformando {len(features_to_transform)} features (symmetric log):")
        
        # Aplicação de transformação feature por feature
        for feat in features_to_transform:
            if feat not in ddf_work.columns:
                print(f"    {feat}: AUSENTE (ignorado)")
                continue

            transform_col = f"{feat}_log_transform"

            print(f"    {feat} -> {transform_col}")
            
            ddf_work[transform_col] = ddf_work[feat].apply(
                lambda x: np.sign(x) * np.log(np.abs(x) + 1) if pd.notna(x) else np.nan,
                meta=(transform_col, 'f8')  # Metadados Float64 para Dask
            )
            transformed_count += 1
        
        print(f"  {transformed_count} log transforms aplicadas")

        # Construção de dataset ML final
        
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
        
        # Remover duplicatas preservando ordem
        ml_features = list(dict.fromkeys(ml_features))
        
        # Filtrar apenas colunas que existem no DataFrame
        ml_features = [col for col in ml_features if col in ddf_work.columns]
        
        print(f"  Dataset ML final: {len(ml_features)} variaveis "
              f"({len(selected_features)} originais, {len(transformed_cols)} transformadas)")
        
        # Seleção final
        result_ddf = ddf_work[ml_features]
        
        # Logging para auditoria
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
            
            print(f"  {missing_pct:.1f}% valores faltantes (amostra n={sample_size})")
            
        except Exception as e:
            self.logger.warning(f"Erro ao computar estatísticas de qualidade: {e}")
            print(f"  [WARN] Estatisticas de qualidade indisponiveis: {e}")
        
        print("  Feature engineering concluido")
        
        return result_ddf
    
    def save_folds(self, ddf: dd.DataFrame, folds: List[Dict]) -> None:
        """Salva folds"""
        print("\nSalvando folds")
        
        for fold in folds:
            fold_id = fold['fold_id']
            fold_dir = f"{self.prep_dir}/folds/fold_{fold_id}"
            os.makedirs(fold_dir, exist_ok=True)
            
            print(f"  Processando fold {fold_id}...")
            
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
                
                print(f"    Fold {fold_id}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
                
            except Exception as e:
                print(f"    [ERROR] Salvamento fold {fold_id}: {e}")
                raise
            
            fold_metadata = {
                **fold,
                'storage_method': 'parquet_files',
                'paradigm': 'schema_on_read'
            }
            self.save_fold_metadata(fold_metadata, fold_dir)
        
        # Master data
        print("\n  Salvando master data...")
        try:
            master_path = f"{self.prep_dir}/master_data_data_lake.parquet"
            master_df = ddf.compute().reset_index(drop=True)
            self._safe_write_parquet_file(master_df, master_path)
            print(f"    Master data: {len(master_df)} registros")
            
        except Exception as e:
            print(f"    [ERROR] Salvamento master data: {e}")
            raise
        
        # Configuração master 
        total_obs = len(master_df)
        total_countries = master_df['country_code'].nunique()
        year_min = int(master_df['year'].min())
        year_max = int(master_df['year'].max())
        
        self.save_master_config(folds, total_obs, total_countries, (year_min, year_max))
        
        print(f"  Data Lake: folds salvos")
    
    
    def _load_from_partitioned_raw_distributed(self) -> dd.DataFrame:
        """Carrega dados particionados com Schema-on-Read."""
        parquet_files = glob.glob(f"{self.fallback_path}/**/*.parquet", recursive=True)
        
        if not parquet_files:
            raise FileNotFoundError("Nenhum arquivo parquet encontrado")
        
        ddf = dd.read_parquet(self.fallback_path, engine='pyarrow').persist()

        # Conversão para formato wide se necessário
        if 'indicator_name' in ddf.columns:
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
            
            print(f"    Conversao concluida: {len(ddf_wide.columns)} colunas")
            return ddf_wide
            
        except Exception as e:
            print(f"    [ERROR] Conversao wide: {e}")
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
    print("=" * 80)
    print("Pipeline ML Data Lake")
    print("=" * 80)

    try:
        setup = DataLakeArchitectureML()
        results = setup.run_setup_with_monitoring()
        
        if results.get('status') == 'success':
            print("Pipeline ok")
            print(f"  Paradigma: {results.get('paradigm', 'N/A')}")
            print(f"  Features selecionadas: {results.get('features_selected', 'N/A')}")
            print(f"  Folds temporais: {results.get('folds_created', 'N/A')}")
            print(f"  Processamento: {results.get('processing_time', 'N/A')}s")
        else:
            print("[ERROR] Pipeline Data Lake falhou")
            if 'error' in results:
                print(f"  Erro: {results['error']}")

        print(f"\nResultados:")
        for key, value in results.items():
            if key not in ['status', 'error']:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"\n[ERROR] Pipeline Data Lake falhou: {e}")
        print("  Verificar se dados foram processados pelo data_lake/processor.py")
        return {'success': False, 'error': str(e)}
    


if __name__ == "__main__":
    main()
