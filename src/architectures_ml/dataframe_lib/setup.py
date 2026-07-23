#!/usr/bin/env python3
"""Setup reprodutível do pipeline ML para a arquitetura Polars DataFrame.

O módulo executa as etapas do protocolo metodológico no paradigma Polars nativo:
carregamento via pl.scan_parquet() (LazyFrame), criação de folds temporais com gaps
anti-leak, alinhamento de features com as arquiteturas Data Lake e Data Warehouse,
e geração de artefatos em `outputs/ml_pipeline/architectures/dataframe_lib/`.

Mantém simetria metodológica com DL e DW para comparação controlada,
diferindo apenas na implementação específica de Polars usando expressions e
lazy evaluation para otimização de memória."""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from typing import Any, List, Dict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.base_architecture import BaseArchitectureML
from core.config import get_absolute_output_path
from core.validation import (DataIntegrityValidator, TemporalValidator,
                             assert_lag_columns)
from core.logging_config import get_logger, log_ml_pipeline


class DataFrameLibArchitectureML(BaseArchitectureML):
    """Implementação do pipeline ML para a arquitetura Polars DataFrame.

    A classe mantém simetria metodológica com as versões Data Lake e Data Warehouse:
    usa os mesmos folds temporais (QP1), garante equivalência de features e validações
    (QP2) e registra todos os artefatos necessários para o benchmark (QP3).

    O processamento utiliza Polars com lazy evaluation (LazyFrames) para otimização
    de memória e expressions idiomáticas para transformações, diferindo no paradigma
    mas mantendo equivalência nos resultados finais.
    """

    PARADIGM_META = {
        'name': 'dataframe_lib',
        'label': 'DataFrame Library (Polars)',
        'processor_module': 'collection.dataframe_lib.processor',
        'processor_class': 'DataFrameLibProcessor',
        'processor_run_method': 'run_dataframe_lib_processing',
        'baseline_module': 'architectures_ml.dataframe_lib.models.baseline_analysis',
        'baseline_class': 'BaselineModelAnalysisDataFrameLib',
        'hierarchical_module': 'architectures_ml.dataframe_lib.models.hierarchical_model',
        'hierarchical_class': 'HierarchicalModelDataFrameLib',
        'setup_script': 'src/architectures_ml/dataframe_lib/setup.py',
        'processor_script': 'src/collection/dataframe_lib/processor.py',
        'baseline_script': 'src/architectures_ml/dataframe_lib/models/baseline_analysis.py',
        'hierarchical_script': 'src/architectures_ml/dataframe_lib/models/hierarchical_model.py',
        # Declarado aqui porque os três paradigmas gravam em layouts
        # distintos; sem isso um módulo de análise precisa conhecer o
        # layout de cada paradigma para encontrar seus resultados.
        'master_artifact': {'kind': 'parquet',
                            'path': 'ml_pipeline/architectures/dataframe_lib/prep/'
                                    'master_data_dataframe_lib.parquet'},
        'baseline_results_json': 'ml_pipeline/architectures/dataframe_lib/models/baseline_results/baseline_analysis_dataframe_lib_results.json',
    }

    def _safe_write_parquet_file(self, df: pl.DataFrame, file_path: str) -> None:
        """
        Escreve arquivo Parquet com tratamento defensivo de conflitos.

        Args:
            df: DataFrame Polars para persistência
            file_path: Caminho de destino para arquivo Parquet

        Tratamento de conflitos:
            - Criação de diretórios pai se ausentes
            - Remoção de arquivos/diretórios conflitantes pré-existentes
            - Escrita atômica via polars.write_parquet
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        df.write_parquet(file_path)

    def __init__(self):
        """Inicializa paths, validadores e logging para o pipeline Polars DataFrame."""
        # Inicialização da arquitetura base
        output_base = get_absolute_output_path('ml_pipeline/architectures/dataframe_lib')
        super().__init__(architecture_name='dataframe_lib', output_base_path=output_base)

        self.logger = get_logger(__name__, with_ml_context=True)
        self.logger.set_context(architecture='dataframe_lib', module='setup')

        print("Inicializando Pipeline ML Polars")
        print("Lazy evaluation com expressions Polars")

        self.parquet_path = get_absolute_output_path('collection/dataframe_lib/processed/final_results.parquet')
        self.fallback_path = get_absolute_output_path('collection/dataframe_lib/raw')

        self.temporal_validator = TemporalValidator(min_gap_years=2)
        self.data_validator = DataIntegrityValidator()

        print(f"  Diretorio base: {self.output_base}")
        print(f"  Dados primarios: {self.parquet_path}")
        print(f"  Dados raw (fallback): {self.fallback_path}")
        print("  Lazy evaluation com expressoes Polars")

    def setup_environment(self) -> None:
        """
        Configura ambiente Polars com otimizações para ML temporal.

        Configurações aplicadas:
            1. String cache: Habilitado para otimização de memória
            2. Streaming: Modo lazy para datasets >1GB
            3. Random seeds: Determinismo em operações estocásticas

        Justificativa dos parâmetros:
            - String cache: Reduz overhead de strings em datasets educacionais
            - Lazy evaluation: Otimização automática de operações
            - Seeds: Controla amostragem e transformações estatísticas

        """
        print("Configurando Polars")

        pl.enable_string_cache()

        print("  String cache habilitado")
        print("  Lazy evaluation habilitado")

        random_seed = self.config['random_seed']
        np.random.seed(random_seed)

        print(f"  Seed {random_seed} configurado (NumPy)")

    def load_data(self) -> pl.DataFrame:
        """
        Carrega dados educacionais com lazy evaluation (LazyFrame) via Polars.

        Returns:
            pl.DataFrame: DataFrame Polars com dados carregados (após .collect())

        Raises:
            FileNotFoundError: Quando nem dados processados nem raw estão disponíveis

        Estratégia de carregamento hierárquica:
            1. Processed data: Dados pós-pipeline Data Lake (Parquet otimizado)
            2. Raw partitioned: Fallback para dados brutos particionados
            3. Error handling: Logging detalhado para debugging

        Vantagens Polars:
            - Lazy evaluation via scan_parquet() para datasets >RAM
            - Apache Arrow engine nativo para performance
            - Expressões idiomáticas para transformações eficientes

        load_data retorna DataFrame coletado (necessario para compatibilidade
        com base class). Lazy evaluation utilizada internamente em transformacoes.
        """
        self.logger.info("Iniciando carregamento com lazy evaluation Polars")
        print("\nCarregando dados (lazy loading)")

        lf = None
        data_source = None

        # Estratégia 1: Dados processados (otimizados)
        if os.path.exists(self.parquet_path):
            try:
                lf = pl.scan_parquet(self.parquet_path)
                data_source = "processed"
                print(f"  LazyFrame carregado")
            except (OSError, pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
                self.logger.warning(f"Erro ao carregar dados processados: {e}")
                print(f"  [ERROR] Dados processados: {e}")

        # Estratégia 2: Dados raw particionados (fallback)
        if lf is None and os.path.exists(self.fallback_path):
            try:
                print("  Fallback para dados raw particionados...")
                # Usar glob para arquivos parquet particionados
                import glob
                parquet_files = glob.glob(f"{self.fallback_path}/**/*.parquet", recursive=True)
                if parquet_files:
                    lf = pl.scan_parquet(f"{self.fallback_path}/*.parquet")
                    data_source = "raw_partitioned"
                    print("  Carregamento raw ok")
            except (OSError, pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
                self.logger.error(f"Erro ao carregar dados raw: {e}")
                print(f"  [ERROR] Dados raw: {e}")

        # Validação de carregamento
        if lf is None:
            raise FileNotFoundError(
                "Dados Polars DataFrame não encontrados em nenhuma fonte.\n"
                f"Verificar: {self.parquet_path} ou {self.fallback_path}\n"
                "Execute 'dataframe_lib/processor.py' para gerar dados processados."
            )

        # Análise de adequação

        # Operações lazy, computadas uma única vez
        stats_lf = lf.select([
            pl.col('year').min().alias('year_min'),
            pl.col('year').max().alias('year_max'),
            pl.col('country_code').n_unique().alias('n_countries'),
            pl.len().alias('total_rows')
        ])

        computed_stats = stats_lf.collect().row(0)
        year_min, year_max, n_countries, total_rows = computed_stats

        # Análise de adequação para ML temporal
        years_span = year_max - year_min + 1
        avg_obs_per_country = total_rows / n_countries if n_countries > 0 else 0

        print(f"  {year_min}-{year_max} ({years_span} anos)")
        print(f"  {n_countries} paises ({avg_obs_per_country:.1f} obs/pais)")
        print(f"  {total_rows:,} observacoes totais")
        print(f"  Fonte: {data_source}")

        if years_span < 10:
            print("  [WARN] Serie temporal curta pode limitar validacao walk-forward")

        if n_countries < 15:
            print("  [WARN] Poucos paises podem afetar generalizacao geografica")

        self.logger.info(f"Dados carregados com sucesso via {data_source}")

        # Retornar DataFrame coletado para compatibilidade com base class
        return lf.collect()

    @log_ml_pipeline('validation')
    def validate_data(self, df: pl.DataFrame) -> None:
        """
        Executa validação com amostragem estratégica Polars.

        Args:
            df: DataFrame Polars com dados educacionais carregados

        Metodologia de validação:
            1. Amostragem adaptativa: min(1000, total_rows) para eficiência
            2. DataIntegrityValidator: Validador centralizado para consistência
            3. Schema validation: Verificação de colunas obrigatórias
            4. Range validation: Detecção de valores impossíveis
            5. Fallback inteligente: Busca automática de variáveis alternativas

        Critérios:
            - Target coverage >50%: Poder estatístico adequado para ML
            - Range [0,100]: Consistência com definições educacionais
            - Schema compliance: Presença de identificadores temporais/geográficos
        """
        print("Validando dados")

        # Amostragem adaptativa para validação eficiente
        total_rows = len(df)
        sample_size = min(1000, total_rows)

        print(f"  Amostragem: {sample_size:,}/{total_rows:,} ({sample_size/total_rows:.1%})")

        # Amostragem com seed reprodutível
        sample_df = df.sample(n=sample_size, seed=self.config['random_seed'])
        sample_pd = sample_df.to_pandas()

        # Validação centralizada com DataIntegrityValidator
        is_valid, validation_report = self.data_validator.validate_dataframe(
            sample_pd,
            target_col=self.source_column,
            check_completeness=True
        )

        if not is_valid:
            warnings = validation_report.get('warnings', [])
            self.logger.warning(f"Problemas de integridade detectados: {len(warnings)} warnings")
            for warning in warnings[:3]:
                print(f"  [WARN] {warning}")

        # The configured target must exist. Substituting a similarly named
        # column would silently move the experiment to a different target,
        # invalidating every downstream comparison.
        if self.source_column not in df.columns:
            raise ValueError(
                f"Target column '{self.source_column}' declared by "
                f"{type(self.dataset_config).__name__} is absent from the "
                f"processed data. Available columns: {sorted(df.columns)}"
            )

        # Análise de qualidade via Polars

        # Computações via Polars expressions
        stats_lf = df.lazy().select([
            pl.col(self.source_column).is_not_null().sum().alias('target_data'),
            pl.col(self.source_column).min().alias('target_min'),
            pl.col(self.source_column).max().alias('target_max'),
            pl.col(self.source_column).mean().alias('target_mean'),
            (pl.col(self.source_column) > 100).sum().alias('over_100_count'),
            (pl.col(self.source_column) < 0).sum().alias('under_0_count'),
            pl.len().alias('total_rows')
        ])

        computed = stats_lf.collect().row(0)
        target_data, target_min, target_max, target_mean, over_100_count, under_0_count, total_rows_check = computed

        target_coverage = (target_data / total_rows_check) * 100 if total_rows_check > 0 else 0

        print(f"  Cobertura: {target_data:,}/{total_rows_check:,} validos ({target_coverage:.1f}%)")
        print(f"  Range: [{target_min:.1f}%, {target_max:.1f}%]")
        print(f"  Media: {target_mean:.1f}%")

        if target_coverage < 50:
            print("  [WARN] Baixa cobertura de target (<50%) pode comprometer ML")

        if over_100_count > 0:
            print(f"  [WARN] {over_100_count} valores >100% (dados invalidos)")

        if under_0_count > 0:
            print(f"  [WARN] {under_0_count} valores <0% (dados invalidos)")

        # Validação de schema obrigatório
        required_cols = ['country_code', 'year']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Schema incompleto para ML temporal: colunas ausentes {missing_cols}.\n"
                "Identificadores país-ano são obrigatórios para validação walk-forward."
            )

        print("  Validacao concluida")

    def create_target_implementation(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Constrói variável target via transformação Polars com expressions idiomáticas.

        Args:
            df: DataFrame Polars com dados educacionais

        Returns:
            DataFrame Polars enriquecido com variável target dropout_rate_dataframe_lib

        Transformação:
            Dropout Rate = 100 - Completion Rate

        Implementação Polars:
            Utiliza .with_columns() com expressions para eficiência, criando
            lags temporais via join temporal (year+k) por country_code.

        """
        print(f"Construindo target: {self.source_column} -> {self.target_column}")
        print("  Dropout Rate = 100 - Completion Rate")

        # Validação de range [0, 100]
        df_with_target = df.with_columns([
            pl.when(
                (pl.col(self.source_column) >= 0) & (pl.col(self.source_column) <= 100)
            ).then(100 - pl.col(self.source_column)).otherwise(None).alias(self.target_column)
        ])

        # Lag temporal via join (valor de exatamente N anos atrás),
        # não shift posicional que assume dados sem gaps anuais.
        print("  Criando lag features (dropout_rate_lag_2, lag_3)")
        try:
            base_lag = df_with_target.select(['country_code', 'year', self.target_column])

            # Lag 2 anos: join com year+2 traz o valor de 2 anos atrás
            lag2 = base_lag.with_columns(
                (pl.col('year') + 2).alias('year')
            ).rename({self.target_column: 'dropout_rate_lag_2'})
            df_with_target = df_with_target.join(
                lag2, on=['country_code', 'year'], how='left'
            )

            # Lag 3 anos: idem para 3 anos atrás
            lag3 = base_lag.with_columns(
                (pl.col('year') + 3).alias('year')
            ).rename({self.target_column: 'dropout_rate_lag_3'})
            df_with_target = df_with_target.join(
                lag3, on=['country_code', 'year'], how='left'
            )

            print("  dropout_rate_lag_2 e dropout_rate_lag_3 criados (join temporal country/year-k)")
        except (pl.exceptions.ColumnNotFoundError, pl.exceptions.ComputeError,
                KeyError) as exc:
            raise ValueError(
                f"dataframe_lib: falha ao criar as defasagens do alvo: {exc}"
            ) from exc

        print("  Target criado via Polars expressions")

        assert_lag_columns(df_with_target.collect_schema().names(),
                           'dataframe_lib', self.TARGET_LAG_ORDERS)
        return df_with_target

    def _compute_target_statistics(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Computa estatísticas descritivas da variável target via Polars.

        Args:
            df: DataFrame Polars com variável target criada

        Returns:
            Dicionário com estatísticas float64 para análise

        Estatísticas computadas:
            - Momentos: média, desvio padrão
            - Range: mínimo, máximo para detecção de outliers
            - Completude: contagem válida vs missing para análise de qualidade

        Otimização Polars:
            Agregações via expressions lazy, computadas uma única vez.
        """
        # Computação via Polars expressions
        stats_lf = df.lazy().select([
            pl.col(self.target_column).mean().alias('mean'),
            pl.col(self.target_column).std().alias('std'),
            pl.col(self.target_column).min().alias('min'),
            pl.col(self.target_column).max().alias('max'),
            pl.col(self.target_column).is_null().sum().alias('missing_count'),
            pl.col(self.target_column).is_not_null().sum().alias('valid_count')
        ])

        computed = stats_lf.collect().row(0)
        mean_val, std_val, min_val, max_val, missing_count, valid_count = computed

        # Conversão para float64 para consistência
        return {
            'mean': float(mean_val) if mean_val is not None else 0.0,
            'std': float(std_val) if std_val is not None else 0.0,
            'min': float(min_val) if min_val is not None else 0.0,
            'max': float(max_val) if max_val is not None else 0.0,
            'missing_count': int(missing_count) if missing_count is not None else 0,
            'valid_count': int(valid_count) if valid_count is not None else 0
        }

    def _validate_temporal_folds(self, df: pl.DataFrame, folds: List[Dict]) -> None:
        """Validação temporal com TemporalValidator via Polars."""
        print("Validando folds temporais")

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

            # Filtros para contagem via Polars expressions
            train_filter = (
                (pl.col('year') >= fold['train_start']) &
                (pl.col('year') <= fold['train_end']) &
                ~((pl.col('year') >= fold['train_gap_start']) &
                  (pl.col('year') <= fold['train_gap_end']))
            )
            val_filter = (
                (pl.col('year') >= fold['val_start']) &
                (pl.col('year') <= fold['val_end'])
            )
            test_filter = (
                (pl.col('year') >= fold['test_start']) &
                (pl.col('year') <= fold['test_end']) &
                ~((pl.col('year') >= fold['val_gap_start']) &
                  (pl.col('year') <= fold['val_gap_end']))
            )

            # Contar dados por fold via Polars
            fold_stats = df.lazy().select([
                train_filter.sum().alias('train_count'),
                val_filter.sum().alias('val_count'),
                test_filter.sum().alias('test_count'),
                pl.when(train_filter).then(pl.col('country_code')).n_unique().alias('train_countries'),
                pl.when(val_filter).then(pl.col('country_code')).n_unique().alias('val_countries'),
                pl.when(test_filter).then(pl.col('country_code')).n_unique().alias('test_countries'),
            ]).collect()

            fold_row = fold_stats.row(0)
            fold['train_count'] = int(fold_row[0]) if fold_row[0] is not None else 0
            fold['val_count'] = int(fold_row[1]) if fold_row[1] is not None else 0
            fold['test_count'] = int(fold_row[2]) if fold_row[2] is not None else 0
            fold['train_countries'] = int(fold_row[3]) if fold_row[3] is not None else 0
            fold['val_countries'] = int(fold_row[4]) if fold_row[4] is not None else 0
            fold['test_countries'] = int(fold_row[5]) if fold_row[5] is not None else 0

            print(f"\n  Fold {fold['fold_id']}:")
            print(f"    Train: {fold['train_count']} obs, {fold['train_countries']} paises")
            print(f"    Val: {fold['val_count']} obs, {fold['val_countries']} paises")
            print(f"    Test: {fold['test_count']} obs, {fold['test_countries']} paises")

    def discover_numeric_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Identifica colunas numéricas via schema de tipos nativo de Polars.

        Args:
            df: DataFrame Polars com dados educacionais

        Returns:
            Lista de nomes de colunas numéricas
        """
        return [
            col for col in df.columns
            if df[col].dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ]
        ]

    def compute_feature_correlations(self, df: pl.DataFrame,
                                     features: List[str]) -> Dict[str, float]:
        """
        Computa correlacoes de Pearson feature-target usando dados completos.

        Args:
            df: DataFrame Polars com dados educacionais completos
            features: Lista de features candidatas para analise de correlacao

        Returns:
            Dicionario {feature_name: absolute_correlation} para ranking

        Metodologia:
            1. Materializa dados completos de treino (sem amostragem)
            2. Calcula correlacao Pearson via pandas para cada feature
            3. Retorna valor absoluto para ranking por relevancia
        """
        print("Analisando correlacoes feature-target")

        target_col = self.target_column
        correlations = {}

        sample_pd = df.select([target_col] + features).drop_nulls(subset=[target_col]).to_pandas()

        print(f"  Dados materializados: {len(sample_pd):,} obs, {len(features)} features")

        successful_correlations = 0
        failed_features = []

        for feat in features:
            if feat not in sample_pd.columns:
                correlations[feat] = 0.0
                continue

            try:
                corr = sample_pd[feat].corr(sample_pd[target_col])

                if pd.isna(corr):
                    correlations[feat] = 0.0
                else:
                    correlations[feat] = abs(float(corr))
                    successful_correlations += 1

            except (ValueError, TypeError, pl.exceptions.ComputeError) as e:
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

    def apply_collinearity_filter(self, df: pl.DataFrame, features: List[str],
                                   threshold: float = 0.8) -> List[str]:
        """
        Remove multicolinearidade via filtragem greedy de correlação pairwise.

        Para cada feature candidata, calcula a correlação absoluta máxima com
        as features já selecionadas e rejeita se max |r| >= threshold.

        Args:
            df: DataFrame Polars com features candidatas
            features: Lista de features para análise de multicolinearidade
            threshold: Limiar de correlação pairwise (padrão 0.8)

        Returns:
            Lista filtrada de features com multicolinearidade reduzida

        Algoritmo greedy:
            1. Primeira feature sempre aceita (baseline)
            2. Features subsequentes aceitas se max |r| < threshold
            3. Ordem preservada para determinismo

        Materializacao:
            - Dados completos de treino (sem amostragem)
            - Matriz de correlacao via pandas apos conversao
        """
        if len(features) <= 1:
            print("  Menos de 2 features - colinearidade desnecessaria")
            return features

        print(f"Filtrando colinearidade: {len(features)} features")

        try:
            corr_data = df.select(features).to_pandas().dropna()

            valid_rows = len(corr_data)
            print(f"  {valid_rows:,} observacoes validas pos-dropna")

            if valid_rows > 10:
                corr_matrix = corr_data.corr().abs()

                selected = []
                rejected_count = 0
                features = sorted(features)

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
                print(f"  Dados insuficientes ({valid_rows}<=10) - fallback top-10")
                return features[:10]

        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            self.logger.error(f"Erro na filtragem de colinearidade: {e}")
            print(f"[ERROR] Filtragem de colinearidade falhou: {e}")
            print("  Fallback: retornando top-10 features")
            return features[:10]

    def prepare_features(self, df: pl.DataFrame, selected_features: List[str]) -> pl.DataFrame:
        """
        Prepara features finais para ML com transformações Polars idiomáticas.

        Args:
            df: DataFrame Polars com features selecionadas via filtragem de colinearidade
            selected_features: Features pós-seleção para transformação

        Returns:
            DataFrame Polars enriquecido com features originais + transformadas

        Engenharia de Features Científica:
            Aplica symmetric log transform: T(x) = sign(x) * ln(|x| + 1)
            às top-5 features para normalização de distribuições assimétricas.

        Justificativas metodológicas:
            1. Top-5 limite: Baseado em curse of dimensionality (Bellman, 1961)
            2. Symmetric log: Trata zeros e negativos naturalmente
            3. Polars expressions: Transformações eficientes via expressions

        Estrutura final:
            - Metadados: country_code, year, target (essenciais ML temporal)
            - Features originais: selected_features (pós-filtragem)
            - Features transformadas: {feature}_log_transform (top-5)
        """
        print("\nFeature engineering")

        # Critério: Limitar escopo por curse of dimensionality
        features_to_transform = selected_features[:5] if len(selected_features) > 5 else selected_features
        transformed_count = 0

        print(f"  Transformando {len(features_to_transform)} features (symmetric log):")

        # Aplicação de transformação via expressions Polars
        new_cols = []
        for feat in features_to_transform:
            if feat not in df.columns:
                print(f"    {feat}: AUSENTE (ignorado)")
                continue

            transform_col = f"{feat}_log_transform"

            print(f"    {feat} -> {transform_col}")

            # Symmetric log transform: sign(x) * ln(|x| + 1)
            new_cols.append(
                pl.when(pl.col(feat).is_null())
                .then(None)
                .otherwise(pl.col(feat).sign() * (pl.col(feat).abs() + 1).log())
                .alias(transform_col)
            )
            transformed_count += 1

        # Aplicar todas as transformações de uma vez via with_columns
        if new_cols:
            df = df.with_columns(new_cols)

        print(f"  {transformed_count} log transforms aplicadas")

        # Construção de dataset ML final

        # Metadados essenciais para ML temporal
        ml_features = ['country_code', 'year', self.target_column]

        # Features originais pós-filtragem de colinearidade
        ml_features.extend(selected_features)

        # Features transformadas
        transformed_cols = [f"{feat}_log_transform" for feat in features_to_transform
                          if f"{feat}_log_transform" in df.columns]
        ml_features.extend(transformed_cols)

        # Incluir lags do target no dataset salvo
        for lag_col in ['dropout_rate_lag_2', 'dropout_rate_lag_3']:
            if lag_col in df.columns and lag_col not in ml_features:
                ml_features.append(lag_col)

        # Remover duplicatas preservando ordem
        ml_features = list(dict.fromkeys(ml_features))
        ml_features = [col for col in ml_features if col in df.columns]

        print(f"  Dataset ML final: {len(ml_features)} variaveis "
              f"({len(selected_features)} originais, {len(transformed_cols)} transformadas)")

        # Seleção final
        result_df = df.select(ml_features)

        print("  Feature engineering concluido")

        return result_df

    def save_folds(self, df: pl.DataFrame, folds: List[Dict]) -> None:
        """
        Salva folds como arquivos Parquet, mantendo paradigma Polars DataFrame.

        Args:
            df: DataFrame Polars processado
            folds: Lista de configurações de folds
        """
        print("\nSalvando folds Polars")

        for fold in folds:
            fold_id = fold['fold_id']
            fold_dir = f"{self.prep_dir}/folds/fold_{fold_id}"
            os.makedirs(fold_dir, exist_ok=True)

            print(f"  Processando fold {fold_id}...")

            # Filtros via Polars expressions
            train_filter = (
                (pl.col('year') >= fold['train_start']) &
                (pl.col('year') <= fold['train_end']) &
                ~((pl.col('year') >= fold['train_gap_start']) &
                  (pl.col('year') <= fold['train_gap_end']))
            )
            val_filter = (
                (pl.col('year') >= fold['val_start']) &
                (pl.col('year') <= fold['val_end'])
            )
            test_filter = (
                (pl.col('year') >= fold['test_start']) &
                (pl.col('year') <= fold['test_end']) &
                ~((pl.col('year') >= fold['val_gap_start']) &
                  (pl.col('year') <= fold['val_gap_end']))
            )

            # Filtragem e salvamento
            try:
                train_df = df.filter(train_filter)
                val_df = df.filter(val_filter)
                test_df = df.filter(test_filter)

                self._safe_write_parquet_file(train_df, f'{fold_dir}/train_data_dataframe_lib.parquet')
                self._safe_write_parquet_file(val_df, f'{fold_dir}/val_data_dataframe_lib.parquet')
                self._safe_write_parquet_file(test_df, f'{fold_dir}/test_data_dataframe_lib.parquet')

                print(f"    Fold {fold_id}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

            except Exception as e:
                print(f"    [ERROR] Salvamento fold {fold_id}: {e}")
                raise

            fold_metadata = {
                **fold,
                'storage_method': 'parquet_files',
                'paradigm': 'dataframe_lib'
            }
            self.save_fold_metadata(fold_metadata, fold_dir)

        # Master data
        print("\n  Salvando master data...")
        try:
            master_path = f"{self.prep_dir}/master_data_dataframe_lib.parquet"
            self._safe_write_parquet_file(df, master_path)
            print(f"    Master data: {len(df)} registros")

        except Exception as e:
            print(f"    [ERROR] Salvamento master data: {e}")
            raise

        # Configuração master
        total_obs = len(df)
        total_countries = df.select(pl.col('country_code').n_unique()).item()
        year_min = int(df.select(pl.col('year').min()).item())
        year_max = int(df.select(pl.col('year').max()).item())

        self.save_master_config(folds, total_obs, total_countries, (year_min, year_max))

        print(f"  Polars: folds salvos")


def main():
    """Executa o pipeline Polars DataFrame end-to-end para validação local."""
    print("=" * 80)
    print("Pipeline ML Polars")
    print("=" * 80)

    try:
        setup = DataFrameLibArchitectureML()
        results = setup.run_setup()

        if results.get('status') == 'success':
            print("Pipeline ok")
            print(f"  Features selecionadas: {results.get('features_selected', 'N/A')}")
            print(f"  Folds criados: {results.get('folds_created', 'N/A')}")
            print(f"  Timestamp: {results.get('setup_timestamp', 'N/A')}")
        else:
            print(f"[ERROR] Pipeline falhou: {results.get('error', 'Erro desconhecido')}")
            return results

        return results

    except Exception as e:
        print(f"\n[ERROR] Pipeline falhou: {e}")
        import traceback
        traceback.print_exc()
        return {
            'architecture': 'dataframe_lib',
            'status': 'failed',
            'error': str(e),
            'setup_timestamp': datetime.now().isoformat()
        }


if __name__ == '__main__':
    # sys.exit, não o builtin exit: este último vem do módulo site e pode não
    # existir sob python -O ou em ambiente embarcado.
    results = main()
    sys.exit(0 if results.get('status') == 'success' else 1)
