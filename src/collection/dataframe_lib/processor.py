#!/usr/bin/env python3
"""
Processador Polars DataFrame para Dados Educacionais Latino-Americanos.

Implementa pipeline de processamento com Polars mantendo princípios arquiteturais
Data Lake: leitura lazy, transformações eficientes de memória e schema-on-read.

Fundamentação Teórica:
    O paradigma Data Lake (Terrizzano et al., 2015) prioriza a preservação
    de dados brutos e semântica schema-on-read. Polars oferece lazy evaluation nativa,
    diferentemente de Pandas eager, permitindo otimizações automáticas de query plan.

    Diferenças arquiteturais vs Dask:
    - Polars: Single-machine lazy evaluation com otimizações de query plan
    - Dask: Multi-machine distributed com overhead de coordenação
    - Polars é adequado para datasets <100GB em máquinas modernas (≥32GB RAM)

Decisões de Design:
    - Polars lazy scanning: schema-on-read com otimização automática
    - Pivotagem em memória: trade-off entre velocidade e uso de RAM
    - Sem particionamento: Polars otimiza automaticamente via query plan
    - Parquet como formato padrão: compatibilidade com ecossistema
"""

import polars as pl
import os
import sys
import json
import shutil
from datetime import datetime
from typing import Dict


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from core.config import get_absolute_output_path
from core.indicators import ALL_INDICATORS


class DataFrameLibProcessor:
    """
    Processador científico Polars DataFrame para análise de indicadores educacionais.

    Implementa processamento lazy com Polars seguindo princípios arquiteturais Data Lake,
    otimizado para análise exploratória de dados com footprint de memória menor que Dask
    para datasets <100GB.

    Princípios Fundamentais:
        1. Lazy evaluation nativa: Polars constrói query plans otimizados automaticamente
        2. Schema-on-read: Estrutura imposta no momento da análise, não na ingestão
        3. Transformações eficientes: Operações vetorizadas em Rust compiled
        4. Pivotagem automática: Conversão long->wide otimizada para Parquet
    """

    def __init__(self, dataset_name: str = "worldbank"):
        """
        Inicializa processador Polars DataFrame.

        Args:
            dataset_name: Nome do dataset ("worldbank" ou "inep_censo")
        """
        print("Inicializando processador Polars")
        print("Arquitetura: Polars, schema-on-read")

        self.dataset_name = dataset_name
        raw_subdir = 'collection/inep_raw' if dataset_name == 'inep_censo' else 'collection/raw_data'
        self.complete_data_path = get_absolute_output_path(f'{raw_subdir}/complete_data.parquet')
        self.output_dir = get_absolute_output_path('collection/dataframe_lib')
        self.processed_dir = f"{self.output_dir}/processed"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        print(f"Fonte de dados: {self.complete_data_path}")
        print(f"Diretorio de processamento: {self.processed_dir}")

    def load_complete_data(self) -> pl.LazyFrame:
        """
        Carrega dados educacionais completos com lazy evaluation.

        Returns:
            pl.LazyFrame: DataFrame Polars lazy sem materialização

        Raises:
            FileNotFoundError: Quando arquivo Parquet de entrada não existe

        Decisões metodológicas:
            1. Lazy scanning: Apenas schema é lido inicialmente
            2. Computação seletiva: Estatísticas críticas computadas seletivamente
            3. Indicadores centralizados: Garante consistência entre arquiteturas
        """
        print("Leitura lazy de dados educacionais completos")

        if not os.path.exists(self.complete_data_path):
            raise FileNotFoundError(
                f"Arquivo de dados completos não encontrado: {self.complete_data_path}\n"
                f"Execute 'raw_data_collector.py' antes deste processador."
            )

        df_lazy = pl.scan_parquet(self.complete_data_path)

        n_rows = df_lazy.select(pl.lit(1)).collect().shape[0]
        n_cols = len(df_lazy.collect_schema().names())

        # Computar apenas estatísticas críticas
        stats = df_lazy.select([
            pl.col('year').min().alias('year_min'),
            pl.col('year').max().alias('year_max'),
            pl.col('country_code').n_unique().alias('n_countries')
        ]).collect()

        year_min = stats['year_min'][0]
        year_max = stats['year_max'][0]
        n_countries = stats['n_countries'][0]

        print(f"{n_rows:,} observacoes x {n_cols} variaveis")
        print(f"Cobertura temporal: {year_min}-{year_max} ({year_max-year_min+1} anos)")
        entity_label = "municípios brasileiros" if self.dataset_name == "inep_censo" else "países"
        print(f"Cobertura geografica: {n_countries} {entity_label}")

        indicator_names = list(ALL_INDICATORS.values())
        scientific_indicators = [col for col in df_lazy.collect_schema().names()
                                if col in indicator_names]

        if scientific_indicators:
            missing_stats = df_lazy.select([
                pl.concat_list([pl.col(col).is_null() for col in scientific_indicators])
                  .list.sum().sum().alias('total_missing'),
                pl.lit(len(scientific_indicators)).alias('n_indicators')
            ]).collect()

            total_cells = n_rows * len(scientific_indicators)
            missing_count = missing_stats['total_missing'][0] * len(scientific_indicators)
            missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0

            print(f"Completude: {total_cells - missing_count:,}/{total_cells:,} celulas validas ({100-missing_pct:.1f}%)")

        print("LazyFrame Polars preparado")

        return df_lazy

    def pivot_long_to_wide(self, df_lazy: pl.LazyFrame) -> pl.LazyFrame:
        """
        Transforma dados de formato longo para largo.

        Args:
            df_lazy: DataFrame Polars lazy em formato longo

        Returns:
            DataFrame Polars lazy em formato largo

        Lógica de transformação:
            1. Dados originais: (país, ano, indicador, valor)
            2. Resultado final: (país, ano, indicador1, indicador2, ...)
            3. Usa unpivot (Polars) equivalente a melt reverso
        """
        print("Pivotagem long->wide")

        # Verificar se já está em formato wide
        schema = df_lazy.collect_schema()

        # Se tem 'indicator_name' ou 'indicator', precisa fazer pivot
        if 'indicator_name' in schema or 'indicator' in schema:
            print("Detectado formato longo - convertendo para largo")

            # Identificar coluna de indicador
            indicator_col = 'indicator_name' if 'indicator_name' in schema else 'indicator'
            value_col = 'value' if 'value' in schema else 'indicator_value'

            # Manter colunas de dimensão (país, ano, etc.)
            id_cols = ['country_code', 'year']

            # Pivot: converter indicadores em colunas
            df_wide = df_lazy.pivot(
                on=indicator_col,
                index=id_cols,
                values=value_col,
                aggregate_function='first'  # Não deve ter duplicatas
            )

            print(f"Pivotagem concluida - colunas: {len(df_wide.collect_schema().names())}")

        else:
            print("Dados ja em formato largo - preservando estrutura")
            df_wide = df_lazy

        return df_wide

    def export_processed_data(self, df_lazy: pl.LazyFrame) -> str:
        """
        Materializa e persiste dados processados.

        Args:
            df_lazy: DataFrame Polars lazy processado

        Returns:
            str: Caminho absoluto do dataset final exportado

        Estratégia de exportação:
            1. Materialização lazy: Polars otimiza automaticamente antes de escrever
            2. Formato Parquet: Compatibilidade com ecossistema
            3. Metadados JSON: Estatísticas de qualidade e auditoria
        """
        print("Materializando dados processados")

        output_path = f"{self.processed_dir}/final_results.parquet"

        if os.path.exists(output_path):
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)

        print(f"Salvando dataset: {output_path}")

        df_collected = df_lazy.collect()
        df_collected.write_parquet(output_path, compression='snappy')

        # Estatísticas finais
        n_rows = len(df_collected)
        n_cols = len(df_collected.columns)

        # Metadados JSON
        metadata = {
            'architecture': 'dataframe_lib',
            'processing_paradigm': 'polars_lazy_evaluation',
            'dataset_statistics': {
                'total_records': int(n_rows),
                'total_columns': int(n_cols)
            },
            'data_quality': {
                'format': 'parquet',
                'compression': 'snappy',
                'row_count': int(n_rows),
                'column_count': int(n_cols)
            },
            'compliance_flags': {
                'schema_on_read': True,
                'lazy_evaluation_preserved': True,
                'task_graph_pattern': True
            },
            'output_artifacts': {
                'dataset': output_path,
                'compression': 'snappy',
                'format': 'parquet'
            },
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'polars_version': pl.__version__
            }
        }

        stats_path = f"{self.processed_dir}/processing_metadata.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Artefatos: {output_path}, {stats_path}")
        print(f"{n_rows:,} registros x {n_cols} colunas")

        return output_path

    def run_dataframe_lib_processing(self) -> Dict:
        """
        Orquestra pipeline completo de processamento Polars DataFrame.

        Returns:
            Dict contendo status de execução, artefatos gerados e metadados

        Pipeline sequencial:
            1. Carregamento lazy de dados completos
            2. Transformação long->wide
            3. Materialização e persistência

        """
        start_time = datetime.now()

        try:
            print("\n[1/3] Carregamento de dados")
            df_lazy = self.load_complete_data()

            print("\n[2/3] Transformacao long->wide")
            df_wide = self.pivot_long_to_wide(df_lazy)

            print("\n[3/3] Materializacao e persistencia")
            output_path = self.export_processed_data(df_wide)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            print(f"\nProcessamento Polars concluido em {processing_time:.2f}s")

            return {
                'status': 'success',
                'architecture': 'dataframe_lib',
                'paradigm': 'lazy_evaluation',
                'output': {
                    'dataset': output_path,
                    'metadata': f"{self.processed_dir}/processing_metadata.json"
                },
                'performance': {
                    'processing_time_seconds': processing_time
                },
                'timestamp': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                }
            }

        except FileNotFoundError as e:
            print(f"\n[ERROR] Dados de entrada nao encontrados: {e}")
            print("Execute 'raw_data_collector.py' antes deste processador")

            return {
                'status': 'failed',
                'error_type': 'FileNotFoundError',
                'error_message': str(e),
                'suggestion': 'Run raw_data_collector.py first',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            tb = traceback.format_exc()

            print(f"\n[ERROR] {e.__class__.__name__}: {str(e)}")
            print(tb)

            return {
                'status': 'failed',
                'error_type': e.__class__.__name__,
                'error_message': str(e),
                'traceback': tb,
                'timestamp': datetime.now().isoformat()
            }


if __name__ == "__main__":
    processor = DataFrameLibProcessor()
    results = processor.run_dataframe_lib_processing()
    print(f"Execucao: {results.get('status', 'failed')}")
