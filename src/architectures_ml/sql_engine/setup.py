#!/usr/bin/env python3
"""Setup reprodutível do pipeline ML para a arquitetura Data Warehouse.

Executa o protocolo metodológico em modo SQL-first: abre a base DuckDB gerada
na fase de coleta, cria folds temporais com gaps anti-leak, mantém o mesmo
processo de engenharia de features usado na arquitetura Data Lake e exporta os
artefatos necessários para comparação. O foco é preservar simetria com o Data
Lake para que diferenças observadas reflitam características do paradigma
schema-on-write."""

import os
import sys
from typing import List, Dict, Any

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.base_architecture import BaseArchitectureML
from core.config import get_absolute_output_path

from collection.sql_engine.connection_manager import (
    DuckDBConnectionManager, 
    SQLProcessingError
)


class SqlEngineArchitectureML(BaseArchitectureML):
    """Implementação do pipeline ML para a arquitetura Data Warehouse.

    Reproduz o mesmo protocolo aplicado à arquitetura Data Lake,
    diferindo apenas pelo processamento SQL in-database. Todos os artefatos
    gerados (folds, estatísticas de target, matrizes de features) seguem o
    padrão do framework para permitir benchmark e equivalência prática."""

    PARADIGM_META = {
        'name': 'sql_engine',
        'label': 'SQL Engine (DuckDB)',
        'processor_module': 'collection.sql_engine.processor',
        'processor_class': 'SqlEngineProcessor',
        'processor_run_method': 'run_sql_engine_processing',
        'baseline_module': 'architectures_ml.sql_engine.models.baseline_analysis',
        'baseline_class': 'BaselineModelAnalysisSqlEngine',
        'hierarchical_module': 'architectures_ml.sql_engine.models.hierarchical_model',
        'hierarchical_class': 'HierarchicalModelSQLFirst',
        'setup_script': 'src/architectures_ml/sql_engine/setup.py',
        'processor_script': 'src/collection/sql_engine/processor.py',
        'baseline_script': 'src/architectures_ml/sql_engine/models/baseline_analysis.py',
        'hierarchical_script': 'src/architectures_ml/sql_engine/models/hierarchical_model.py',
    }

    def __init__(self):
        """Inicializa paths, conexão DuckDB e logger."""
        # Inicialização da arquitetura base
        output_base = get_absolute_output_path('ml_pipeline/architectures/sql_engine')
        super().__init__(architecture_name='sql_engine', output_base_path=output_base)
        
        print("Inicializando Pipeline ML DuckDB")
        print("SQL-first com validacao temporal")
        
        # Configurações específicas do Data Warehouse
        dataset_name = self.dataset_config.name
        self.db_path = get_absolute_output_path(f'collection/sql_engine/{dataset_name}_data.duckdb')
        self.conn_manager = None
        
        print(f"  Diretorio base: {self.output_base}")
        print(f"  DuckDB: {self.db_path}")
        print("  Zero file I/O, processamento SQL nativo sem cache")
    
    def setup_environment(self) -> None:
        """Abre a base DuckDB produzida na fase de coleta e inicializa o gerenciador.

        Levanta ``FileNotFoundError`` quando o banco não está disponível, sinalizando
        que a etapa de coleta precisa ser reexecutada. O `DuckDBConnectionManager`
        encapsula retries simples (3 tentativas, pausa de 1s) para evitar falhas
        transitórias e garante que as queries subsequentes permaneçam dentro do
        mesmo contexto transacional."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Banco DuckDB não encontrado: {self.db_path}\n"
                f"Execute 'sql_engine/processor.py' antes deste pipeline ML."
            )
        
        self.conn_manager = DuckDBConnectionManager(
            self.db_path,
            max_retries=3,      
            retry_delay=1.0
        )
        
        print("  Connection manager configurado")
        print("  ACID compliance habilitado")
    
    def load_data(self) -> None:
        """
        Executa carregamento de dados via SQL nativo.

        Returns:
            None: Dados permanecem in-database seguindo paradigma Data Warehouse

        Dados permanecem in-database -- materialização sob demanda via SQL.
        """
        print("\nAnalisando dados via SQL nativo")
        
        stats_query = """
            SELECT
                COUNT(*) as total_records,
                (SELECT COUNT(*) FROM information_schema.columns
                 WHERE table_name = 'analytics_wide') as total_columns,
                MIN(year) as min_year,
                MAX(year) as max_year,
                COUNT(DISTINCT country_code) as unique_countries,
                COUNT(DISTINCT year) as temporal_periods
            FROM analytics_wide
        """
        
        stats_result = self.conn_manager.execute_sql(stats_query).iloc[0]
        
        total_records = int(stats_result['total_records'])
        total_columns = int(stats_result['total_columns'])
        min_year = int(stats_result['min_year'])
        max_year = int(stats_result['max_year'])
        unique_countries = int(stats_result['unique_countries'])
        temporal_periods = int(stats_result['temporal_periods'])
        
        years_span = max_year - min_year + 1
        avg_obs_per_country = total_records / unique_countries if unique_countries > 0 else 0

        print(f"  {total_records:,} observacoes x {total_columns} variaveis")
        print(f"  {min_year}-{max_year} ({years_span} anos, {temporal_periods} periodos)")
        print(f"  {unique_countries} paises ({avg_obs_per_country:.1f} obs/pais)")
        
        if years_span < 10:
            print(f"  [WARN] Serie temporal curta ({years_span} anos) pode limitar validacao walk-forward")

        if total_records < 100:
            print(f"  [WARN] Dataset pequeno ({total_records} obs) pode afetar poder estatistico")
        
        # Paradigma Data Warehouse: dados nunca saem do banco
        return None
    
    def validate_data(self, data: Any) -> None:
        """
        Executa validação de integridade e adequação dos dados.
        
        Args:
            data: Ignorado - validação executada diretamente no banco de dados
            
        Validações implementadas:
            1. Schema validation: Verificação de existência de colunas essenciais
            2. Data coverage analysis: Análise de completude para variável target
            3. Temporal consistency: Verificação de continuidade das séries
            4. Geographic coverage: Análise de representatividade por país
        
        Critérios de adequação:
            - Mínimo 50 observações válidas para target (poder estatístico)
            - Presença obrigatória de country_code, year (identificadores únicos)
            - Target coverage >20% para evitar extreme class imbalance
        
        Aborts when the configured target column is absent, rather than
        substituting a similarly named one.
        """
        print("Validando integridade dos dados")
        
        target_source_col = self.source_column
        
        # 1. Validação de Schema
        print("  [1/4] Validacao de schema")
        column_exists = self.conn_manager.execute_scalar(f"""
            SELECT COUNT(*) > 0
            FROM information_schema.columns
            WHERE table_name = 'analytics_wide'
            AND column_name = '{target_source_col}'
        """)
        
        # The configured target must exist. Substituting a similarly named
        # column would silently move the experiment to a different target,
        # invalidating every downstream comparison.
        if not column_exists:
            available = self.conn_manager.execute_sql("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'analytics_wide'
                ORDER BY column_name
            """)
            raise ValueError(
                f"Target column '{target_source_col}' declared by "
                f"{type(self.dataset_config).__name__} is absent from "
                f"analytics_wide. Available columns: {available}"
            )
        
        # 2. Análise de Cobertura
        print("  [2/4] Analise de cobertura")
        coverage_stats = self.conn_manager.execute_sql(f"""
            SELECT
                COUNT(*) as total_records,
                COUNT({self.source_column}) as valid_target,
                AVG({self.source_column}) as target_mean,
                STDDEV({self.source_column}) as target_std,
                MIN({self.source_column}) as target_min,
                MAX({self.source_column}) as target_max
            FROM analytics_wide
        """).iloc[0]
        
        total_records = int(coverage_stats['total_records'])
        valid_target = int(coverage_stats['valid_target'])
        target_coverage = (valid_target / total_records) * 100 if total_records > 0 else 0
        
        print(f"    Cobertura target: {valid_target:,}/{total_records:,} ({target_coverage:.1f}%)")
        print(f"    Estatisticas: media={coverage_stats['target_mean']:.1f}, std={coverage_stats['target_std']:.1f}")
        print(f"    Range: [{coverage_stats['target_min']:.1f}, {coverage_stats['target_max']:.1f}]")
        
        if valid_target < 50:
            print(f"    [WARN] Muito poucos dados validos ({valid_target}<50)")
            print("      Pode comprometer poder estatistico dos modelos ML")

        if target_coverage < 20:
            print(f"    [WARN] Cobertura baixa ({target_coverage:.1f}%<20%)")
            print("      Risco de vies de selecao em predicoes")
        
        # 3. Consistência Temporal
        print("  [3/4] Consistencia temporal")
        temporal_stats = self.conn_manager.execute_sql("""
            SELECT
                COUNT(DISTINCT year) as unique_years,
                MIN(year) as min_year,
                MAX(year) as max_year,
                COUNT(DISTINCT country_code) as unique_countries
            FROM analytics_wide
        """).iloc[0]
        
        years_span = int(temporal_stats['max_year']) - int(temporal_stats['min_year']) + 1
        actual_years = int(temporal_stats['unique_years'])
        temporal_completeness = (actual_years / years_span) * 100
        
        print(f"    Span temporal: {temporal_stats['min_year']}-{temporal_stats['max_year']} ({years_span} anos)")
        print(f"    Completude temporal: {actual_years}/{years_span} anos ({temporal_completeness:.1f}%)")

        if temporal_completeness < 80:
            print("    [WARN] Gaps temporais significativos podem afetar validacao walk-forward")
        
        # 4. Cobertura Geográfica
        print("  [4/4] Representatividade geografica")
        unique_countries = int(temporal_stats['unique_countries'])
        obs_per_country = total_records / unique_countries if unique_countries > 0 else 0
        
        print(f"    Paises unicos: {unique_countries}")
        print(f"    Observacoes medias/pais: {obs_per_country:.1f}")

        if unique_countries < 10:
            print("    [WARN] Poucos paises podem limitar generalizacao geografica")
        
        # 5. Validação de Colunas Obrigatórias
        required_cols = ['country_code', 'year']
        missing_cols = []
        
        for col in required_cols:
            col_exists = self.conn_manager.execute_scalar(f"""
                SELECT COUNT(*) > 0
                FROM information_schema.columns
                WHERE table_name = 'analytics_wide'
                AND column_name = '{col}'
            """)
            
            if not col_exists:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(
                f"Colunas obrigatórias ausentes: {missing_cols}. "
                f"Schema incompatível com análise temporal."
            )
        
        print("  Validacao concluida")

    def create_target_implementation(self, data: Any) -> None:
        """
        Constrói variável target via transformação SQL com fundamentação educacional.
        
        Args:
            data: Ignorado - transformação executada diretamente no Data Warehouse
            
        Returns:
            None: Target adicionado como coluna persistent na tabela principal
            
        Transformação aplicada:
            Dropout Rate = 100 - Completion Rate
            
        Justificativa educacional:
            A taxa de abandono (dropout rate) é métricamente mais interpretável
            que taxa de conclusão para análise de políticas educacionais:
            
            1. Orientação por problema: Valores altos indicam necessidade de intervenção
            2. Linearidade: Relação direta com fatores socioeconômicos adversos
            3. Comparabilidade: Padrão internacional em literatura educacional
               (UNESCO, 2018; World Bank Education Statistics)
        
        Robustez:
            - Preserva NULLs originais (não imputa dados faltantes)
            - Mantém range [0,100] para interpretabilidade
            - Utiliza CASE statement para tratamento explícito de edge cases
        
        """
        print(f"Construindo target: {self.source_column} -> {self.target_column}")
        print("  Dropout Rate = 100 - Completion Rate")
        self.conn_manager.execute_sql_no_return(f"""
            ALTER TABLE analytics_wide
            ADD COLUMN IF NOT EXISTS {self.target_column} DOUBLE
        """)
        
        transformation_query = f"""
            UPDATE analytics_wide
            SET {self.target_column} =
                CASE
                    WHEN {self.source_column} IS NULL THEN NULL
                    WHEN {self.source_column} < 0 THEN NULL     -- Dados inválidos
                    WHEN {self.source_column} > 100 THEN NULL   -- Dados inválidos
                    ELSE 100 - {self.source_column}
                END
        """
        
        self.conn_manager.execute_sql_no_return(transformation_query)
        
        # Validação post-transformação
        validation_stats = self.conn_manager.execute_sql(f"""
            SELECT
                COUNT(*) as total_records,
                COUNT({self.target_column}) as valid_targets,
                AVG({self.target_column}) as mean_dropout,
                STDDEV({self.target_column}) as std_dropout,
                MIN({self.target_column}) as min_dropout,
                MAX({self.target_column}) as max_dropout
            FROM analytics_wide
        """).iloc[0]
        
        valid_targets = int(validation_stats['valid_targets'])
        total_records = int(validation_stats['total_records'])
        success_rate = (valid_targets / total_records) * 100 if total_records > 0 else 0
        
        print(f"  Registros validos: {valid_targets:,}/{total_records:,} ({success_rate:.1f}%)")
        print(f"  Dropout rate medio: {validation_stats['mean_dropout']:.1f}% +/- {validation_stats['std_dropout']:.1f}%")
        print(f"  Range: [{validation_stats['min_dropout']:.1f}%, {validation_stats['max_dropout']:.1f}%]")

        # Retorna dados atualizados do banco para uso no pipeline
        return self.conn_manager.execute_sql("SELECT * FROM analytics_wide")
    
    def _compute_target_statistics(self, data: Any) -> Dict[str, float]:
        """
        Computa estatísticas descritivas completas da variável target via SQL otimizado.
        
        Args:
            data: Ignorado - análise executada diretamente no Data Warehouse
            
        Returns:
            Dict contendo estatísticas descritivas: momentos centrais,
            quartis, medidas de assimetria e adequação para modelagem ML
            
        Estatísticas calculadas:
            1. Momentos: média, desvio padrão, assimetria (skewness)
            2. Range: mínimo, máximo, amplitude
            3. Quartis: Q1, mediana, Q3 para análise de distribuição
            4. Completude: contagem válida vs missing para análise de qualidade
            
        Otimização:
            Single query com múltiplas agregações para minimizar roundtrips
            ao banco de dados (1 query vs 6+ queries individuais).
        """
        comprehensive_stats_query = f"""
            WITH target_stats AS (
                SELECT
                    COUNT(*) as total_records,
                    COUNT({self.target_column}) as valid_count,
                    COUNT(*) - COUNT({self.target_column}) as missing_count,
                    AVG({self.target_column}) as mean_val,
                    STDDEV({self.target_column}) as std_val,
                    MIN({self.target_column}) as min_val,
                    MAX({self.target_column}) as max_val,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {self.target_column}) as q1,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {self.target_column}) as median,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {self.target_column}) as q3
                FROM analytics_wide
                WHERE {self.target_column} IS NOT NULL
            )
            SELECT *,
                   (max_val - min_val) as range_val,
                   (q3 - q1) as iqr_val
            FROM target_stats
        """
        
        stats_result = self.conn_manager.execute_sql(comprehensive_stats_query).iloc[0]
        
        # Estruturação dos resultados
        statistics = {
            # Contagens e completude
            'total_records': int(stats_result.get('total_records', 0)),
            'valid_count': int(stats_result.get('valid_count', 0)),
            'missing_count': int(stats_result.get('missing_count', 0)),
            
            # Momentos centrais
            'mean': float(stats_result.get('mean_val', 0) or 0),
            'std': float(stats_result.get('std_val', 0) or 0),
            'variance': float((stats_result.get('std_val', 0) or 0) ** 2),
            
            # Range e extremos
            'min': float(stats_result.get('min_val', 0) or 0),
            'max': float(stats_result.get('max_val', 0) or 0),
            'range': float(stats_result.get('range_val', 0) or 0),
            
            # Quartis e medidas de posição
            'q1': float(stats_result.get('q1', 0) or 0),
            'median': float(stats_result.get('median', 0) or 0),
            'q3': float(stats_result.get('q3', 0) or 0),
            'iqr': float(stats_result.get('iqr_val', 0) or 0),
        }
        
        # Métricas derivadas para análise ML
        if statistics['valid_count'] > 0:
            statistics['completeness_rate'] = statistics['valid_count'] / statistics['total_records']
            
            # Coeficiente de variação (normalização da variabilidade)
            if statistics['mean'] != 0:
                statistics['coefficient_variation'] = statistics['std'] / abs(statistics['mean'])
            else:
                statistics['coefficient_variation'] = float('inf')
        else:
            statistics['completeness_rate'] = 0.0
            statistics['coefficient_variation'] = float('nan')
        
        return statistics
    
    def _validate_temporal_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Valida a estrutura de folds temporais.
        
        Args:
            data: Ignorado - análise executada via SQL no Data Warehouse
            folds: Lista de configurações de folds para validação walk-forward
            
        Validações implementadas:
            1. Adequação estatística: Mínimo 30 observações por fold (CLT)
            2. Representatividade geográfica: Cobertura de países por fold
            3. Consistência temporal: Verificação de gaps anti-leakage
            4. Balanceamento: Distribuição equilibrada entre treino/validação/teste
            
        Critérios:
            - Treino: Mínimo 30 obs (regra prática CLT)
            - Validação: Mínimo 15 obs (poder estatístico básico)
            - Teste: Mínimo 10 obs (avaliação out-of-sample mínima)
            - Geographic coverage: >50% países em cada fold para generalização
            
        Validação cruzada temporal (Bergmeir & Benítez, 2012) requer
            estrutura específica para evitar data leakage em séries temporais.
        """
        print("Validando folds temporais")
        
        for i, fold in enumerate(folds):
            fold_id = fold['fold_id']
            print(f"\n  Fold {fold_id}:")
            
            # Query otimizada para estatísticas completas do fold
            fold_stats_query = f"""
                WITH fold_analysis AS (
                    SELECT
                        'train' as split_type,
                        COUNT(*) as obs_count,
                        COUNT(DISTINCT country_code) as country_count,
                        COUNT({self.target_column}) as valid_targets,
                        AVG({self.target_column}) as target_mean
                    FROM analytics_wide
                    WHERE year >= {fold['train_start']} AND year <= {fold['train_end']}
                      AND NOT (year >= {fold.get('train_gap_start', fold['train_end'])}
                               AND year <= {fold.get('train_gap_end', fold['train_end'])})
                    
                    UNION ALL
                    
                    SELECT
                        'val' as split_type,
                        COUNT(*) as obs_count,
                        COUNT(DISTINCT country_code) as country_count,
                        COUNT({self.target_column}) as valid_targets,
                        AVG({self.target_column}) as target_mean
                    FROM analytics_wide
                    WHERE year >= {fold['val_start']} AND year <= {fold['val_end']}
                    
                    UNION ALL
                    
                    SELECT
                        'test' as split_type,
                        COUNT(*) as obs_count,
                        COUNT(DISTINCT country_code) as country_count,
                        COUNT({self.target_column}) as valid_targets,
                        AVG({self.target_column}) as target_mean
                    FROM analytics_wide
                    WHERE year >= {fold['test_start']} AND year <= {fold['test_end']}
                      AND NOT (year >= {fold.get('val_gap_start', fold['test_start'])}
                               AND year <= {fold.get('val_gap_end', fold['test_start'])})
                )
                SELECT * FROM fold_analysis ORDER BY
                    CASE split_type
                        WHEN 'train' THEN 1
                        WHEN 'val' THEN 2
                        WHEN 'test' THEN 3
                    END
            """
            
            fold_results = self.conn_manager.execute_sql(fold_stats_query)
            
            # Extração e validação por split
            validation_warnings = []
            
            for _, row in fold_results.iterrows():
                split_type = row['split_type']
                obs_count = int(row['obs_count'])
                country_count = int(row['country_count'])
                valid_targets = int(row['valid_targets'] or 0)
                target_mean = float(row['target_mean'] or 0)
                
                # Armazenar no fold para uso posterior
                fold[f'{split_type}_count'] = obs_count
                fold[f'{split_type}_countries'] = country_count
                fold[f'{split_type}_valid_targets'] = valid_targets
                fold[f'{split_type}_target_mean'] = target_mean
                
                min_obs_required = {'train': 30, 'val': 15, 'test': 10}[split_type]

                print(f"    {split_type.upper()}: {obs_count:,} obs, {country_count} paises, "
                      f"{valid_targets} targets validos (media={target_mean:.1f}%)")
                
                if obs_count < min_obs_required:
                    warning = f"{split_type}: Poucos dados ({obs_count}<{min_obs_required}) - poder estatístico limitado"
                    validation_warnings.append(warning)
                
                # Critério 2: Representatividade geográfica
                total_countries = self.conn_manager.execute_scalar(
                    "SELECT COUNT(DISTINCT country_code) FROM analytics_wide"
                )
                geographic_coverage = (country_count / total_countries) * 100 if total_countries > 0 else 0
                
                if geographic_coverage < 50:
                    warning = f"{split_type}: Baixa cobertura geográfica ({geographic_coverage:.1f}%<50%)"
                    validation_warnings.append(warning)
                
                # Critério 3: Completude de targets
                target_completeness = (valid_targets / obs_count) * 100 if obs_count > 0 else 0
                if target_completeness < 70:
                    warning = f"{split_type}: Baixa completude de targets ({target_completeness:.1f}%<70%)"
                    validation_warnings.append(warning)
            
            # Validação de Consistência Temporal
            train_end = fold['train_end']
            val_start = fold['val_start']
            val_end = fold['val_end']
            test_start = fold['test_start']
            
            train_val_gap = val_start - train_end - 1
            val_test_gap = test_start - val_end - 1
            
            print(f"    Gaps: Treino->Val: {train_val_gap} anos, Val->Teste: {val_test_gap} anos")
            
            MIN_GAP = 2
            if train_val_gap < MIN_GAP:
                validation_warnings.append(
                    f"Gap treino-validação insuficiente ({train_val_gap}<{MIN_GAP} anos pulados)"
                )
            
            if val_test_gap < MIN_GAP:
                validation_warnings.append(
                    f"Gap validação-teste insuficiente ({val_test_gap}<{MIN_GAP} anos pulados)"
                )
            
            if validation_warnings:
                for warning in validation_warnings:
                    print(f"    [WARN] {warning}")
            else:
                print("    Fold ok")
    
    def save_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Cria temporal views ao invés de salvar arquivos.
        
        Args:
            data: Ignorado (usa SQL direto)
            folds: Lista de folds
        """
        print("\nCriando temporal views DuckDB")
        
        for fold in folds:
            fold_id = fold['fold_id']
            fold_dir = f"{self.prep_dir}/folds/fold_{fold_id}"
            os.makedirs(fold_dir, exist_ok=True)
            
            print(f"  Criando temporal views para fold {fold_id}...")
            
            try:
                train_view_query = f"""
                    CREATE OR REPLACE VIEW vw_fold_{fold_id}_train AS
                    SELECT * FROM vw_selected_features 
                    WHERE year >= {fold['train_start']} AND year <= {fold['train_end']}
                      AND NOT (year >= {fold['train_gap_start']} AND year <= {fold['train_gap_end']})
                    ORDER BY country_code, year
                """
                self.conn_manager.execute_sql_no_return(train_view_query)
                
                val_view_query = f"""
                    CREATE OR REPLACE VIEW vw_fold_{fold_id}_val AS
                    SELECT * FROM vw_selected_features 
                    WHERE year >= {fold['val_start']} AND year <= {fold['val_end']}
                    ORDER BY country_code, year
                """
                self.conn_manager.execute_sql_no_return(val_view_query)
                
                test_view_query = f"""
                    CREATE OR REPLACE VIEW vw_fold_{fold_id}_test AS
                    SELECT * FROM vw_selected_features 
                    WHERE year >= {fold['test_start']} AND year <= {fold['test_end']}
                      AND NOT (year >= {fold['val_gap_start']} AND year <= {fold['val_gap_end']})
                    ORDER BY country_code, year
                """
                self.conn_manager.execute_sql_no_return(test_view_query)
                
                print(f"    Views criadas: vw_fold_{fold_id}_{{train,val,test}}")
                
                train_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM vw_fold_{fold_id}_train")
                val_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM vw_fold_{fold_id}_val")
                test_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM vw_fold_{fold_id}_test")
                
                print(f"    Verificacao: Train={train_count}, Val={val_count}, Test={test_count}")
                
            except SQLProcessingError as e:
                print(f"    [ERROR] Criacao de views falhou para fold {fold_id}: {e}")
                raise
            
            fold_metadata = {
                **fold,
                'storage_method': 'duckdb_temporal_views',
                'view_names': {
                    'train': f'vw_fold_{fold_id}_train',
                    'val': f'vw_fold_{fold_id}_val',
                    'test': f'vw_fold_{fold_id}_test'
                }
            }
            self.save_fold_metadata(fold_metadata, fold_dir)
        
        try:
            master_view_query = """
                CREATE OR REPLACE VIEW vw_master_data AS
                SELECT * FROM analytics_wide 
                ORDER BY country_code, year
            """
            self.conn_manager.execute_sql_no_return(master_view_query)
            print(f"  Master view criada: vw_master_data")
            
        except SQLProcessingError as e:
            print(f"  [ERROR] Criacao de master view falhou: {e}")
            raise
        
        total_obs = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
        total_countries = self.conn_manager.execute_scalar("SELECT COUNT(DISTINCT country_code) FROM analytics_wide")
        min_year = self.conn_manager.execute_scalar("SELECT MIN(year) FROM analytics_wide")
        max_year = self.conn_manager.execute_scalar("SELECT MAX(year) FROM analytics_wide")
        
        self.save_master_config(folds, total_obs, total_countries, (int(min_year), int(max_year)))
        
        print(f"  DuckDB: Views temporais criadas, zero file I/O")
    
    def discover_numeric_columns(self, data: Any) -> List[str]:
        """
        Identifica colunas numéricas consultando o catálogo do engine.

        Args:
            data: Ignorado - análise executada via SQL metadata queries

        Returns:
            Lista de nomes de colunas numéricas

        Consulta a information_schema em vez de varrer os dados, o que é a
        forma nativa de um engine SQL responder essa pergunta.

        Limitações:
            - Não detecta variáveis categóricas numéricas (e.g., códigos país)
            - Ignora features derivadas não persistidas no schema
            - Assume que todos os tipos numéricos são apropriados para ML
        """
        numeric_columns_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'analytics_wide'
            AND data_type IN ('DOUBLE', 'INTEGER', 'FLOAT', 'DECIMAL', 'NUMERIC')
            ORDER BY column_name
        """
        result = self.conn_manager.execute_sql(numeric_columns_query)
        return result['column_name'].tolist()
    
    def compute_feature_correlations(self, data: Any,
                                    features: List[str]) -> Dict[str, float]:
        """
        Computa correlacoes de Pearson feature-target sobre dados de treino.

        Args:
            data: DataFrame pandas filtrado ao periodo de treino
            features: Lista de features candidatas para analise de correlacao

        Returns:
            Dicionario {feature_name: absolute_correlation} para ranking
        """
        print(f"Analisando correlacao de {len(features)} features com target")

        target_col = self.target_column
        correlations = {}
        failed_features = []

        df = data[features + [target_col]].dropna(subset=[target_col])
        print(f"  {len(df):,} observacoes validas")

        for feat in features:
            try:
                if feat not in df.columns:
                    correlations[feat] = 0.0
                    continue
                corr = df[feat].corr(df[target_col])
                if pd.isna(corr):
                    correlations[feat] = 0.0
                else:
                    correlations[feat] = abs(float(corr))
            except Exception as e:
                print(f"  [ERROR] Correlacao para {feat}: {e}")
                correlations[feat] = 0.0
                failed_features.append(feat)

        valid_correlations = [r for r in correlations.values() if r > 0]

        if valid_correlations:
            avg_correlation = sum(valid_correlations) / len(valid_correlations)
            max_correlation = max(valid_correlations)
            print(f"  Correlacao media: {avg_correlation:.3f}, maxima: {max_correlation:.3f}")

        if failed_features:
            print(f"  Features com falha: {len(failed_features)}")

        return correlations
    
    def apply_collinearity_filter(self, data: Any, features: List[str],
                                   threshold: float = 0.8) -> List[str]:
        """
        Remove multicolinearidade via filtragem greedy de correlacao pairwise.

        Args:
            data: DataFrame pandas filtrado ao periodo de treino
            features: Lista de features candidatas para analise
            threshold: Limiar de correlacao pairwise (padrao 0.8)

        Returns:
            Lista filtrada de features com multicolinearidade reduzida
        """
        if len(features) < 2:
            print("  Menos de 2 features - colinearidade desnecessaria")
            return features

        print(f"Filtrando colinearidade: {len(features)} features, threshold={threshold}")

        try:
            corr_data = data[features].dropna()

            print(f"  {len(corr_data):,} observacoes validas pos-dropna")

            if len(corr_data) > 10:
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
                raise ValueError(
                    f"Collinearity filtering needs more than 10 complete rows; "
                    f"got {len(corr_data)}. Returning an arbitrary subset would "
                    f"give this paradigm a different feature set from the "
                    f"others, and the comparison assumes they share one."
                )

        except Exception as e:
            print(f"[ERROR] Collinearity filtering failed: {e}")
            raise
    
    def prepare_features(self, data: Any, selected_features: List[str]) -> None:
        """
        Constrói view final com features selecionadas e transformações.
        
        Args:
            data: Ignorado - processamento executado via SQL no Data Warehouse
            selected_features: Features pós-filtragem de colinearidade para transformação
            
        Returns:
            None: View vw_selected_features criada persistentemente no banco
            
        Engenharia de Features Científica:
            Aplica symmetric log transform às top-5 features mais
            correlacionadas para normalização de distribuições assimétricas:

            T(x) = sign(x) * ln(|x| + 1)

            Preserva zeros e funciona com valores negativos, adequada para
            dados educacionais que podem incluir déficits/declínios.

            Top-5 limite baseado em curse of dimensionality (Bellman, 1961).

        Schema da view resultante:
            - Metadados: country_code, year, {target_column}
            - Features originais: selected_features (pós-filtragem de colinearidade)
            - Features transformadas: {feature}_log_transform para top-5
        """
        print(f"Preparando view final com {len(selected_features)} features")
        
        # Critério: Limitar a top-5 features mais promissoras
        features_to_transform = selected_features[:5]
        transformed_features_sql = []
        
        print(f"  Transformando {len(features_to_transform)} features (symmetric log):")
        
        for feat in features_to_transform:
            # Tranformação, log simétrico: sign(x) * ln(|x| + 1)
            # Prefixo a. necessário pois a query usa self-join com alias
            transformation_sql = f"""
                CASE
                    WHEN a.{feat} IS NULL THEN NULL
                    WHEN a.{feat} = 0 THEN 0.0
                    ELSE SIGN(a.{feat}) * LN(ABS(a.{feat}) + 1)
                END AS {feat}_log_transform
            """
            transformed_features_sql.append(transformation_sql)
            print(f"    {feat} -> {feat}_log_transform")

        # Construção da Query de View
        all_features_sql = selected_features.copy()

        if transformed_features_sql:
            print(f"  {len(transformed_features_sql)} log transforms aplicadas")
        
        # Query SQL para view final estruturada
        # Lag temporal via self-join (valor de exatamente N anos atrás),
        # não LAG() posicional que assume dados sem gaps anuais.
        feature_view_query = f"""
            CREATE OR REPLACE VIEW vw_selected_features AS
            SELECT
                -- Metadados temporais e geográficos (essenciais para ML temporal)
                a.country_code,
                a.year,
                a.{self.target_column},
                -- Lags do target (2 e 3 anos) via join temporal sem vazamento
                lag2.{self.target_column} AS dropout_rate_lag_2,
                lag3.{self.target_column} AS dropout_rate_lag_3,

                -- Features originais pós-filtragem de colinearidade
                {', '.join(['a.' + f for f in all_features_sql])}

                {', -- Features transformadas (symmetric log)' if transformed_features_sql else ''}
                {', '.join(transformed_features_sql) if transformed_features_sql else ''}

            FROM analytics_wide a
            LEFT JOIN analytics_wide lag2
                ON a.country_code = lag2.country_code AND a.year = lag2.year + 2
            LEFT JOIN analytics_wide lag3
                ON a.country_code = lag3.country_code AND a.year = lag3.year + 3
            WHERE a.{self.target_column} IS NOT NULL  -- Filtro essencial para ML supervisionado
            ORDER BY a.country_code, a.year           -- Preserva ordem temporal para walk-forward
        """
        
        try:
            self.conn_manager.execute_sql_no_return(feature_view_query)
            
            # Validação e Relatório da View
            view_validation_query = f"""
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT country_code) as unique_countries,
                    MIN(year) as min_year,
                    MAX(year) as max_year,
                    AVG({self.target_column}) as avg_target
                FROM vw_selected_features
            """
            
            validation_result = self.conn_manager.execute_sql(view_validation_query).iloc[0]
            
            total_records = int(validation_result['total_records'])
            unique_countries = int(validation_result['unique_countries'])
            min_year = int(validation_result['min_year'])
            max_year = int(validation_result['max_year'])
            avg_target = float(validation_result['avg_target'])
            
            # Cálculo de dimensionalidade final
            original_features = len(selected_features)
            transformed_features = len(transformed_features_sql)
            total_features = original_features + transformed_features + 3  # +3 metadados
            
            print(f"  View vw_selected_features criada:")
            print(f"    {total_records:,} obs, {total_features} variaveis ({original_features} originais, "
                  f"{transformed_features} transformadas)")
            print(f"    {unique_countries} paises, {min_year}-{max_year}, target medio: {avg_target:.1f}%")
            
            # Análise de adequação para ML
            observations_per_feature = total_records / total_features if total_features > 0 else 0
            
            if observations_per_feature < 10:
                print("    [WARN] Poucos dados/feature - risco de overfitting")
            elif observations_per_feature > 50:
                print("    Boa relacao observacoes/features para ML")
            
        except SQLProcessingError as e:
            print(f"  [ERROR] Falha na criacao da view de features: {e}")
            raise RuntimeError(f"Impossível criar view de features: {e}")
        
        print("  Features prontas para modelagem ML")
        # Paradigma Data Warehouse: dados permanecem no banco para eficiência
        return None


def main():
    """
    Função principal para execução e teste do pipeline ML Data Warehouse.
    
    Executa pipeline completo de ML temporal seguindo metodologia:
    1. Setup e validação de ambiente
    2. Carregamento e validação de dados
    3. Construção de variável target
    4. Seleção de features com filtragem de colinearidade pairwise
    5. Criação de folds temporais walk-forward
    6. Preparação final para modelagem
    
    Adequado para:
        - Desenvolvimento e debugging do pipeline
        - Validação da metodologia
        - Benchmark de performance arquitetural
        - Testes de integração antes de produção
        
    Não adequado para:
        - Execução em produção (usar API específica)
        - Análises exploratórias (usar notebooks)
        - Comparação arquitetural (usar architectural_benchmark.py)
    """
    print("=" * 80)
    print("Pipeline ML DuckDB")
    print("=" * 80)

    try:
        setup = SqlEngineArchitectureML()
        results = setup.run_setup()
        
        success_flag = results.get('success', None)
        status_flag = results.get('status', None)
        is_success = (success_flag is True) or (isinstance(status_flag, str) and status_flag.lower() == 'success')
        if is_success:
            print("Pipeline ok")
            print(f"  Arquitetura: {results.get('architecture', 'N/A')}")
            print(f"  Features selecionadas: {results.get('features_selected', results.get('selected_features_count', 'N/A'))}")
            print(f"  Folds temporais: {results.get('folds_created', results.get('total_folds', 'N/A'))}")
            if isinstance(results.get('total_observations', None), (int, float)):
                print(f"  Observacoes processadas: {int(results.get('total_observations')):,}")

            if 'processing_time' in results and isinstance(results.get('processing_time'), (int, float)):
                print(f"  Tempo de processamento: {results['processing_time']:.2f}s")
        else:
            print("[ERROR] Pipeline falhou")
            if 'error' in results:
                print(f"  Erro: {results['error']}")

        print(f"\nResultados:")
        for key, value in results.items():
            if key not in ['status', 'error']:
                print(f"  {key}: {value}")

        return results

    except Exception as e:
        print(f"\n[ERROR] Pipeline falhou: {e}")
        print("  Verifique se DuckDB foi processado corretamente")
        print("  Execute sql_engine/processor.py antes deste script")
        return {'status': 'failed', 'error': str(e)}
    


if __name__ == "__main__":
    results = main()
    # A failed setup must not report success to the pipeline, which runs each
    # stage as a subprocess and reads its exit status.
    sys.exit(0 if isinstance(results, dict)
             and results.get('status') == 'success' else 1)
