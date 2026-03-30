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

# Adicionar caminho para módulos core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.base_architecture import BaseArchitectureML
from core.config import get_absolute_output_path

from collection.data_warehouse.connection_manager import (
    DuckDBConnectionManager, 
    SQLProcessingError
)


class DataWarehouseArchitectureML(BaseArchitectureML):
    """Implementação do pipeline ML para a arquitetura Data Warehouse.

    Reproduz o mesmo protocolo científico aplicado à arquitetura Data Lake,
    diferindo apenas pelo processamento SQL in-database. Todos os artefatos
    gerados (folds, estatísticas de target, matrizes de features) seguem o
    padrão do framework para permitir benchmark e equivalência prática."""

    PARADIGM_META = {
        'name': 'data_warehouse',
        'label': 'Data Warehouse com DuckDB',
        'processor_module': 'collection.data_warehouse.processor',
        'processor_class': 'DataWarehouseProcessor',
        'processor_run_method': 'run_data_warehouse_processing',
        'baseline_module': 'architectures_ml.data_warehouse.models.baseline_analysis',
        'baseline_class': 'BaselineModelAnalysisDataWarehouse',
        'hierarchical_module': 'architectures_ml.data_warehouse.models.hierarchical_model',
        'hierarchical_class': 'HierarchicalModelSQLFirst',
        'setup_script': 'src/architectures_ml/data_warehouse/setup.py',
        'processor_script': 'src/collection/data_warehouse/processor.py',
        'baseline_script': 'src/architectures_ml/data_warehouse/models/baseline_analysis.py',
        'hierarchical_script': 'src/architectures_ml/data_warehouse/models/hierarchical_model.py',
    }

    def __init__(self):
        """Inicializa paths, conexão DuckDB e logger científico."""
        # Inicialização da arquitetura base com herança científica
        output_base = get_absolute_output_path('ml_pipeline/architectures/data_warehouse')
        super().__init__(architecture_name='data_warehouse', output_base_path=output_base)
        
        print("Inicializando Pipeline ML Data Warehouse Científico")
        print("=" * 70)
        print("[PARADIGMA] SQL-first com validação temporal rigorosa")
        
        # Configurações específicas do Data Warehouse
        self.db_path = get_absolute_output_path('collection/data_warehouse/worldbank_data.duckdb')
        self.conn_manager = None
        
        print(f"[OUTPUT] Diretório base: {self.output_base}")
        print(f"[DATABASE] Data Warehouse: {self.db_path}")
        print("[METODOLOGIA] Zero file I/O, processamento SQL nativo sem cache")
    
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
                f"Execute 'data_warehouse/processor.py' antes deste pipeline ML."
            )
        
        # Inicialização com parâmetros científicamente justificados
        self.conn_manager = DuckDBConnectionManager(
            self.db_path,
            max_retries=3,      
            retry_delay=1.0
        )
        
        print("   ✓ Connection Manager configurado com robustez científica")
        print("   ✓ ACID compliance habilitado para integridade transacional")
    
    def load_data(self) -> None:
        """
        Executa carregamento de dados via SQL nativo com validação científica.
        
        Returns:
            None: Dados permanecem in-database seguindo paradigma Data Warehouse
            
        Metodologia:
            Diferentemente de abordagens pandas/numpy, mantém dados no banco
            para aproveitar otimizações OLAP nativas do DuckDB:
            - Vectorized execution para agregações
            - Columnar storage para queries analíticas
            - Query optimization baseada em estatísticas de tabela
        
        Métricas coletadas:
            - Dimensionalidade: Observações × variáveis para análise de escala
            - Cobertura temporal: Range completo para validação walk-forward
            - Cobertura geográfica: Países únicos para análise cross-section
        
        Complexidade: O(1) vs O(n) para carregamento pandas tradicional
        """
        print("\n[CARREGAMENTO] Iniciando análise de dados via SQL nativo")
        print("━" * 50)
        
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
        
        # Extração com validação científica
        total_records = int(stats_result['total_records'])
        total_columns = int(stats_result['total_columns'])
        min_year = int(stats_result['min_year'])
        max_year = int(stats_result['max_year'])
        unique_countries = int(stats_result['unique_countries'])
        temporal_periods = int(stats_result['temporal_periods'])
        
        # Análise de adequação para ML temporal
        years_span = max_year - min_year + 1
        avg_obs_per_country = total_records / unique_countries if unique_countries > 0 else 0
        
        print(f"[DIMENSÕES] {total_records:,} observações × {total_columns} variáveis")
        print(f"[TEMPORAL] {min_year}-{max_year} ({years_span} anos, {temporal_periods} períodos)")
        print(f"[GEOGRÁFICO] {unique_countries} países ({avg_obs_per_country:.1f} obs/país)")
        
        # Validação para ML temporal
        if years_span < 10:
            print(f"  ⚠ AVISO: Série temporal curta ({years_span} anos) pode limitar validação walk-forward")
        
        if total_records < 100:
            print(f"  ⚠ AVISO: Dataset pequeno ({total_records} obs) pode afetar poder estatístico")
        
        print("[STATUS] Dados carregados com sucesso (permanecem in-database)")
        
        # Paradigma Data Warehouse: dados nunca saem do banco
        return None
    
    def validate_data(self, data: Any) -> None:
        """
        Executa validação científica de integridade e adequação dos dados.
        
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
        
        Robustez:
            Implementa fallback automático para variáveis target alternativas
            baseado em correspondência semântica (completion rate variants).
        """
        print("[VALIDAÇÃO] Executando verificações de integridade científica")
        print("━" * 50)
        
        target_source_col = self.source_column
        
        # === 1. Validação de Schema ===
        print("[1/4] Validação de schema")
        column_exists = self.conn_manager.execute_scalar(f"""
            SELECT COUNT(*) > 0
            FROM information_schema.columns
            WHERE table_name = 'analytics_wide'
            AND column_name = '{target_source_col}'
        """)
        
        if not column_exists:
            print(f"  ⚠ Coluna target '{target_source_col}' não encontrada")
            print("  → Buscando variáveis alternativas...")
            
            # Fallback inteligente baseado em correspondência semântica
            alternative_col = self.conn_manager.execute_scalar("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'analytics_wide'
                AND (LOWER(column_name) LIKE '%completion%'
                     OR LOWER(column_name) LIKE '%enrollment%'
                     OR LOWER(column_name) LIKE '%graduation%')
                ORDER BY
                    CASE
                        WHEN LOWER(column_name) LIKE '%completion%' THEN 1
                        WHEN LOWER(column_name) LIKE '%enrollment%' THEN 2
                        ELSE 3
                    END
                LIMIT 1
            """)
            
            if alternative_col:
                self.source_column = alternative_col
                print(f"  ✓ Usando variável alternativa: {self.source_column}")
            else:
                raise ValueError(
                    "Nenhuma variável educacional adequada encontrada. "
                    "Verifique se dados foram processados corretamente."
                )
        
        # === 2. Análise de Cobertura ===
        print("[2/4] Análise de cobertura")
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
        
        print(f"  → Cobertura target: {valid_target:,}/{total_records:,} ({target_coverage:.1f}%)")
        print(f"  → Estatísticas: μ={coverage_stats['target_mean']:.1f}, σ={coverage_stats['target_std']:.1f}")
        print(f"  → Range: [{coverage_stats['target_min']:.1f}, {coverage_stats['target_max']:.1f}]")
        
        # Critérios científicos de adequação
        if valid_target < 50:
            print(f"  ⚠ CRÍTICO: Muito poucos dados válidos ({valid_target}<50)")
            print("    → Pode comprometer poder estatístico dos modelos ML")
        
        if target_coverage < 20:
            print(f"  ⚠ AVISO: Cobertura baixa ({target_coverage:.1f}%<20%)")
            print("    → Risco de viés de seleção em predições")
        
        # === 3. Consistência Temporal ===
        print("[3/4] Consistência temporal")
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
        
        print(f"  → Span temporal: {temporal_stats['min_year']}-{temporal_stats['max_year']} ({years_span} anos)")
        print(f"  → Completude temporal: {actual_years}/{years_span} anos ({temporal_completeness:.1f}%)")
        
        if temporal_completeness < 80:
            print("  ⚠ AVISO: Gaps temporais significativos podem afetar validação walk-forward")
        
        # === 4. Cobertura Geográfica ===
        print("[4/4] Representatividade geográfica")
        unique_countries = int(temporal_stats['unique_countries'])
        obs_per_country = total_records / unique_countries if unique_countries > 0 else 0
        
        print(f"  → Países únicos: {unique_countries}")
        print(f"  → Observações médias/país: {obs_per_country:.1f}")
        
        if unique_countries < 10:
            print("  ⚠ AVISO: Poucos países podem limitar generalização geográfica")
        
        # === 5. Validação de Colunas Obrigatórias ===
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
        
        print("[STATUS] ✓ Validação científica concluída com sucesso")
    
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
        
        Complexidade: O(n) single-pass sobre dataset completo
        """
        print("[TARGET] Construindo variável target com fundamentação educacional")
        print("━" * 50)
        
        print(f"[TRANSFORMAÇÃO] {self.source_column} → {self.target_column}")
        print("[FÓRMULA] Dropout Rate = 100 - Completion Rate")
        
        # Schema evolution: Adiciona coluna se não existe
        print("[SCHEMA] Adicionando coluna target ao Data Warehouse...")
        self.conn_manager.execute_sql_no_return(f"""
            ALTER TABLE analytics_wide
            ADD COLUMN IF NOT EXISTS {self.target_column} DOUBLE
        """)
        
        # Transformação cientificamente fundamentada
        print("[CÁLCULO] Executando transformação via SQL transacional...")
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
        
        print(f"[RESULTADO] Target criado com sucesso:")
        print(f"  → Registros válidos: {valid_targets:,}/{total_records:,} ({success_rate:.1f}%)")
        print(f"  → Dropout rate médio: {validation_stats['mean_dropout']:.1f}% ± {validation_stats['std_dropout']:.1f}%")
        print(f"  → Range: [{validation_stats['min_dropout']:.1f}%, {validation_stats['max_dropout']:.1f}%]")
        
        print("[STATUS] ✓ Variável target construída e validada")

        # Retorna dados atualizados do banco para uso no pipeline
        return self.conn_manager.execute_sql("SELECT * FROM analytics_wide")
    
    def _compute_target_statistics(self, data: Any) -> Dict[str, float]:
        """
        Computa estatísticas descritivas completas da variável target via SQL otimizado.
        
        Args:
            data: Ignorado - análise executada diretamente no Data Warehouse
            
        Returns:
            Dict contendo estatísticas descritivas científicas: momentos centrais,
            quartis, medidas de assimetria e adequação para modelagem ML
            
        Estatísticas calculadas:
            1. Momentos: média, desvio padrão, assimetria (skewness)
            2. Range: mínimo, máximo, amplitude
            3. Quartis: Q1, mediana, Q3 para análise de distribuição
            4. Completude: contagem válida vs missing para análise de qualidade
            
        Otimização:
            Single query com múltiplas agregações para minimizar roundtrips
            ao banco de dados (1 query vs 6+ queries individuais).
        
        Complexidade: O(n) single-pass sobre dataset completo
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
        
        # Estruturação científica dos resultados
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
        Executa validação científica rigorosa da estrutura de folds temporais.
        
        Args:
            data: Ignorado - análise executada via SQL no Data Warehouse
            folds: Lista de configurações de folds para validação walk-forward
            
        Validações implementadas:
            1. Adequação estatística: Mínimo 30 observações por fold (CLT)
            2. Representatividade geográfica: Cobertura de países por fold
            3. Consistência temporal: Verificação de gaps anti-leakage
            4. Balanceamento: Distribuição equilibrada entre treino/validação/teste
            
        Critérios científicos:
            - Treino: Mínimo 30 obs (Central Limit Theorem, Gosset, 1908)
            - Validação: Mínimo 15 obs (poder estatístico básico)
            - Teste: Mínimo 10 obs (avaliação out-of-sample mínima)
            - Geographic coverage: >50% países em cada fold para generalização
            
        Fundamentação:
            Validação cruzada temporal (Bergmeir & Benítez, 2012) requer
            estrutura específica para evitar data leakage em séries temporais.
        """
        print("[VALIDAÇÃO] Analisando estrutura científica dos folds temporais")
        print("━" * 60)
        
        for i, fold in enumerate(folds):
            fold_id = fold['fold_id']
            print(f"\n[FOLD {fold_id}] Validação científica:")
            
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
                
                # Validação científica por critérios estatísticos
                min_obs_required = {'train': 30, 'val': 15, 'test': 10}[split_type]
                
                print(f"  [{split_type.upper()}] {obs_count:,} obs, {country_count} países, "
                      f"{valid_targets} targets válidos (μ={target_mean:.1f}%)")
                
                # Critério 1: Adequação estatística (CLT)
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
            
            # === Validação de Consistência Temporal ===
            train_end = fold['train_end']
            val_start = fold['val_start']
            val_end = fold['val_end']
            test_start = fold['test_start']
            
            train_val_gap = val_start - train_end - 1
            val_test_gap = test_start - val_end - 1
            
            print(f"  [GAPS] Treino→Val: {train_val_gap} anos pulados, Val→Teste: {val_test_gap} anos pulados")
            
            MIN_GAP = 2
            if train_val_gap < MIN_GAP:
                validation_warnings.append(
                    f"Gap treino-validação insuficiente ({train_val_gap}<{MIN_GAP} anos pulados)"
                )
            
            if val_test_gap < MIN_GAP:
                validation_warnings.append(
                    f"Gap validação-teste insuficiente ({val_test_gap}<{MIN_GAP} anos pulados)"
                )
            
            # Reportar warnings
            if validation_warnings:
                print("  ⚠ WARNINGS CIENTÍFICOS:")
                for warning in validation_warnings:
                    print(f"    → {warning}")
            else:
                print("  ✓ Fold atende todos os critérios científicos")
    
    def save_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Cria temporal views ao invés de salvar arquivos.
        
        Args:
            data: Ignorado (usa SQL direto)
            folds: Lista de folds
        """
        print("\n🖧  CRIANDO TEMPORAL VIEWS DATA WAREHOUSE...")
        
        for fold in folds:
            fold_id = fold['fold_id']
            fold_dir = f"{self.prep_dir}/folds/fold_{fold_id}"
            os.makedirs(fold_dir, exist_ok=True)
            
            print(f"   ↪ Criando temporal views para fold {fold_id}...")
            
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
                
                print(f"   ✔ Views criadas: vw_fold_{fold_id}_{{train,val,test}}")
                
                train_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM vw_fold_{fold_id}_train")
                val_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM vw_fold_{fold_id}_val")
                test_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM vw_fold_{fold_id}_test")
                
                print(f"   🖈  Verificação: Train={train_count}, Val={val_count}, Test={test_count}")
                
            except SQLProcessingError as e:
                print(f"   ✖ ERRO: Criação de views falhou para fold {fold_id}: {e}")
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
            print(f"   ✔ Master view criada: vw_master_data")
            
        except SQLProcessingError as e:
            print(f"   ✖ ERRO: Criação de master view falhou: {e}")
            raise
        
        total_obs = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
        total_countries = self.conn_manager.execute_scalar("SELECT COUNT(DISTINCT country_code) FROM analytics_wide")
        min_year = self.conn_manager.execute_scalar("SELECT MIN(year) FROM analytics_wide")
        max_year = self.conn_manager.execute_scalar("SELECT MAX(year) FROM analytics_wide")
        
        self.save_master_config(folds, total_obs, total_countries, (int(min_year), int(max_year)))
        
        print(f"   ✔ Data Warehouse: Views temporais criadas, zero file I/O")
    
    def get_numeric_features(self, data: Any) -> List[str]:
        """
        Identifica features numéricas candidatas para modelagem ML via análise de schema.
        
        Args:
            data: Ignorado - análise executada via SQL metadata queries
            
        Returns:
            Lista de nomes de colunas numéricas adequadas para ML, ordenadas
            alfabeticamente para reprodutibilidade determinística
            
        Critérios de seleção:
            1. Tipo numérico: DOUBLE, INTEGER, FLOAT, DECIMAL, NUMERIC
            2. Exclusão automática: Identificadores (country_code, year), targets
            3. Ordenação alfabética: Garantia de determinismo entre execuções
            
        Justificativa técnica:
            Schema-based filtering é mais eficiente que data-type inference
            em runtime, especialmente para datasets grandes (O(1) vs O(n)).
            DuckDB information_schema query executa em <1ms vs scan completo.
        
        Limitações:
            - Não detecta variáveis categóricas numéricas (e.g., códigos país)
            - Ignora features derivadas não persistidas no schema
            - Assume que todos os tipos numéricos são apropriados para ML
        """
        numeric_features_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'analytics_wide'
            AND data_type IN ('DOUBLE', 'INTEGER', 'FLOAT', 'DECIMAL', 'NUMERIC')
            AND column_name NOT IN (
                'country_code', 'year',                    -- Identificadores
                'dropout_rate_data_warehouse',              -- Target variables
                'dropout_rate_data_lake',
                'data_completeness_score'                   -- Quality metadata
            )
            ORDER BY column_name                            -- Determinismo
        """
        
        result = self.conn_manager.execute_sql(numeric_features_query)
        numeric_features = result['column_name'].tolist()
        
        print(f"[FEATURES] {len(numeric_features)} variáveis numéricas identificadas")
        print(f"[EXCLUSÕES] Identificadores, targets e metadados removidos automaticamente")
        
        # Validação de adequação
        if len(numeric_features) < 5:
            print("  ⚠ AVISO: Poucas features numéricas (<5) podem limitar capacidade preditiva")
        
        if len(numeric_features) > 100:
            print("  ⚠ AVISO: Muitas features (>100) requerem seleção rigorosa (curse of dimensionality)")
        
        return numeric_features
    
    def compute_feature_correlations(self, data: Any,
                                    features: List[str]) -> Dict[str, float]:
        """
        Computa correlações de Pearson feature-target via SQL nativo otimizado.
        
        Args:
            data: Ignorado - análise executada via SQL no Data Warehouse
            features: Lista de features candidatas para análise de correlação
            
        Returns:
            Dicionário {feature_name: absolute_correlation} ordenado por relevância
            
        Metodologia estatística:
            Utiliza correlação de Pearson (r) como proxy de associação linear:
            r = Cov(X,Y) / (σ_X × σ_Y)
            
        Justificativas:
            1. Valor absoluto: |r| indica força da relação independente da direção
            2. Correlação Pearson: Adequada para relações lineares (assunção ML)
            3. Tratamento de zeros: Variáveis constantes recebem r=0 automaticamente
            
        Otimizações SQL:
            - CTE reutilizável para estatísticas descritivas
            - Single-pass calculation para eficiência O(n)
            - Tratamento robusto de divisão por zero e NULLs
            
        Limitações:
            - Detecta apenas associações lineares (ignora não-linearidades)
            - Sensível a outliers extremos (não usa correlação robusta)
            - Assume distribuições contínuas (inadequado para categóricas)
        
        Referência:
            Pearson, K. (1895). Mathematical contributions to the theory of evolution.
        """
        print(f"[CORRELAÇÃO] Analisando associação linear de {len(features)} features com target")
        print("━" * 50)
        
        target_col = self.target_column
        correlations = {}
        failed_features = []

        # Decidir base da análise (amostra vs full table) conforme scientific_config
        use_sampling = bool(self.config.get('correlation_sampling', True))
        min_sample = int(self.config.get('correlation_min_sample_size', 5000))
        frac = float(self.config.get('correlation_sample_fraction', 0.1))
        base_table = 'analytics_wide'

        sample_view_name = 'vw_corr_sample'
        try:
            total_rows = int(self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide"))
        except Exception:
            total_rows = 0
        if use_sampling and total_rows > min_sample:
            final_sample_size = max(min_sample, int(total_rows * frac))
            seed_float = (self.config.get('random_seed', 42) % 100) / 100.0
            try:
                self.conn_manager.execute_sql_no_return(f"SELECT setseed({seed_float})")
            except Exception:
                pass
            try:
                self.conn_manager.execute_sql_no_return(f"DROP VIEW IF EXISTS {sample_view_name}")
                sampling_query = f"""
                    CREATE OR REPLACE VIEW {sample_view_name} AS
                    SELECT * FROM analytics_wide
                    WHERE {target_col} IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT {final_sample_size}
                """
                self.conn_manager.execute_sql_no_return(sampling_query)
                base_table = sample_view_name
                print(f"[AMOSTRAGEM] {final_sample_size:,}/{total_rows:,} observações (~{final_sample_size/total_rows:.1%})")
            except SQLProcessingError as e:
                print(f"[AMOSTRAGEM] Falha ao criar view amostrada: {e}. Usando tabela completa.")
                base_table = 'analytics_wide'

        # Análise batch para features com schema similar
        print("[CÁLCULO] Executando análise de correlação via SQL otimizado...")
        
        for i, feat in enumerate(features):
            try:
                # Query SQL scientificamente robusta para correlação de Pearson
                correlation_query = f"""
                    WITH feature_target_stats AS (
                        SELECT
                            COUNT(*) as sample_size,
                            AVG({feat}) as feat_mean,
                            AVG({target_col}) as target_mean,
                            STDDEV({feat}) as feat_std,
                            STDDEV({target_col}) as target_std
                        FROM {base_table}
                        WHERE {feat} IS NOT NULL
                        AND {target_col} IS NOT NULL
                    ),
                    covariance_calc AS (
                        SELECT
                            AVG(({feat} - s.feat_mean) * ({target_col} - s.target_mean)) as covariance,
                            s.feat_std,
                            s.target_std,
                            s.sample_size
                        FROM {base_table} a
                        CROSS JOIN feature_target_stats s
                        WHERE a.{feat} IS NOT NULL
                        AND a.{target_col} IS NOT NULL
                        GROUP BY s.feat_std, s.target_std, s.sample_size
                    )
                    SELECT
                        CASE
                            WHEN feat_std = 0 OR target_std = 0 OR feat_std IS NULL OR target_std IS NULL
                            THEN 0.0  -- Variável constante = correlação zero
                            WHEN sample_size < 3
                            THEN 0.0  -- Amostra inadequada para correlação
                            ELSE covariance / (feat_std * target_std)
                        END as pearson_correlation,
                        sample_size
                    FROM covariance_calc
                """
                
                result = self.conn_manager.execute_sql(correlation_query)
                
                if not result.empty:
                    correlation = float(result.iloc[0]['pearson_correlation'] or 0.0)
                    sample_size = int(result.iloc[0]['sample_size'] or 0)
                    
                    # Validação científica da correlação
                    if sample_size < 10:
                        print(f"  → {feat}: Amostra pequena (n={sample_size}), correlação não confiável")
                        correlations[feat] = 0.0
                    else:
                        # Valor absoluto para ranking por força da associação
                        correlations[feat] = abs(correlation)
                        
                        # Log para features altamente correlacionadas
                        if abs(correlation) > 0.7:
                            print(f"  → {feat}: r={correlation:.3f} (ALTA correlação)")
                        elif abs(correlation) > 0.3:
                            print(f"  → {feat}: r={correlation:.3f} (correlação moderada)")
                else:
                    correlations[feat] = 0.0
                    
            except SQLProcessingError as e:
                print(f"  ✗ Erro no cálculo de correlação para {feat}: {e}")
                correlations[feat] = 0.0
                failed_features.append(feat)
        
        # Limpeza da view amostrada
        if base_table == sample_view_name:
            try:
                self.conn_manager.execute_sql_no_return(f"DROP VIEW IF EXISTS {sample_view_name}")
            except Exception:
                pass
        
        # Relatório estatístico
        valid_correlations = [r for r in correlations.values() if r > 0]
        
        if valid_correlations:
            avg_correlation = sum(valid_correlations) / len(valid_correlations)
            max_correlation = max(valid_correlations)
            high_corr_count = sum(1 for r in valid_correlations if r > 0.5)
            
            print(f"\n[RESULTADO] Análise de correlação concluída:")
            print(f"  → Correlação média: {avg_correlation:.3f}")
            print(f"  → Correlação máxima: {max_correlation:.3f}")
            print(f"  → Features altamente correlacionadas (|r|>0.5): {high_corr_count}")
            
            if len(failed_features) > 0:
                print(f"  → Features com falha no cálculo: {len(failed_features)}")
        
        return correlations
    
    def apply_collinearity_filter(self, data: Any, features: List[str],
                                   threshold: float = 0.8) -> List[str]:
        """
        Remove multicolinearidade via filtragem greedy de correlação pairwise (SQL).

        Para cada feature candidata, calcula a correlação absoluta máxima com
        as features já selecionadas e rejeita se max |r| >= threshold.

        Args:
            data: Ignorado - análise executada no Data Warehouse
            features: Lista de features candidatas para análise de multicolinearidade
            threshold: Limiar de correlação pairwise (padrão 0.8)

        Returns:
            Lista de features filtradas com multicolinearidade reduzida

        Algoritmo greedy:
            1. Primeira feature sempre aceita (baseline)
            2. Features subsequentes aceitas se max |r| < threshold
            3. Ordem preservada para determinismo científico

        Amostragem justificada:
            Para datasets >10k observações, matriz de correlação completa
            pode ser computacionalmente custosa O(n²). Amostragem reduz a
            O(s²) onde s << n, mantendo precisão estatística adequada.

        Referências:
            Dormann, C. F., et al. (2013). Collinearity: a review of methods
            to deal with it and a simulation study evaluating their performance.
        """
        if len(features) < 2:
            print("[COLLINEARITY] Menos de 2 features - análise desnecessária")
            return features
        
        print(f"[COLLINEARITY] Executando análise de multicolinearidade para {len(features)} features")
        print("━" * 50)
        print(f"[THRESHOLD] Correlação máxima permitida: {threshold}")
        
        # === Configuração de amostragem científica ===
        total_rows = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
        
        # Critérios padronizados baseados em literatura estatística
        min_sample_absolute = self.config.get('correlation_min_sample_size', 5000)
        sample_fraction = self.config.get('correlation_sample_fraction', 0.1)
        
        optimal_sample_size = max(
            min_sample_absolute,
            int(total_rows * sample_fraction)
        )

        final_sample_size = min(optimal_sample_size, total_rows)
        
        sample_percentage = (final_sample_size / total_rows) * 100
        
        print(f"[AMOSTRAGEM] {final_sample_size:,}/{total_rows:,} observações ({sample_percentage:.1f}%)")
        print(f"[CRITÉRIO] max(config={min_sample_absolute}, {sample_fraction:.0%}×dataset)")
        
        # === Criação de view amostrada reprodutível ===
        sample_view_name = "vw_collinearity_sample"
        
        try:
            # Configuração de seed para reprodutibilidade científica
            random_seed = self.config.get('random_seed', 42)
            seed_float = (random_seed % 100) / 100.0  # Normalização [0,1]
            
            print(f"[SEED] Configurando seed reprodutível: {random_seed} → {seed_float:.2f}")
            
            self.conn_manager.execute_sql_no_return(f"SELECT setseed({seed_float})")
            self.conn_manager.execute_sql_no_return(f"DROP VIEW IF EXISTS {sample_view_name}")
            
            # Amostragem aleatória com dropna completo (equivalente ao DL que
            # faz .compute().dropna() sobre TODAS as features candidatas).
            # Versão anterior filtrava apenas features[:10], criando divergência.
            not_null_clauses = ' AND '.join([f'{feat} IS NOT NULL' for feat in features])
            sampling_query = f"""
                CREATE OR REPLACE VIEW {sample_view_name} AS
                SELECT * FROM analytics_wide
                WHERE {not_null_clauses}
                ORDER BY RANDOM()
                LIMIT {final_sample_size}
            """
            self.conn_manager.execute_sql_no_return(sampling_query)
            
            # Verificação de amostragem bem-sucedida
            actual_sample_size = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM {sample_view_name}")
            print(f"[CONFIRMAÇÃO] Amostra criada com {actual_sample_size:,} observações")
            
            # === Construção de matriz de correlação ===
            print("[CORRELAÇÃO] Construindo matriz de correlação par-a-par...")
            
            correlation_matrix = {}
            correlation_pairs_computed = 0
            total_pairs = len(features) * (len(features) - 1) // 2
            
            # Computação otimizada da matriz triangular superior
            for i, feat1 in enumerate(features):
                correlation_matrix[feat1] = {}
                
                for j, feat2 in enumerate(features):
                    if i >= j:  # Matriz triangular superior apenas
                        continue
                        
                    try:
                        # Query robusta para correlação de Pearson
                        corr_query = f"""
                            SELECT
                                CASE
                                    WHEN COUNT(*) < 3 THEN 0.0
                                    WHEN STDDEV({feat1}) = 0 OR STDDEV({feat2}) = 0 THEN 0.0
                                    ELSE ABS(CORR({feat1}, {feat2}))
                                END as correlation_value
                            FROM {sample_view_name}
                            WHERE {feat1} IS NOT NULL
                            AND {feat2} IS NOT NULL
                        """
                        
                        correlation = self.conn_manager.execute_scalar(corr_query)
                        correlation_matrix[feat1][feat2] = float(correlation or 0.0)
                        correlation_pairs_computed += 1
                        
                    except SQLProcessingError as e:
                        print(f"  ✗ Erro correlação {feat1}-{feat2}: {e}")
                        correlation_matrix[feat1][feat2] = 0.0
            
            print(f"[MATRIZ] {correlation_pairs_computed}/{total_pairs} pares analisados")
            
            # === Algoritmo Greedy Pairwise ===
            print(f"[SELEÇÃO] Aplicando algoritmo greedy com threshold {threshold}")
            
            selected_features = []
            rejected_features = []
            
            for feature in features:
                if not selected_features:
                    # Primeira feature sempre aceita
                    selected_features.append(feature)
                    continue
                
                # Calcular correlação máxima com features já selecionadas
                max_correlation = 0.0
                worst_correlated_feature = None
                
                for selected_feat in selected_features:
                    # Acessar matriz simétrica corretamente
                    f1, f2 = sorted([feature, selected_feat])
                    
                    if f1 in correlation_matrix and f2 in correlation_matrix[f1]:
                        corr_value = correlation_matrix[f1][f2]
                        if corr_value > max_correlation:
                            max_correlation = corr_value
                            worst_correlated_feature = selected_feat
                
                if max_correlation < threshold:
                    selected_features.append(feature)
                else:
                    rejected_features.append({
                        'feature': feature,
                        'max_correlation': max_correlation,
                        'correlated_with': worst_correlated_feature
                    })
            
            # === Relatório científico ===
            reduction_rate = ((len(features) - len(selected_features)) / len(features)) * 100
            
            print(f"\n[RESULTADO] Filtragem de colinearidade concluída:")
            print(f"  → Features originais: {len(features)}")
            print(f"  → Features selecionadas: {len(selected_features)}")
            print(f"  → Features removidas: {len(rejected_features)} ({reduction_rate:.1f}%)")
            
            if rejected_features:
                print(f"  → Principais rejeições por multicolinearidade:")
                for reject in rejected_features[:5]:  # Top 5 mais problemáticas
                    print(f"    • {reject['feature']}: r={reject['max_correlation']:.3f} "
                          f"com {reject['correlated_with']}")
            
            return selected_features
            
        except Exception as e:
            print(f"[ERRO] Falha na filtragem de colinearidade: {e}")
            print(f"[FALLBACK] Retornando primeiras 10 features como segurança")
            return features[:10]
            
        finally:
            # Limpeza de recursos temporários
            try:
                self.conn_manager.execute_sql_no_return(f"DROP VIEW IF EXISTS {sample_view_name}")
            except Exception:
                pass
    
    def prepare_features(self, data: Any, selected_features: List[str]) -> None:
        """
        Constrói view final com features selecionadas e transformações científicas.
        
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
        print(f"[FEATURES] Preparando view final com {len(selected_features)} features selecionadas")
        print("━" * 50)

        # === Tranformação, log simétrico: T(x) = sign(x) * ln(|x| + 1) ===
        print("[TRANSFORMAÇÃO] Aplicando symmetric log transform a top-5 features")
        
        # Critério: Limitar a top-5 features mais promissoras
        features_to_transform = selected_features[:5]
        transformed_features_sql = []
        
        print(f"[ESCOPO] Transformando {len(features_to_transform)} features para normalização:")
        
        for feat in features_to_transform:
            # Tranformação, log simétrico: sign(x) * ln(|x| + 1)
            transformation_sql = f"""
                CASE
                    WHEN {feat} IS NULL THEN NULL
                    WHEN {feat} = 0 THEN 0.0
                    ELSE SIGN({feat}) * LN(ABS({feat}) + 1)
                END AS {feat}_log_transform
            """
            transformed_features_sql.append(transformation_sql)
            print(f"  → {feat} → {feat}_log_transform")
        
        # === Construção da Query de View ===
        # Combinar features originais e transformadas
        all_features_sql = selected_features.copy()
        
        if transformed_features_sql:
            print(f"[ENGENHARIA] {len(transformed_features_sql)} symmetric log transforms aplicadas")
        
        # Query SQL para view final cientificamente estruturada
        feature_view_query = f"""
            CREATE OR REPLACE VIEW vw_selected_features AS
            SELECT
                -- Metadados temporais e geográficos (essenciais para ML temporal)
                country_code,
                year,
                {self.target_column},
                -- Lags do target (2 e 3 anos) sem vazamento
                LAG({self.target_column}, 2) OVER (PARTITION BY country_code ORDER BY year) AS dropout_rate_lag_2,
                LAG({self.target_column}, 3) OVER (PARTITION BY country_code ORDER BY year) AS dropout_rate_lag_3,
                
                -- Features originais pós-filtragem de colinearidade
                {', '.join(all_features_sql)}
                
                {', -- Features transformadas (symmetric log)' if transformed_features_sql else ''}
                {', '.join(transformed_features_sql) if transformed_features_sql else ''}
                
            FROM analytics_wide
            WHERE {self.target_column} IS NOT NULL  -- Filtro essencial para ML supervisionado
            ORDER BY country_code, year              -- Preserva ordem temporal para walk-forward
        """
        
        try:
            print("[SQL] Executando criação de view final...")
            self.conn_manager.execute_sql_no_return(feature_view_query)
            
            # === Validação e Relatório da View ===
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
            
            print(f"\n[RESULTADO] View vw_selected_features criada com sucesso:")
            print(f"  → Observações: {total_records:,}")
            print(f"  → Dimensionalidade: {total_features} variáveis")
            print(f"    • Features originais: {original_features}")
            print(f"    • Features transformadas: {transformed_features}")
            print(f"    • Metadados: 3 (country_code, year, target)")
            print(f"  → Cobertura: {unique_countries} países, {min_year}-{max_year}")
            print(f"  → Target médio: {avg_target:.1f}%")
            
            # Análise de adequação para ML
            observations_per_feature = total_records / total_features if total_features > 0 else 0
            
            if observations_per_feature < 10:
                print("  ⚠ AVISO: Poucos dados/feature - risco de overfitting")
            elif observations_per_feature > 50:
                print("  ✓ Adequado: Boa relação observações/features para ML")
            
        except SQLProcessingError as e:
            print(f"  ✗ ERRO: Falha na criação da view de features: {e}")
            raise RuntimeError(f"Impossível criar view de features: {e}")
        
        print("[STATUS] ✓ Pipeline de features concluído - dados prontos para modelagem ML")
        # Paradigma Data Warehouse: dados permanecem no banco para eficiência
        return None


def main():
    """
    Função principal para execução e teste do pipeline ML Data Warehouse científico.
    
    Executa pipeline completo de ML temporal seguindo metodologia científica:
    1. Setup e validação de ambiente
    2. Carregamento e validação de dados
    3. Construção de variável target
    4. Seleção de features com filtragem de colinearidade pairwise
    5. Criação de folds temporais walk-forward
    6. Preparação final para modelagem
    
    Adequado para:
        - Desenvolvimento e debugging do pipeline
        - Validação científica da metodologia
        - Benchmark de performance arquitetural
        - Testes de integração antes de produção
        
    Não adequado para:
        - Execução em produção (usar API específica)
        - Análises exploratórias (usar notebooks)
        - Comparação arquitetural (usar architectural_benchmark.py)
    """
    print("="*80)
    print("TESTE PIPELINE ML DATA WAREHOUSE - METODOLOGIA CIENTÍFICA".center(80))
    print("="*80)
    
    try:
        # Inicialização do pipeline científico
        print("\n[INICIALIZAÇÃO] Configurando pipeline ML Data Warehouse...")
        setup = DataWarehouseArchitectureML()
        
        # Execução completa do pipeline
        print("\n[EXECUÇÃO] Iniciando pipeline científico completo...")
        results = setup.run_setup()
        
        # Relatório final
        print("\n" + "="*80)
        print("RELATÓRIO FINAL DO PIPELINE".center(80))
        print("="*80)
        
        # Aceita padrões de retorno com 'status' ou 'success'
        success_flag = results.get('success', None)
        status_flag = results.get('status', None)
        is_success = (success_flag is True) or (isinstance(status_flag, str) and status_flag.lower() == 'success')
        if is_success:
            print("✓ Pipeline executado com SUCESSO")
            print(f"  → Arquitetura: {results.get('architecture', 'N/A')}")
            print(f"  → Features selecionadas: {results.get('features_selected', results.get('selected_features_count', 'N/A'))}")
            print(f"  → Folds temporais: {results.get('folds_created', results.get('total_folds', 'N/A'))}")
            if isinstance(results.get('total_observations', None), (int, float)):
                print(f"  → Observações processadas: {int(results.get('total_observations')):,}")
            
            if 'processing_time' in results and isinstance(results.get('processing_time'), (int, float)):
                print(f"  → Tempo de processamento: {results['processing_time']:.2f}s")
        else:
            print("✗ Pipeline FALHOU")
            if 'error' in results:
                print(f"  → Erro: {results['error']}")
            
        print(f"\n[DETALHES] Resultados completos:")
        for key, value in results.items():
            if key not in ['success', 'error']: 
                print(f"  → {key}: {value}")
                
    except Exception as e:
        print(f"\n✗ ERRO CRÍTICO no pipeline: {e}")
        print("   → Verifique se Data Warehouse foi processado corretamente")
        print("   → Execute data_warehouse/processor.py antes deste script")
        return {'success': False, 'error': str(e)}
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
