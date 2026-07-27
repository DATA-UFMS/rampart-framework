-- ============================================================================
-- SCHEMA DATA WAREHOUSE - EDUCATIONAL ANALYTICS
-- Arquitetura Data Warehouse com Wide Table otimizada para ML Analytics
-- Metodologia: Data Warehouse para garantir integridade e performance
-- ============================================================================

-- ============================================================================
-- DIMENSÃO PAÍSES
-- Única dimensão conectada, contém estratificação para análises comparativas
-- ============================================================================

-- Dimensão Países: Metadados geográficos e estratificação econômica
CREATE TABLE IF NOT EXISTS dim_countries (
    entity_id VARCHAR(3) PRIMARY KEY,
    entity_name VARCHAR(100) NOT NULL,
    entity_stratum VARCHAR(50) NOT NULL
);

-- ============================================================================
-- TABELA PRINCIPAL WIDE TABLE
-- Estrutura otimizada para Feature Engineering e ML Pipeline
-- ============================================================================

-- Wide Table: Todas as métricas desnormalizadas para performance ML
DROP TABLE IF EXISTS analytics_wide;

CREATE TABLE IF NOT EXISTS analytics_wide (
    -- Chave primária composta
    entity_id VARCHAR(3) NOT NULL,
    year INTEGER NOT NULL,
    
    -- Metadados dimensionais desnormalizados para performance
    entity_name VARCHAR(100),
    entity_stratum VARCHAR(50),
    
    -- GRUPO: Variáveis Target Educacionais
    target_source_rate DOUBLE,
    enrollment_rate_secondary_net DOUBLE,
    
    -- GRUPO: Indicadores Macroeconômicos
    gdp_per_capita_constant_2015 DOUBLE,
    poverty_headcount_national DOUBLE,
    gini_index DOUBLE,
    unemployment_total DOUBLE,
    
    -- GRUPO: Infraestrutura e Acesso a Serviços
    electricity_access_percent DOUBLE,
    basic_water_services_percent DOUBLE,
    internet_users_percent DOUBLE,
    
    -- GRUPO: Indicadores de Saúde Pública
    malnutrition_prevalence_weight_age DOUBLE,
    immunization_measles_percent DOUBLE,
    mortality_rate_infant_per_1000 DOUBLE,
    
    -- GRUPO: Demografia e Estrutura Populacional
    population_ages_0_14_percent DOUBLE,
    population_growth_annual DOUBLE,
    adolescent_fertility_rate DOUBLE,
    
    -- GRUPO: Sistema Educacional (Métricas Complementares)
    education_expenditure_gdp_percent DOUBLE,
    gender_parity_index_secondary DOUBLE,
    adult_literacy_rate DOUBLE,
    pupil_teacher_ratio_primary DOUBLE,
    female_teachers_secondary_percent DOUBLE,
    pupil_teacher_ratio_secondary DOUBLE,
    
    -- GRUPO: Governança e Segurança
    intentional_homicides_per_100k DOUBLE,
    government_effectiveness DOUBLE,
    
    -- GRUPO: Metadados de Qualidade e Rastreabilidade (Data Warehouse)
    data_source VARCHAR,
    data_completeness_score DOUBLE,
    etl_batch_id VARCHAR,
    collection_timestamp TIMESTAMP,
    
    -- Constraints de integridade
    PRIMARY KEY (entity_id, year),
    FOREIGN KEY (entity_id) REFERENCES dim_countries(entity_id)
);



-- ============================================================================
-- ÍNDICES PARA OTIMIZAÇÃO DE CONSULTAS ANALÍTICAS
-- Estratégia: Acelerar JOINs, agregações e filtros mais frequentes
-- ============================================================================

-- Índices compostos para wide table (ordem baseada em cardinalidade e seletividade)
CREATE INDEX IF NOT EXISTS idx_analytics_country_year ON analytics_wide(entity_id, year);
CREATE INDEX IF NOT EXISTS idx_analytics_targets ON analytics_wide(target_source_rate, enrollment_rate_secondary_net);
CREATE INDEX IF NOT EXISTS idx_analytics_stratum ON analytics_wide(entity_stratum);
CREATE INDEX IF NOT EXISTS idx_analytics_completeness ON analytics_wide(data_completeness_score);
CREATE INDEX IF NOT EXISTS idx_analytics_year ON analytics_wide(year);

-- Índices para dimensão países (aceleram JOINs e filtros dimensionais)
CREATE INDEX IF NOT EXISTS idx_dim_countries_stratum ON dim_countries(entity_stratum);

-- ============================================================================
-- VIEWS ANALÍTICAS PARA RELATÓRIOS CIENTÍFICOS
-- Implementação: Views não-materializadas para flexibilidade de consulta
-- ============================================================================

-- View agregada: Estatísticas longitudinais por país
CREATE OR REPLACE VIEW vw_country_statistics AS
SELECT 
    c.entity_id,
    c.entity_name,
    c.entity_stratum,
    COUNT(DISTINCT a.year) as years_covered,
    MIN(a.year) as first_year,
    MAX(a.year) as last_year,
    AVG(a.data_completeness_score) as avg_completeness,
    AVG(a.target_source_rate) as avg_target_source_rate,
    AVG(a.gdp_per_capita_constant_2015) as avg_gdp_per_capita
FROM dim_countries c
LEFT JOIN analytics_wide a USING (entity_id)
GROUP BY c.entity_id, c.entity_name, c.entity_stratum;

-- View de controle: Estatísticas de cobertura de dados
CREATE OR REPLACE VIEW vw_data_coverage AS
SELECT 
    entity_stratum,
    COUNT(DISTINCT entity_id) as countries_count,
    COUNT(DISTINCT year) as years_covered,
    AVG(data_completeness_score) as avg_completeness
FROM analytics_wide
GROUP BY entity_stratum;

-- ============================================================================
-- SCHEMA WIDE TABLE INICIALIZADO
-- Estrutura optimizada para Framework CAES (Controlled Architectural Study)
-- Metodologia: Data Warehouse + Wide Table para performance ML máxima
-- ============================================================================