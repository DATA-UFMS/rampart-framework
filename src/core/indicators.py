"""
Definições de indicadores do Banco Mundial
"""

# ============================================================================
# TARGET VARIABLES (Variáveis Dependentes)
# ============================================================================
TARGET_INDICATORS = {
    'SE.SEC.CMPT.LO.ZS': 'lower_secondary_completion_rate',
    'SE.SEC.NENR': 'enrollment_rate_secondary_net',
}

# ============================================================================
# PREDICTORS - Variáveis Socioeconômicas Estruturais
# ============================================================================
SOCIOECONOMIC_INDICATORS = {
    'NY.GDP.PCAP.KD': 'gdp_per_capita_constant_2015',
    'SI.POV.NAHC': 'poverty_headcount_national',
    'SI.POV.GINI': 'gini_index',
    'SL.UEM.TOTL.ZS': 'unemployment_total',
}

# ============================================================================
# PREDICTORS - Variáveis de Infraestrutura e Acesso
# ============================================================================
INFRASTRUCTURE_INDICATORS = {
    'EG.ELC.ACCS.ZS': 'electricity_access_percent',
    'SH.H2O.BASW.ZS': 'basic_water_services_percent',
    'IT.NET.USER.ZS': 'internet_users_percent',
}

# ============================================================================
# PREDICTORS - Variáveis Demográficas de Contexto
# ============================================================================
DEMOGRAPHIC_INDICATORS = {
    'SP.POP.0014.TO.ZS': 'population_ages_0_14_percent',
    'SP.POP.GROW': 'population_growth_annual',
    'SP.ADO.TFRT': 'adolescent_fertility_rate',
}

# ============================================================================
# PREDICTORS - Variáveis Educacionais de Contexto (Sem Leakage)
# ============================================================================
EDUCATION_CONTEXT_INDICATORS = {
    'SE.XPD.TOTL.GD.ZS': 'education_expenditure_gdp_percent',
    'SE.ENR.PRSC.FM.ZS': 'gender_parity_index_secondary',
    'SE.ADT.LITR.ZS': 'adult_literacy_rate',
    'SE.PRM.ENRL.TC.ZS': 'pupil_teacher_ratio_primary',
    'SE.SEC.TCHR.FE.ZS': 'female_teachers_secondary_percent',
    'SE.SEC.ENRL.TC.ZS': 'pupil_teacher_ratio_secondary',   
}

# ============================================================================
# PREDICTORS - Variáveis de Saúde e Bem-Estar
# ============================================================================
HEALTH_INDICATORS = {
    'SH.STA.MALN.ZS': 'malnutrition_prevalence_weight_age',
    'SH.IMM.MEAS': 'immunization_measles_percent',
    'SP.DYN.IMRT.IN': 'mortality_rate_infant_per_1000',
}

# ============================================================================
# PREDICTORS - Variáveis Institucionais e Governança
# ============================================================================
GOVERNANCE_INDICATORS = {
    'VC.IHR.PSRC.P5': 'intentional_homicides_per_100k',
    'GE.EST': 'government_effectiveness',
}

# ============================================================================
# PREDICTORS - Variáveis Econômicas Complementares
# ============================================================================
ECONOMIC_INDICATORS = {
    # Economic indicators section now empty - keeping structure for future additions
}

# ============================================================================
# CONSOLIDAÇÃO
# ============================================================================
ALL_INDICATORS = {
    **TARGET_INDICATORS,
    **SOCIOECONOMIC_INDICATORS,
    **INFRASTRUCTURE_INDICATORS,
    **DEMOGRAPHIC_INDICATORS,
    **EDUCATION_CONTEXT_INDICATORS,
    **HEALTH_INDICATORS,
    **GOVERNANCE_INDICATORS,
    **ECONOMIC_INDICATORS
}

# Mapeamento de categorias
INDICATOR_CATEGORIES = {
    'target': TARGET_INDICATORS,
    'socioeconomic': SOCIOECONOMIC_INDICATORS,
    'infrastructure': INFRASTRUCTURE_INDICATORS,
    'demographic': DEMOGRAPHIC_INDICATORS,
    'education_context': EDUCATION_CONTEXT_INDICATORS,
    'health': HEALTH_INDICATORS,
    'governance': GOVERNANCE_INDICATORS,
    'economic': ECONOMIC_INDICATORS
}

# ============================================================================
# METADADOS DOS INDICADORES
# ============================================================================
INDICATOR_METADATA = {
    # Targets
    'SE.SEC.CMPT.LO.ZS': {
        'name': 'Lower secondary completion rate',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'high'
    },
    'SE.SEC.NENR': {
        'name': 'School enrollment, secondary (% net)',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'high'
    },
    
    # Socioeconômicos
    'NY.GDP.PCAP.KD': {
        'name': 'GDP per capita (constant 2015 US$)',
        'unit': 'US Dollars',
        'expected_range': (0, 50000),
        'data_quality': 'high'
    },
    'SI.POV.GINI': {
        'name': 'Gini index',
        'unit': 'Index (0-100)',
        'expected_range': (20, 70),
        'data_quality': 'medium'
    },
    
    # Adicionar mais metadados conforme necessário...
}

def get_indicator_info(indicator_code: str) -> dict:
    """Retorna informações sobre um indicador específico"""
    return INDICATOR_METADATA.get(indicator_code, {
        'name': ALL_INDICATORS.get(indicator_code, 'Unknown'),
        'unit': 'Unknown',
        'expected_range': (None, None),
        'data_quality': 'unknown'
    })

def get_indicators_by_category(category: str) -> dict:
    """Retorna indicadores de uma categoria específica"""
    return INDICATOR_CATEGORIES.get(category, {})

def validate_indicator_value(indicator_code: str, value: float) -> bool:
    """Valida se um valor está dentro do range esperado para o indicador"""
    metadata = get_indicator_info(indicator_code)
    min_val, max_val = metadata['expected_range']
    
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    
    return True