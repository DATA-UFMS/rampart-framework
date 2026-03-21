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
    # ── Targets ──────────────────────────────────────────────────────────
    'SE.SEC.CMPT.LO.ZS': {
        'name': 'Lower secondary completion rate',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'high',
    },
    'SE.SEC.NENR': {
        'name': 'School enrollment, secondary (% net)',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'high',
    },

    # ── Socioeconômicos ──────────────────────────────────────────────────
    'NY.GDP.PCAP.KD': {
        'name': 'GDP per capita (constant 2015 US$)',
        'unit': 'US Dollars',
        'expected_range': (0, 50000),
        'data_quality': 'high',
    },
    'SI.POV.NAHC': {
        'name': 'Poverty headcount ratio at national poverty lines (% of population)',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'medium',
    },
    'SI.POV.GINI': {
        'name': 'Gini index',
        'unit': 'Index (0-100)',
        'expected_range': (20, 70),
        'data_quality': 'medium',
    },
    'SL.UEM.TOTL.ZS': {
        'name': 'Unemployment, total (% of total labor force)',
        'unit': 'Percentage',
        'expected_range': (0, 50),
        'data_quality': 'high',
    },

    # ── Infraestrutura e Acesso ──────────────────────────────────────────
    'EG.ELC.ACCS.ZS': {
        'name': 'Access to electricity (% of population)',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'high',
    },
    'SH.H2O.BASW.ZS': {
        'name': 'People using at least basic drinking water services (% of population)',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'high',
    },
    'IT.NET.USER.ZS': {
        'name': 'Individuals using the Internet (% of population)',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'high',
    },

    # ── Demográficos ─────────────────────────────────────────────────────
    'SP.POP.0014.TO.ZS': {
        'name': 'Population ages 0-14 (% of total population)',
        'unit': 'Percentage',
        'expected_range': (10, 50),
        'data_quality': 'high',
    },
    'SP.POP.GROW': {
        'name': 'Population growth (annual %)',
        'unit': 'Percentage',
        'expected_range': (-3, 5),
        'data_quality': 'high',
    },
    'SP.ADO.TFRT': {
        'name': 'Adolescent fertility rate (births per 1,000 women ages 15-19)',
        'unit': 'Per 1000',
        'expected_range': (0, 200),
        'data_quality': 'high',
    },

    # ── Educacionais de Contexto ─────────────────────────────────────────
    'SE.XPD.TOTL.GD.ZS': {
        'name': 'Government expenditure on education, total (% of GDP)',
        'unit': 'Percentage',
        'expected_range': (0, 15),
        'data_quality': 'medium',
    },
    'SE.ENR.PRSC.FM.ZS': {
        'name': 'School enrollment, primary and secondary (gross), gender parity index (GPI)',
        'unit': 'Index',
        'expected_range': (0.5, 1.5),
        'data_quality': 'medium',
    },
    'SE.ADT.LITR.ZS': {
        'name': 'Literacy rate, adult total (% of people ages 15 and above)',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'medium',
    },
    'SE.PRM.ENRL.TC.ZS': {
        'name': 'Pupil-teacher ratio, primary',
        'unit': 'Ratio',
        'expected_range': (5, 80),
        'data_quality': 'medium',
    },
    'SE.SEC.TCHR.FE.ZS': {
        'name': 'Female teachers in secondary education (% of total teachers)',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'medium',
    },
    'SE.SEC.ENRL.TC.ZS': {
        'name': 'Pupil-teacher ratio, secondary',
        'unit': 'Ratio',
        'expected_range': (5, 80),
        'data_quality': 'medium',
    },

    # ── Saúde e Bem-Estar ────────────────────────────────────────────────
    'SH.STA.MALN.ZS': {
        'name': 'Prevalence of underweight, weight for age (% of children under 5)',
        'unit': 'Percentage',
        'expected_range': (0, 50),
        'data_quality': 'medium',
    },
    'SH.IMM.MEAS': {
        'name': 'Immunization, measles (% of children ages 12-23 months)',
        'unit': 'Percentage',
        'expected_range': (0, 100),
        'data_quality': 'high',
    },
    'SP.DYN.IMRT.IN': {
        'name': 'Mortality rate, infant (per 1,000 live births)',
        'unit': 'Per 1000',
        'expected_range': (0, 200),
        'data_quality': 'high',
    },

    # ── Governança e Institucional ───────────────────────────────────────
    'VC.IHR.PSRC.P5': {
        'name': 'Intentional homicides (per 100,000 people)',
        'unit': 'Per 100k',
        'expected_range': (0, 120),
        'data_quality': 'medium',
    },
    'GE.EST': {
        'name': 'Government effectiveness: estimate',
        'unit': 'Index (-2.5 to 2.5)',
        'expected_range': (-2.5, 2.5),
        'data_quality': 'medium',
    },
}

