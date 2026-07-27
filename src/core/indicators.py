"""
World Bank indicator definitions
"""

# ============================================================================
# TARGET VARIABLES (Dependent Variables)
# ============================================================================
TARGET_INDICATORS = {
    'SE.SEC.CMPT.LO.ZS': 'lower_secondary_completion_rate',
    'SE.SEC.NENR': 'enrollment_rate_secondary_net',
}

# ============================================================================
# PREDICTORS - Structural Socioeconomic Variables
# ============================================================================
SOCIOECONOMIC_INDICATORS = {
    'NY.GDP.PCAP.KD': 'gdp_per_capita_constant_2015',
    'SI.POV.NAHC': 'poverty_headcount_national',
    'SI.POV.GINI': 'gini_index',
    'SL.UEM.TOTL.ZS': 'unemployment_total',
}

# ============================================================================
# PREDICTORS - Infrastructure and Access Variables
# ============================================================================
INFRASTRUCTURE_INDICATORS = {
    'EG.ELC.ACCS.ZS': 'electricity_access_percent',
    'SH.H2O.BASW.ZS': 'basic_water_services_percent',
    'IT.NET.USER.ZS': 'internet_users_percent',
}

# ============================================================================
# PREDICTORS - Demographic Context Variables
# ============================================================================
DEMOGRAPHIC_INDICATORS = {
    'SP.POP.0014.TO.ZS': 'population_ages_0_14_percent',
    'SP.POP.GROW': 'population_growth_annual',
    'SP.ADO.TFRT': 'adolescent_fertility_rate',
}

# ============================================================================
# PREDICTORS - Educational Context Variables (No Leakage)
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
# PREDICTORS - Health and Well-Being Variables
# ============================================================================
HEALTH_INDICATORS = {
    'SH.STA.MALN.ZS': 'malnutrition_prevalence_weight_age',
    'SH.IMM.MEAS': 'immunization_measles_percent',
    'SP.DYN.IMRT.IN': 'mortality_rate_infant_per_1000',
}

# ============================================================================
# PREDICTORS - Institutional and Governance Variables
# ============================================================================
GOVERNANCE_INDICATORS = {
    'VC.IHR.PSRC.P5': 'intentional_homicides_per_100k',
}

# ============================================================================
# PREDICTORS - Complementary Economic Variables (reserved for expansion)
# ============================================================================
ECONOMIC_INDICATORS = {}

# ============================================================================
# CONSOLIDATION
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

