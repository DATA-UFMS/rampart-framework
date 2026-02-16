"""
Configurações compartilhadas entre arquiteturas
"""

from datetime import datetime
from typing import Dict, List

# ============================================================================
# CONFIGURAÇÃO TEMPORAL
# ============================================================================
START_YEAR = 2000
END_YEAR = 2023

# ============================================================================
# ESTRATIFICAÇÃO GEOGRÁFICA (códigos ISO 2 letras - API World Bank)
# ============================================================================
COUNTRY_STRATA = {
    'large_economies': ['BR', 'MX', 'AR', 'CO', 'CL', 'PE'],
    'medium_economies': ['VE', 'EC', 'GT', 'UY', 'CR', 'PA', 'BO'],
    'small_economies': ['PY', 'SV', 'NI', 'HN', 'JM', 'TT', 'GY'],
    'caribbean_small': ['BB', 'BZ', 'SR', 'AG', 'DM', 'GD', 'KN', 'LC', 'VC'],
    'special_cases': ['CU', 'HT', 'DO']
}

# Lista completa de países
LATIN_AMERICA_COUNTRIES = []
for stratum in COUNTRY_STRATA.values():
    LATIN_AMERICA_COUNTRIES.extend(stratum)

# ============================================================================
# CONFIGURAÇÃO DE ARQUIVOS
# ============================================================================
BASE_DATA_DIR = 'data'
SQL_FIRST_DIR = f'{BASE_DATA_DIR}/sql_first'
SCHEMA_ON_READ_DIR = f'{BASE_DATA_DIR}/schema_on_read'
BENCHMARKS_DIR = f'{BASE_DATA_DIR}/benchmarks'

# ============================================================================
# CONFIGURAÇÃO DE PERFORMANCE
# ============================================================================
BENCHMARK_CONFIG = {
    'repetitions': 10,  # Número de repetições por teste
    'warmup_runs': 2,   # Execuções de aquecimento
    'timeout_seconds': 3600,  # Timeout por operação
    'memory_limit_gb': 16,
    'profile_memory': True,
    'profile_cpu': True,
    'save_intermediate': True
}

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_prefix': 'research_architectures',
    'max_file_size_mb': 100,
    'backup_count': 5
}

# ============================================================================
# CONFIGURAÇÃO DE VALIDAÇÃO
# ============================================================================
VALIDATION_CONFIG = {
    'tolerance_numeric': 1e-6,  # Tolerância para comparações numéricas
    'tolerance_percentage': 0.01,  # Tolerância percentual
    'required_columns': [  # Colunas obrigatórias no resultado final
        'country_code', 'year', 'lower_secondary_completion_rate',
        'enrollment_rate_secondary_net'
    ],
    'validate_collinearity': True,  # Validar filtragem de colinearidade
    'validate_correlations': True,  # Validar matriz de correlação
    'validate_imputation': True  # Validar qualidade da imputação
}

# ============================================================================
# METADADOS DE EXECUÇÃO
# ============================================================================
def get_project_root() -> str:
    """
    Retorna o path absoluto do root do projeto
    Funciona independente do working directory atual
    """
    import os
    
    # Primeiro, tentar a partir do arquivo atual
    try:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while current_dir != '/' and current_dir != os.path.dirname(current_dir):
            if os.path.exists(os.path.join(current_dir, 'README.md')):
                return current_dir
            current_dir = os.path.dirname(current_dir)
    except:
        pass
    
    # Fallback: tentar a partir do working directory atual
    current_dir = os.path.abspath(os.getcwd())
    while current_dir != '/' and current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, 'README.md')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # Último fallback: assumir que estamos no root se README.md existe
    if os.path.exists('README.md'):
        return os.path.abspath('.')
    
    raise FileNotFoundError("Não foi possível encontrar o root do projeto (README.md não encontrado)")

def get_absolute_output_path(relative_path: str) -> str:
    """
    Converte path relativo 'outputs/...' para path absoluto
    Exemplo: 'outputs/data_collection/sql_first' -> '/full/path/to/project/outputs/data_collection/sql_first'
    """
    import os
    project_root = get_project_root()
    if relative_path.startswith('outputs/'):
        return os.path.join(project_root, relative_path)
    else:
        return os.path.join(project_root, 'outputs', relative_path)

def get_execution_metadata() -> Dict:
    """Retorna metadados da execução atual"""
    import platform
    import psutil
    
    return {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / 1024**3,
        'disk_total_gb': psutil.disk_usage('/').total / 1024**3,
        'config_version': '1.0.0',
        'project_root': get_project_root()
    }
