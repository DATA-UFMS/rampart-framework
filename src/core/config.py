"""
Configurações compartilhadas entre arquiteturas
"""

from datetime import datetime
from typing import Dict

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

LATIN_AMERICA_COUNTRIES = []
for stratum in COUNTRY_STRATA.values():
    LATIN_AMERICA_COUNTRIES.extend(stratum)

# ============================================================================
# CONFIGURAÇÃO DE ARQUIVOS
# ============================================================================
BASE_DATA_DIR = 'data'

# ============================================================================
# CONFIGURAÇÃO DE PERFORMANCE
# ============================================================================
BENCHMARK_CONFIG = {
    'repetitions': 10,  # Número de repetições por teste (n do protocolo experimental)
    'warmup_runs': 2,   # Execuções de aquecimento
    'timeout_seconds': 3600,  # Timeout por operação
    'memory_limit_gb': 16,
    'profile_memory': True,
    'profile_cpu': True,
    'save_intermediate': True
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
    
    try:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while current_dir != '/' and current_dir != os.path.dirname(current_dir):
            if os.path.exists(os.path.join(current_dir, 'README.md')):
                return current_dir
            current_dir = os.path.dirname(current_dir)
    except Exception:
        pass

    current_dir = os.path.abspath(os.getcwd())
    while current_dir != '/' and current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, 'README.md')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    if os.path.exists('README.md'):
        return os.path.abspath('.')
    
    raise FileNotFoundError("Não foi possível encontrar o root do projeto (README.md não encontrado)")

DEFAULT_DATASET = 'worldbank'


def get_dataset_name() -> str:
    """
    Dataset em execução, propagado pelo pipeline via DATASET_NAME.

    Um único ponto de leitura: o default replicado em dois lugares permitiria
    que um módulo resolvesse um dataset e outro resolvesse outro na mesma
    execução.
    """
    import os
    return os.environ.get('DATASET_NAME', DEFAULT_DATASET)


def get_outputs_root() -> str:
    """
    Raiz dos artefatos do dataset em execução.

    Segregada por dataset. Sem isso, executar um segundo dataset sobrescreve os
    artefatos do primeiro sob os mesmos nomes, e uma execução interrompida
    deixa artefatos de dois datasets convivendo sem que nada o registre — os
    resultados publicados foram separados em diretórios manualmente, e não pelo
    código.
    """
    import os
    return os.path.join(get_project_root(), 'outputs', get_dataset_name())


def get_absolute_output_path(relative_path: str) -> str:
    """
    Converte path relativo 'outputs/...' em path absoluto sob a raiz do dataset.

    Exemplo, com DATASET_NAME=inep_censo:
      'outputs/statistics' -> '/<projeto>/outputs/inep_censo/statistics'
    """
    import os
    prefix = 'outputs' + os.sep
    if relative_path.startswith(prefix) or relative_path.startswith('outputs/'):
        relative_path = relative_path.split('/', 1)[1] if '/' in relative_path \
            else ''
    return os.path.join(get_outputs_root(), relative_path)

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
