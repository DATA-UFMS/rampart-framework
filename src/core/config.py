"""
Settings shared across architectures
"""

from datetime import datetime
from typing import Dict, Optional

# ============================================================================
# TEMPORAL CONFIGURATION
# ============================================================================
START_YEAR = 2000
END_YEAR = 2023

# ============================================================================
# GEOGRAPHIC STRATIFICATION (ISO 2-letter codes - World Bank API)
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
# FILE CONFIGURATION
# ============================================================================
BASE_DATA_DIR = 'data'

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================
BENCHMARK_CONFIG = {
    'repetitions': 10,  # Number of repetitions per test (n of the experimental protocol)
    'warmup_runs': 2,   # Warm-up runs
    'timeout_seconds': 3600,  # Timeout per operation
    'memory_limit_gb': 16,
    'profile_memory': True,
    'profile_cpu': True,
    'save_intermediate': True
}

# ============================================================================
# EXECUTION METADATA
# ============================================================================
def get_project_root() -> str:
    """
    Returns the absolute path of the project root
    Works regardless of the current working directory
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
    
    raise FileNotFoundError("Could not find the project root (README.md not found)")

DEFAULT_DATASET = 'worldbank'


def get_dataset_name() -> str:
    """
    Dataset being run, propagated by the pipeline via DATASET_NAME.

    A single point of reading: the default replicated in two places would let
    one module resolve one dataset and another resolve a different one in the
    same run.
    """
    import os
    return os.environ.get('DATASET_NAME', DEFAULT_DATASET)


def get_outputs_root() -> str:
    """
    Root of the artifacts for the dataset being run.

    Segregated per dataset. Without it, running a second dataset overwrites the
    first one's artifacts under the same names, and an interrupted run leaves
    artifacts from two datasets coexisting with nothing recording it — the
    published results were separated into directories by hand, not by the
    code.
    """
    import os
    return os.path.join(get_project_root(), 'outputs', get_dataset_name())


def get_absolute_output_path(relative_path: str) -> str:
    """
    Converts a relative 'outputs/...' path into an absolute path under the dataset root.

    Example, with DATASET_NAME=inep_censo:
      'outputs/statistics' -> '/<project>/outputs/inep_censo/statistics'
    """
    import os

    # By component, not by textual prefix. The bare form 'outputs' does not
    # start with 'outputs/', so it escaped the stripping and produced
    # outputs/<dataset>/outputs -- that is how the environment snapshot came to
    # be written one level below where every consumer reads it.
    parts = [part for part in relative_path.replace(os.sep, '/').split('/')
             if part]
    if parts and parts[0] == 'outputs':
        parts = parts[1:]
    return os.path.join(get_outputs_root(), *parts)

def get_execution_metadata() -> Dict:
    """Returns metadata of the current run"""
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


def write_environment_snapshot(destination: str, *, extra: Optional[Dict] = None) -> str:
    """Writes the run's configuration and environment receipt.

    Lives here, and not in the orchestrator, because the benchmark needs it too: a
    standalone benchmark run produced no receipt at all, and a latency without
    the environment that produced it is comparable to nothing.

    Args:
        destination: directory to write to
        extra: additional fields from the caller (the measured phase, for example)

    Returns:
        Path of the written file.
    """
    import hashlib
    import importlib.metadata
    import json
    import os
    import platform
    import subprocess
    import sys
    from datetime import datetime, timezone

    from core.scientific_config import SCIENTIFIC_CONFIG

    os.makedirs(destination, exist_ok=True)
    payload: Dict = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'dataset': get_dataset_name(),
        'scientific_config': SCIENTIFIC_CONFIG,
        'python': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
    }

    root = get_project_root()
    try:
        payload['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except Exception:
        payload['git_commit'] = 'unavailable'

    try:
        payload['installed_packages'] = {
            dist.metadata['Name']: dist.version
            for dist in importlib.metadata.distributions()
        }
    except Exception:
        payload['installed_packages'] = 'unavailable'

    try:
        payload['hardware'] = get_execution_metadata()
    except Exception:
        payload['hardware'] = 'unavailable'

    lock_path = os.path.join(root, 'requirements-lock.txt')
    if os.path.exists(lock_path):
        with open(lock_path, 'rb') as handler:
            payload['requirements_lock_sha256'] = hashlib.sha256(
                handler.read()).hexdigest()

    if extra:
        payload.update(extra)

    path = os.path.join(destination, 'scientific_config_snapshot.json')
    with open(path, 'w', encoding='utf-8') as handler:
        json.dump(payload, handler, indent=2, ensure_ascii=False)
    return path
