#!/usr/bin/env python3
"""
Pipeline orquestrador da pesquisa: executa as fases na ordem com caminhos absolutos.

Fases:
  1) Coleta bruta (World Bank)
  2) Processamento arquitetural (Data Lake, Data Warehouse, Polars DataFrame)
  3) Setup ML (folds idênticos com gaps de 2 anos; seleção de features)
  4) Baselines (3 arquiteturas)
  5) Hierárquicos (3 arquiteturas)
  6) Benchmark arquitetural (3 arquiteturas)
  7) Testes estatísticos de validação
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

try:
    from src.core.scientific_config import SCIENTIFIC_CONFIG
except Exception as exc:
    print(f"[WARN] Falha ao importar SCIENTIFIC_CONFIG: {exc}")
    SCIENTIFIC_CONFIG = {}

try:
    from src.core.config import get_execution_metadata
except Exception as exc:
    print(f"[WARN] Falha ao importar get_execution_metadata: {exc}")
    get_execution_metadata = None

try:
    from src.core.validation import TemporalValidator
except Exception as exc:
    print(f"[WARN] Falha ao importar TemporalValidator: {exc}")
    TemporalValidator = None

def print_conclusion(msg: str) -> None:
    print(f"\n{msg}")

def print_system(msg: str) -> None:
    print(f"\n{msg}")

def print_config(msg: str) -> None:
    print(f"  {msg}")

def print_step(msg: str) -> None:
    print(f"  {msg}")

def print_success(msg: str) -> None:
    print(f"  {msg}")

def print_error(msg: str) -> None:
    print(f"  ERRO: {msg}")

def run(cmd: str) -> None:
    """Executa um subprocesso com PYTHONPATH configurado para src/ para imports consistentes."""
    print(f"\n$ {cmd}")
    env = os.environ.copy()
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, shell=True, check=True, env=env)

def _snapshot_scientific_config(root: str) -> None:
    """Salva snapshot da configuração científica e do ambiente."""
    snapshot_dir = os.path.join(root, "outputs")
    os.makedirs(snapshot_dir, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scientific_config": SCIENTIFIC_CONFIG,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": root,
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        payload["git_commit"] = commit
    except Exception:
        payload["git_commit"] = "unavailable"

    try:
        payload["installed_packages"] = {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        }
    except Exception:
        payload["installed_packages"] = "unavailable"

    if get_execution_metadata is not None:
        try:
            payload["hardware"] = get_execution_metadata()
        except Exception:
            payload["hardware"] = "unavailable"

    req_path = os.path.join(root, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "rb") as f:
            payload["requirements_txt_sha256"] = hashlib.sha256(f.read()).hexdigest()

    snapshot_path = os.path.join(snapshot_dir, "scientific_config_snapshot.json")
    with open(snapshot_path, "w", encoding="utf-8") as handler:
        json.dump(payload, handler, indent=2, ensure_ascii=False)
    print_system(f"Snapshot científico registrado em {snapshot_path}")


def _discover():
    """Descoberta preguiçosa de paradigmas — importação aciona o registro."""
    from src.core.paradigm_registry import discover_paradigms
    paradigms = discover_paradigms()
    if not paradigms:
        raise RuntimeError("Nenhum paradigma descoberto — verifique src/architectures_ml/*/setup.py")
    return paradigms


def _validate_anti_leakage_gate(root: str) -> None:
    """Valida integridade temporal de todos os folds antes de prosseguir ao benchmark."""
    if TemporalValidator is None:
        raise RuntimeError("TemporalValidator não disponível — validação anti-leakage impossível")

    gap = int(SCIENTIFIC_CONFIG.get('temporal_gap_years', 2))
    embargo = int(SCIENTIFIC_CONFIG.get('embargo_years', 0))
    validator = TemporalValidator(min_gap_years=gap, embargo_years=embargo)

    for arch in _discover():
        folds_path = os.path.join(
            root, 'outputs', 'ml_pipeline', 'architectures', arch, 'prep',
            f'temporal_folds_{arch}.json'
        )
        if not os.path.exists(folds_path):
            raise FileNotFoundError(f"Folds não encontrados: {folds_path}")

        with open(folds_path, 'r') as f:
            folds_config = json.load(f)

        folds = folds_config.get('folds', [])
        validator.enforce_walk_forward(folds)
        print_config(f"  {arch}: {len(folds)} folds — integridade temporal verificada")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de pesquisa - benchmarking arquitetural")
    parser.add_argument(
        '--dataset', default='worldbank',
        choices=['worldbank', 'inep_censo'],
        help='Dataset a processar (default: worldbank)'
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    os.environ['DATASET_NAME'] = dataset_name  # propaga via run() para subprocessos
    root = os.path.abspath(os.path.dirname(__file__))
    py = sys.executable

    print_conclusion(f"Pipeline iniciado (dataset: {dataset_name})")
    _snapshot_scientific_config(root)
    paradigms = _discover()
    print_system("Etapa 0: Snapshot de reprodutibilidade")
    print_config("Snapshot salvo em outputs/scientific_config_snapshot.json")

    if dataset_name == 'worldbank':
        print_system("Etapa 1/9: Coleta")
        print_config("Fonte: World Bank")
        run(f"{py} {root}/src/collection/raw_data_collector.py")
    else:
        print_system("Etapa 1/9: Coleta")
        print_config("Fonte: INEP Censo Escolar")
        run(f"{py} {root}/src/collection/inep_collector.py")
    print_success("Etapa 1 concluida")

    n_paradigms = len(paradigms)
    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print_system(f"Etapa 2{chr(96+i)}/9: Processamento {arch}")
        print_config(f"Arquitetura: {info['label']}")
        run(f"{py} {root}/{info['processor_script']}")
    print_success(f"Etapa 2 concluida ({n_paradigms} paradigmas)")

    print_system("Etapa 3: Setup ML")
    print_config("Gaps temporais: 2 anos (P1-P3)")
    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print_system(f"Etapa 3{chr(96+i)}/9: Setup ML {arch}")
        print_config(f"Arquitetura: {info['label']}")
        run(f"{py} {root}/{info['setup_script']}")
    print_success(f"Etapa 3 concluida ({n_paradigms} paradigmas)")

    print_system("Gate anti-leakage")
    _validate_anti_leakage_gate(root)
    print_success("Todos os folds passaram na validacao temporal")

    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print_system(f"Etapa 4{chr(96+i)}/9: Baselines {arch}")
        run(f"{py} {root}/{info['baseline_script']}")
    print_success(f"Etapa 4 concluida ({n_paradigms} paradigmas)")

    print_system("Etapa 5/9: Hierarquicos")
    for arch, info in paradigms.items():
        run(f"{py} {root}/{info['hierarchical_script']}")
    print_success(f"Etapa 5 concluida ({n_paradigms} paradigmas)")

    print_system("Etapa 6/9: Benchmark arquitetural")
    run(f"{py} {root}/src/benchmarking/architectural_benchmark.py --repetitions 30 --warmup 2")
    print_success("Etapa 6 concluida")

    print_system("Etapa 7/9: Testes estatisticos")

    benchmark_csv = f"{root}/outputs/benchmarks/architectural_benchmark_results.csv"
    if os.path.exists(benchmark_csv):
        print_step("Etapa 7a/9: Significancia (bootstrap)")
        run(f"{py} {root}/src/statistical_validation/significance_tests.py")
    else:
        print_error("Arquivo de benchmark nao encontrado, pulando testes de significancia")

    print_step("Etapa 7b/9: Equivalencia (SESOI + IC)")
    run(f"{py} {root}/src/statistical_validation/equivalence_estimation.py --latex")

    print_success("Etapa 7 concluida")

    print_conclusion("Pipeline concluido")
    print("Resultados em: outputs/")

if __name__ == "__main__":
    main()
