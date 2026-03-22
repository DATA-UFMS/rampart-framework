#!/usr/bin/env python3
"""
Pipeline orquestrador da pesquisa: executa as fases na ordem com caminhos absolutos.

Fases:
  1) Coleta bruta (World Bank)
  2) Processamento arquitetural (Data Lake e Data Warehouse)
  3) Setup ML (folds idênticos com gaps de 2 anos; seleção de features)
  4) Feature engineering (opcional)
  5) Baselines
  6) Hierárquicos
  7) Benchmark arquitetural
  8) Testes estatísticos de validação
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
except Exception:
    SCIENTIFIC_CONFIG = {}

try:
    from src.core.config import get_execution_metadata
except Exception:
    get_execution_metadata = None

try:
    from src.core.validation import TemporalValidator
except Exception:
    TemporalValidator = None

def print_conclusion(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def print_system(msg: str) -> None:
    print(f"\n[SISTEMA] {msg}")

def print_config(msg: str) -> None:
    print(f"[CONFIG] {msg}")

def print_step(msg: str) -> None:
    print(f"[ETAPA] {msg}")

def print_success(msg: str) -> None:
    print(f"[SUCESSO] {msg}")

def print_error(msg: str) -> None:
    print(f"[ERRO] {msg}")

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

    # Versões dos pacotes instalados
    try:
        payload["installed_packages"] = {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        }
    except Exception:
        payload["installed_packages"] = "unavailable"

    # Informações de hardware (reutiliza get_execution_metadata de config.py)
    if get_execution_metadata is not None:
        try:
            payload["hardware"] = get_execution_metadata()
        except Exception:
            payload["hardware"] = "unavailable"

    # Hash do requirements.txt para detecção de drift
    req_path = os.path.join(root, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "rb") as f:
            payload["requirements_txt_sha256"] = hashlib.sha256(f.read()).hexdigest()

    snapshot_path = os.path.join(snapshot_dir, "scientific_config_snapshot.json")
    with open(snapshot_path, "w", encoding="utf-8") as handler:
        json.dump(payload, handler, indent=2, ensure_ascii=False)
    print_system(f"Snapshot científico registrado em {snapshot_path}")


def _validate_anti_leakage_gate(root: str) -> None:
    """Valida integridade temporal de todos os folds antes de prosseguir ao benchmark."""
    if TemporalValidator is None:
        raise RuntimeError("TemporalValidator não disponível — validação anti-leakage impossível")

    gap = int(SCIENTIFIC_CONFIG.get('temporal_gap_years', 2))
    embargo = int(SCIENTIFIC_CONFIG.get('embargo_years', 0))
    validator = TemporalValidator(min_gap_years=gap, embargo_years=embargo)

    for arch in ['data_lake', 'data_warehouse']:
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
    # CLI básica (sem modos de folds; geração é sempre automática via config científica)
    parser = argparse.ArgumentParser(description="Pipeline de pesquisa - DW vs DL")
    _ = parser.parse_args()
    root = os.path.abspath(os.path.dirname(__file__))
    py = sys.executable

    print_conclusion("INICIANDO PIPELINE METODOLÓGICO COMPLETO (RQ1–RQ3)")
    _snapshot_scientific_config(root)
    print_system("ETAPA 0 — SNAPSHOT DE REPRODUTIBILIDADE (P3: separação)")
    print_config("Snapshot de configuração e ambiente salvo em outputs/scientific_config_snapshot.json")

    # Coleta
    print_system("PROCESSADOR DE DADOS BRUTOS")
    print_config("Fonte: Banco Mundial (World Bank)")
    print_step("ETAPA 1/7: Coletando dados brutos...")
    run(f"{py} {root}/src/collection/raw_data_collector.py")
    print_success("ETAPA 1 CONCLUÍDA: Dados brutos coletados")

    # Processamento arquitetural
    print_system("PROCESSADOR DATA LAKE")
    print_config("Arquitetura: Data Lake com Dask")
    print_step("ETAPA 2a/7: Processando Data Lake...")
    run(f"{py} {root}/src/collection/data_lake/processor.py")

    print_system("PROCESSADOR DATA WAREHOUSE")
    print_config("Arquitetura: Data Warehouse com DuckDB")
    print_step("ETAPA 2b/7: Processando Data Warehouse...")
    run(f"{py} {root}/src/collection/data_warehouse/processor.py")
    print_success("ETAPA 2 CONCLUÍDA: Processamento arquitetural completo")

    print_system("PROTOCOLO ANTI-LEAKAGE — VALIDAÇÃO TEMPORAL (RQ1)")
    print_config("Gaps temporais: 2 anos (P1: ordenação, P2: gap, P3: separação)")
    # Setup ML (gaps 2 anos em ambas arquiteturas)
    print_system("SETUP ML DATA LAKE")
    print_config("Arquitetura: Data Lake (schema-on-read)")
    print_step("ETAPA 3a/7: Configurando ML Data Lake...")
    run(f"{py} {root}/src/architectures_ml/data_lake/setup.py")

    print_system("SETUP ML DATA WAREHOUSE")
    print_config("Arquitetura: Data Warehouse (schema-on-write)")
    print_step("ETAPA 3b/7: Configurando ML Data Warehouse...")
    run(f"{py} {root}/src/architectures_ml/data_warehouse/setup.py")
    print_success("ETAPA 3 CONCLUÍDA: Setup ML completo")

    # Gate anti-leakage: interromper pipeline se integridade temporal for violada
    print_system("GATE ANTI-LEAKAGE")
    print_step("Verificando integridade temporal de todos os folds...")
    _validate_anti_leakage_gate(root)
    print_success("GATE ANTI-LEAKAGE: Todos os folds passaram na validação")

    # Baselines
    print_system("MODELOS BASELINE DATA LAKE")
    print_config("Modelos: Média, Tendência, Naive, Cross-Country")
    print_step("ETAPA 4a/7: Executando Baselines Data Lake...")
    run(f"{py} {root}/src/architectures_ml/data_lake/models/baseline_analysis.py")

    print_system("MODELOS BASELINE DATA WAREHOUSE")
    print_config("Modelos: Média, Tendência, Naive, Cross-Country")
    print_step("ETAPA 4b/7: Executando Baselines Data Warehouse...")
    run(f"{py} {root}/src/architectures_ml/data_warehouse/models/baseline_analysis.py")
    print_success("ETAPA 4 CONCLUÍDA: Modelos baseline executados")

    # Hierárquicos
    print_system("MODELOS HIERÁRQUICOS")
    print_step("ETAPA 5/7: Executando Modelos Hierárquicos...")
    run(f"{py} {root}/src/architectures_ml/data_lake/models/hierarchical_model.py")
    run(f"{py} {root}/src/architectures_ml/data_warehouse/models/hierarchical_model.py")
    print_success("ETAPA 5 CONCLUÍDA: Modelos hierárquicos executados")

    # Benchmark arquitetural
    print_system("BENCHMARK ARQUITETURAL (RQ3)")
    print_config("Comparação demonstrativa: schema-on-write vs schema-on-read")
    print_step("ETAPA 6/7: Executando Benchmark Arquitetural...")
    run(f"{py} {root}/src/benchmarking/architectural_benchmark.py --repetitions 35 --warmup 2")
    print_success("ETAPA 6 CONCLUÍDA: Benchmark arquitetural executado")

    # Testes estatísticos de validação
    print_system("EQUIVALÊNCIA PRÁTICA (RQ2)")
    print_config("Validação: SESOI + IC95% com bootstrap e estatísticas robustas")
    print_step("ETAPA 7/7: Executando Testes Estatísticos...")
    
    # Testes de significância com bootstrap (se dados de benchmark existirem)
    benchmark_csv = f"{root}/outputs/benchmarks/architectural_benchmark_results.csv"
    if os.path.exists(benchmark_csv):
        print_step("ETAPA 7a/7: Testes de significância (bootstrap)...")
        run(f"{py} {root}/src/statistical_validation/significance_tests.py")
    else:
        print_error("Arquivo de benchmark não encontrado, pulando testes de significância")
    
    # Equivalência por estimativa (SESOI + IC) (sempre executa para gerar estrutura)
    print_step("ETAPA 7b/7: Equivalência por estimativa (SESOI + IC)...")
    run(f"{py} {root}/src/statistical_validation/tost_baseline.py --latex")
    
    print_success("ETAPA 7 CONCLUÍDA: Testes estatísticos executados")

    print_conclusion("PIPELINE METODOLÓGICO EXECUTADO COM SUCESSO")
    print("Resultados disponíveis em: outputs/ (ver checklist no README.md)")

if __name__ == "__main__":
    main()
