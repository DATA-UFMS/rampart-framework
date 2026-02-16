#!/usr/bin/env python3
"""
Pipeline orquestrador da pesquisa: executa as fases na ordem com caminhos absolutos.

Fases:
  1) Coleta bruta (World Bank)
  2) Processamento arquitetural (Data Lake e Data Warehouse)
  3) Setup ML (folds idênticos com gaps de 2 anos; seleção de features)
  4) Feature engineering (opcional)
  5) Baselines
  6) Hierárquicos (modo básico - para usar features enhanced, execute manualmente com --enhanced)
  7) Benchmark arquitetural
  8) Testes estatísticos de validação
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

try:
    from src.core.scientific_config import SCIENTIFIC_CONFIG  # type: ignore
except Exception:
    SCIENTIFIC_CONFIG = {}

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

    snapshot_path = os.path.join(snapshot_dir, "scientific_config_snapshot.json")
    with open(snapshot_path, "w", encoding="utf-8") as handler:
        json.dump(payload, handler, indent=2, ensure_ascii=False)
    print_system(f"Snapshot científico registrado em {snapshot_path}")


def main() -> None:
    # CLI básica (sem modos de folds; geração é sempre automática via config científica)
    parser = argparse.ArgumentParser(description="Pipeline de pesquisa - DW vs DL")
    _ = parser.parse_args()
    root = os.path.abspath(os.path.dirname(__file__))
    py = sys.executable

    print_conclusion("INICIANDO PIPELINE METODOLÓGICO COMPLETO (QP1–QP3)")
    _snapshot_scientific_config(root)
    print_system("PROTOCOLO 4/4 — REPRODUTIBILIDADE ATIVADA")
    print_config("Snapshot de configuração e ambiente salvo em outputs/scientific_config_snapshot.json")

    # 1) Coleta
    print_system("PROCESSADOR DE DADOS BRUTOS")
    print_config("Fonte: Banco Mundial (World Bank)")
    print_step("ETAPA 1/8: Coletando dados brutos...")
    run(f"{py} {root}/src/collection/raw_data_collector.py")
    print_success("ETAPA 1 CONCLUÍDA: Dados brutos coletados")

    # 2) Processamento arquitetural
    print_system("PROCESSADOR DATA LAKE")
    print_config("Arquitetura: Data Lake com Dask")
    print_step("ETAPA 2a/8: Processando Data Lake...")
    run(f"{py} {root}/src/collection/data_lake/processor.py")

    print_system("PROCESSADOR DATA WAREHOUSE")
    print_config("Arquitetura: Data Warehouse com DuckDB")
    print_step("ETAPA 2b/8: Processando Data Warehouse...")
    run(f"{py} {root}/src/collection/data_warehouse/processor.py")
    print_success("ETAPA 2 CONCLUÍDA: Processamento arquitetural completo")

    print_system("PROTOCOLO 1/4 — VALIDAÇÃO TEMPORAL (QP1)")
    print_config("Gaps temporais: 2 anos (anti-leak)")
    # 3) Setup ML (gaps 2 anos em ambas arquiteturas)
    print_system("SETUP ML DATA LAKE")
    print_config("Arquitetura: Data Lake (schema-on-read)")
    print_step("ETAPA 3a/8: Configurando ML Data Lake...")
    run(f"{py} {root}/src/architectures_ml/data_lake/setup.py")

    print_system("SETUP ML DATA WAREHOUSE")
    print_config("Arquitetura: Data Warehouse (schema-on-write)")
    print_step("ETAPA 3b/8: Configurando ML Data Warehouse...")
    run(f"{py} {root}/src/architectures_ml/data_warehouse/setup.py")
    print_success("ETAPA 3 CONCLUÍDA: Setup ML completo")

    # 4) Feature engineering (opcional); não interromper se falhar
    print_system("FEATURE ENGINEERING")
    print_config("Modo: Opcional (pode falhar)")
    print_step("ETAPA 4/8: Executando Feature Engineering...")
    try:
        run(f"{py} {root}/src/architectures_ml/data_lake/feature_engineering.py")
    except Exception as e:
        print_error(f"Feature engineering DL opcional falhou: {e}")
    try:
        run(f"{py} {root}/src/architectures_ml/data_warehouse/feature_engineering.py")
    except Exception as e:
        print_error(f"Feature engineering DW opcional falhou: {e}")
    print_success("ETAPA 4 CONCLUÍDA: Feature engineering executado")

    # 5) Baselines
    print_system("MODELOS BASELINE DATA LAKE")
    print_config("Modelos: Média, Tendência, Naive, Cross-Country")
    print_step("ETAPA 5a/8: Executando Baselines Data Lake...")
    run(f"{py} {root}/src/architectures_ml/data_lake/models/baseline_analysis.py")

    print_system("MODELOS BASELINE DATA WAREHOUSE")
    print_config("Modelos: Média, Tendência, Naive, Cross-Country")
    print_step("ETAPA 5b/8: Executando Baselines Data Warehouse...")
    run(f"{py} {root}/src/architectures_ml/data_warehouse/models/baseline_analysis.py")
    print_success("ETAPA 5 CONCLUÍDA: Modelos baseline executados")

    # 6) Hierárquicos (modo básico)
    print_system("MODELOS HIERÁRQUICOS")
    print_config("Modo: Básico (sem features enhanced)")
    print_step("ETAPA 6/8: Executando Modelos Hierárquicos...")
    print("Para usar features enhanced do feature engineering, execute manualmente:")
    print("   python src/.../hierarchical_model.py --enhanced")
    run(f"{py} {root}/src/architectures_ml/data_lake/models/hierarchical_model.py")
    run(f"{py} {root}/src/architectures_ml/data_warehouse/models/hierarchical_model.py")
    print_success("ETAPA 6 CONCLUÍDA: Modelos hierárquicos executados")

    # 7) Benchmark arquitetural
    print_system("PROTOCOLO 3/4 — BENCHMARK ARQUITETURAL (QP3)")
    print_config("Comparação demonstrativa: schema-on-write vs schema-on-read")
    print_step("ETAPA 7/8: Executando Benchmark Arquitetural...")
    run(f"{py} {root}/src/benchmarking/architectural_benchmark.py --repetitions 5 --warmup 1")
    print_success("ETAPA 7 CONCLUÍDA: Benchmark arquitetural executado")

    # 8) Testes estatísticos de validação
    print_system("PROTOCOLO 2/4 — EQUIVALÊNCIA PRÁTICA (QP2)")
    print_config("Validação: SESOI + IC95% com bootstrap e estatísticas robustas")
    print_step("ETAPA 8/8: Executando Testes Estatísticos...")
    
    # 8a) Testes de significância com bootstrap (se dados de benchmark existirem)
    benchmark_csv = f"{root}/outputs/benchmarks/architectural_benchmark_results.csv"
    if os.path.exists(benchmark_csv):
        print_step("ETAPA 8a/8: Testes de significância (bootstrap)...")
        run(f"{py} {root}/src/statistical_validation/significance_tests.py")
    else:
        print_error("Arquivo de benchmark não encontrado, pulando testes de significância")
    
    # 8b) Equivalência por estimativa (SESOI + IC) (sempre executa para gerar estrutura)
    print_step("ETAPA 8b/8: Equivalência por estimativa (SESOI + IC)...")
    run(f"{py} {root}/src/statistical_validation/tost_baseline.py --latex")
    
    print_success("ETAPA 8 CONCLUÍDA: Testes estatísticos executados")

    print_conclusion("PIPELINE METODOLÓGICO EXECUTADO COM SUCESSO")
    print("Resultados disponíveis em: outputs/ (ver checklist no README.md)")

if __name__ == "__main__":
    main()
