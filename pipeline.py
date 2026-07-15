#!/usr/bin/env python3
"""
Pipeline orquestrador da pesquisa: executa as fases na ordem com caminhos absolutos.

Fases:
  1) Coleta bruta (World Bank ou INEP Censo Escolar)
  2) Processamento por paradigma (sql_engine, task_graph, dataframe_lib)
  3) Setup ML (folds idênticos com gaps de 2 anos; seleção de features)
  4) Baselines (um por paradigma)
  5) Hierárquicos (um por paradigma)
  6) Benchmark arquitetural (um por paradigma)
  7) Análise estatística e tabelas derivadas

Cada artefato publicado é produzido por uma etapa daqui. Um script de análise
fora deste orquestrador significa que reproduzir os resultados exige conhecer
uma sequência que não está escrita em lugar algum.
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

# Supports running from a checkout, without installing the package.
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Imported without a fallback. An empty SCIENTIFIC_CONFIG would let the whole
# experiment run on implicit defaults -- the temporal gap, the seed, the SESOI
# thresholds -- behind a warning on stdout, and record that empty configuration
# in the reproducibility snapshot. A run that cannot read its own configuration
# is not a run worth completing.
from core.config import get_execution_metadata
from core.scientific_config import SCIENTIFIC_CONFIG
from core.validation import TemporalValidator

def _log(msg: str) -> None:
    print(f"  {msg}")

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
    print(f"\nSnapshot científico registrado em {snapshot_path}")


def _discover():
    """Descoberta preguiçosa de paradigmas — importação aciona o registro."""
    from core.paradigm_registry import discover_paradigms
    paradigms = discover_paradigms()
    if not paradigms:
        raise RuntimeError("Nenhum paradigma descoberto — verifique src/architectures_ml/*/setup.py")
    return paradigms


def _validate_anti_leakage_gate(root: str, started_at: datetime) -> None:
    """Valida integridade temporal de todos os folds antes de prosseguir ao benchmark."""
    # Indexado, não .get com default: um default silencioso aqui deixaria o
    # gate validar um gap diferente do configurado.
    gap = int(SCIENTIFIC_CONFIG['temporal_gap_years'])
    embargo = int(SCIENTIFIC_CONFIG['embargo_years'])
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

        # Folds left by an earlier run would be validated in place of the ones
        # the models are about to consume, so the gate would attest to
        # artifacts that no longer exist.
        created = folds_config.get('creation_timestamp')
        if created is None:
            raise ValueError(
                f"{arch}: fold configuration carries no creation_timestamp, so "
                f"it cannot be shown to belong to this run: {folds_path}"
            )
        if datetime.fromisoformat(created) < started_at:
            raise ValueError(
                f"{arch}: fold configuration predates this run "
                f"(created {created}, run started {started_at.isoformat()}). "
                f"Stale folds must not be validated in place of current ones."
            )

        folds = folds_config.get('folds', [])
        validator.enforce_walk_forward(folds)
        _log(f"  {arch}: {len(folds)} folds — integridade temporal verificada")


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
    started_at = datetime.now()

    print(f"\nPipeline iniciado (dataset: {dataset_name})")
    _snapshot_scientific_config(root)
    paradigms = _discover()
    print("\nEtapa 0: Snapshot de reprodutibilidade")
    _log("Snapshot salvo em outputs/scientific_config_snapshot.json")

    if dataset_name == 'worldbank':
        print("\nEtapa 1/7: Coleta")
        _log("Fonte: World Bank")
        run(f"{py} {root}/src/collection/raw_data_collector.py")
    else:
        print("\nEtapa 1/7: Coleta")
        _log("Fonte: INEP Censo Escolar")
        run(f"{py} {root}/src/collection/inep_collector.py")
    _log("Etapa 1 concluida")

    n_paradigms = len(paradigms)
    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print(f"\nEtapa 2{chr(96+i)}/7: Processamento {arch}")
        _log(f"Arquitetura: {info['label']}")
        run(f"{py} {root}/{info['processor_script']}")
    _log(f"Etapa 2 concluida ({n_paradigms} paradigmas)")

    print("\nEtapa 3: Setup ML")
    _log("Gaps temporais: 2 anos (P1-P3)")
    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print(f"\nEtapa 3{chr(96+i)}/7: Setup ML {arch}")
        _log(f"Arquitetura: {info['label']}")
        run(f"{py} {root}/{info['setup_script']}")
    _log(f"Etapa 3 concluida ({n_paradigms} paradigmas)")

    print("\nGate anti-leakage")
    _validate_anti_leakage_gate(root, started_at)
    _log("Todos os folds passaram na validacao temporal")

    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print(f"\nEtapa 4{chr(96+i)}/7: Baselines {arch}")
        run(f"{py} {root}/{info['baseline_script']}")
    _log(f"Etapa 4 concluida ({n_paradigms} paradigmas)")

    print("\nEtapa 5/7: Hierarquicos")
    for arch, info in paradigms.items():
        run(f"{py} {root}/{info['hierarchical_script']}")
    _log(f"Etapa 5 concluida ({n_paradigms} paradigmas)")

    # Precedes the benchmark: a latency comparison between paradigms is only
    # meaningful once they are established to predict the same values for the
    # same rows. Running it afterwards could report a timing difference between
    # paradigms that were not doing the same work.
    print("\nGate de equivalencia de predicoes")
    run(f"{py} {root}/src/statistical_validation/prediction_equivalence.py")
    _log("Predicoes identicas entre os paradigmas")

    print("\nEtapa 6/7: Benchmark arquitetural")
    run(f"{py} {root}/src/benchmarking/architectural_benchmark.py --repetitions 10 --warmup 2")
    _log("Etapa 6 concluida")

    print("\nEtapa 7/7: Analise estatistica e tabelas derivadas")

    # Skipping the analysis on a missing benchmark would leave a run that
    # reports success while producing an incomplete set of artifacts. The
    # benchmark stage above runs with check=True, so its absence here means
    # something upstream is wrong.
    benchmark_csv = os.path.join(root, 'outputs', 'benchmarks',
                                 'architectural_benchmark_results.csv')
    if not os.path.exists(benchmark_csv):
        raise FileNotFoundError(
            f"Benchmark results absent after the benchmark stage: "
            f"{benchmark_csv}. The statistical analysis cannot be derived."
        )

    # Ordered by dependency, not by convenience:
    #   the panel consumes the latency percentiles and the resource table;
    #   the scorecard consumes significance, equivalence and the resource table.
    ANALYSIS_STAGES = [
        ('a', 'Significancia (bootstrap)',
         'src/statistical_validation/significance_tests.py', ''),
        ('b', 'Equivalencia (SESOI + IC)',
         'src/statistical_validation/equivalence_estimation.py', '--latex'),
        ('c', 'Tamanhos de efeito e comparacoes multiplas',
         'src/statistical_validation/effect_analysis.py', ''),
        ('d', 'Sensibilidade ao numero de resamples',
         'src/statistical_validation/bootstrap_sensitivity.py', ''),
        ('e', 'Percentis de latencia',
         'src/benchmarking/derive_latency_percentiles.py', ''),
        ('f', 'Percentis de throughput',
         'src/benchmarking/derive_throughput_percentiles.py', ''),
        ('g', 'Uso de recursos',
         'src/benchmarking/derive_resource_usage_table.py', ''),
        ('h', 'Painel operacional',
         'src/benchmarking/derive_operational_panel.py', ''),
        ('i', 'Scorecard',
         'src/statistical_validation/make_scorecard.py', ''),
    ]
    for suffix, description, script, script_args in ANALYSIS_STAGES:
        _log(f"Etapa 7{suffix}/7: {description}")
        run(f"{py} {root}/{script} {script_args}".rstrip())

    _log("Etapa 7 concluida")

    print("\nPipeline concluido")
    print("Resultados em: outputs/")

if __name__ == "__main__":
    main()
