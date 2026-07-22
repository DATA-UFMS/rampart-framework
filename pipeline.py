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
import json
import os
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
from core.config import (get_absolute_output_path, get_dataset_name,
                         write_environment_snapshot)
from core.scientific_config import SCIENTIFIC_CONFIG
from core.validation import TemporalValidator

def _log(msg: str) -> None:
    print(f"  {msg}")

def deterministic_environment() -> dict:
    """Variáveis que precisam existir antes do import de NumPy.

    As bibliotecas numéricas dimensionam seus pools de threads no momento do
    carregamento, e não há como reduzi-los depois de dentro do processo — por
    isso são exportadas ao subprocesso, não definidas no orquestrador.

    PYTHONHASHSEED fixa a ordem de iteração de conjuntos e dicionários com
    chaves textuais, da qual dependem alguns caminhos de agregação.
    """
    blas = str(int(SCIENTIFIC_CONFIG['blas_threads']))
    engine = str(int(SCIENTIFIC_CONFIG['engine_threads']))
    return {
        # Componente comum aos paradigmas: todos materializam em pandas antes do
        # scikit-learn, então o pool do BLAS não é propriedade de paradigma algum.
        'OMP_NUM_THREADS': blas,
        'OPENBLAS_NUM_THREADS': blas,
        'MKL_NUM_THREADS': blas,
        'NUMEXPR_NUM_THREADS': blas,
        'VECLIB_MAXIMUM_THREADS': blas,
        # Componente do paradigma: o Polars dimensiona seu pool Rayon no import,
        # de modo que só uma variável de ambiente o alcança. O Dask lê
        # DASK_NUM_WORKERS da mesma forma, o que o alcança também nas etapas de
        # baseline e hierárquico -- elas rodam como processos separados e não
        # herdavam o dask.config.set feito na etapa de processamento, então
        # mediam com o número de núcleos do host.
        'POLARS_MAX_THREADS': engine,
        'DASK_NUM_WORKERS': engine,
        'PYTHONHASHSEED': str(int(SCIENTIFIC_CONFIG['random_seed'])),
    }


def _validate_core_budget() -> None:
    """O orçamento declarado precisa caber na máquina.

    Sobrescrever os núcleos faria a latência refletir contenção de escalonamento
    em vez do paradigma, e faria isso em silêncio.
    """
    import multiprocessing
    available = multiprocessing.cpu_count()
    engine = int(SCIENTIFIC_CONFIG['engine_threads'])
    blas = int(SCIENTIFIC_CONFIG['blas_threads'])
    if engine + blas - 1 > available:
        raise RuntimeError(
            f"Orçamento de núcleos não cabe nesta máquina: engine_threads="
            f"{engine} e blas_threads={blas}, com {available} núcleos "
            f"disponíveis. Ajuste scientific_config em vez de sobrescrever os "
            f"núcleos: a latência medida passaria a refletir contenção."
        )


def run(argv: list) -> None:
    """Executa um subprocesso com PYTHONPATH apontando para src/.

    Argumentos em lista, sem shell: um caminho de repositório com espaços
    quebraria a forma em string, e não há razão para interpretar metacaracteres
    em caminhos que este módulo mesmo constrói.
    """
    print(f"\n$ {' '.join(argv)}")
    env = os.environ.copy()
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    env.update(deterministic_environment())
    subprocess.run(argv, check=True, env=env)

def _snapshot_scientific_config(root: str) -> None:
    """Registra configuração e ambiente da execução.

    Delega a core.config: o benchmark grava o mesmo registro, e duas cópias
    divergiriam justamente no arquivo que existe para dizer como a execução foi
    feita.
    """
    path = write_environment_snapshot(get_absolute_output_path("outputs"))
    print(f"\nSnapshot científico registrado em {path}")


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

    per_paradigm: dict = {}
    for arch in _discover():
        folds_path = os.path.join(
            get_absolute_output_path('outputs/ml_pipeline/architectures'),
            arch, 'prep', f'temporal_folds_{arch}.json'
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
        per_paradigm[arch] = [
            (f['train_start'], f['train_end'], f['val_start'], f['val_end'],
             f['test_start'], f['test_end']) for f in folds
        ]

    # Cada paradigma era validado isoladamente, e nada exigia que os três
    # tivessem os mesmos folds. Splits diferentes tornam a comparação entre
    # paradigmas uma comparação entre problemas diferentes -- e o Δ=0 seria
    # falsificado por essa razão, não pela implementação.
    distinct = {arch: tuple(windows) for arch, windows in per_paradigm.items()}
    if len(set(distinct.values())) > 1:
        divergent = {arch: len(windows) for arch, windows in distinct.items()}
        raise ValueError(
            f"Os paradigmas não compartilham os mesmos folds temporais "
            f"{divergent}. A comparação entre eles pressupõe splits idênticos; "
            f"caso contrário mede problemas diferentes."
        )



def _prediction_digests() -> dict:
    """SHA-256 of every paradigm's prediction artifacts, keyed by path."""
    import hashlib

    from core.prediction_store import predictions_path

    digests = {}
    for paradigm in _discover():
        for stage in ("baseline", "hierarchical"):
            path = predictions_path(paradigm, stage)
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    digests[path] = hashlib.sha256(handle.read()).hexdigest()
    return digests


def _assert_benchmark_left_predictions_intact(before: dict) -> None:
    """The published artifacts must be the ones the gate certified.

    The equivalence gate runs before the benchmark, and the benchmark then
    re-executes setup, baseline and hierarchical `warmup + n` times per
    paradigm -- each execution overwriting the prediction artifacts. What ends
    up archived is the last repetition's output, which nothing had looked at.
    The gate attested to files that no longer existed.

    Comparing digests across the benchmark closes that, and asserts something
    the paper wants anyway: the repetitions are deterministic, so the latency
    distribution comes from runs that all produced the same predictions.
    """
    after = _prediction_digests()

    missing = sorted(set(before) - set(after))
    if missing:
        raise ValueError(
            f"O benchmark removeu artefatos de predicao que o gate havia "
            f"verificado: {missing}"
        )

    appeared = sorted(set(after) - set(before))
    if appeared:
        raise ValueError(
            f"O benchmark criou artefatos de predicao que o gate nao viu: "
            f"{appeared}. O que sera publicado nao foi verificado."
        )

    changed = sorted(path for path, digest in after.items()
                     if before[path] != digest)
    if changed:
        raise ValueError(
            f"As repeticoes do benchmark produziram predicoes diferentes das "
            f"que o gate verificou: {changed}. Ou a execucao nao e "
            f"determinista, ou os artefatos publicados nao sao os que foram "
            f"atestados -- em qualquer dos casos a afirmacao de equivalencia "
            f"nao cobre o que esta no pacote."
        )

    _log(f"  {len(after)} artefatos de predicao intactos apos o benchmark")


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

    _validate_core_budget()
    print(f"\nPipeline iniciado (dataset: {dataset_name})")
    _log(f"Orcamento: {SCIENTIFIC_CONFIG['engine_threads']} nucleos por engine, "
         f"{SCIENTIFIC_CONFIG['blas_threads']} thread(s) de BLAS")
    _snapshot_scientific_config(root)
    paradigms = _discover()
    print("\nEtapa 0: Snapshot de reprodutibilidade")
    _log(f"Snapshot salvo em {get_absolute_output_path('outputs')}")

    if dataset_name == 'worldbank':
        print("\nEtapa 1/7: Coleta")
        _log("Fonte: World Bank")
        run([py, os.path.join(root, "src/collection/raw_data_collector.py")])
    else:
        print("\nEtapa 1/7: Coleta")
        _log("Fonte: INEP Censo Escolar")
        run([py, os.path.join(root, "src/collection/inep_collector.py")])
    _log("Etapa 1 concluida")

    n_paradigms = len(paradigms)
    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print(f"\nEtapa 2{chr(96+i)}/7: Processamento {arch}")
        _log(f"Arquitetura: {info['label']}")
        run([py, os.path.join(root, info["processor_script"])])
    _log(f"Etapa 2 concluida ({n_paradigms} paradigmas)")

    print("\nEtapa 3: Setup ML")
    _log("Gaps temporais: 2 anos (P1-P3)")
    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print(f"\nEtapa 3{chr(96+i)}/7: Setup ML {arch}")
        _log(f"Arquitetura: {info['label']}")
        run([py, os.path.join(root, info["setup_script"])])
    _log(f"Etapa 3 concluida ({n_paradigms} paradigmas)")

    print("\nGate anti-leakage")
    _validate_anti_leakage_gate(root, started_at)
    _log("Todos os folds passaram na validacao temporal")

    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print(f"\nEtapa 4{chr(96+i)}/7: Baselines {arch}")
        run([py, os.path.join(root, info["baseline_script"])])
    _log(f"Etapa 4 concluida ({n_paradigms} paradigmas)")

    print("\nEtapa 5/7: Hierarquicos")
    for arch, info in paradigms.items():
        run([py, os.path.join(root, info["hierarchical_script"])])
    _log(f"Etapa 5 concluida ({n_paradigms} paradigmas)")

    # Precedes the benchmark: a latency comparison between paradigms is only
    # meaningful once they are established to predict the same values for the
    # same rows. Running it afterwards could report a timing difference between
    # paradigms that were not doing the same work.
    print("\nGate de equivalencia de predicoes")
    run([py, os.path.join(root, "src/statistical_validation/prediction_equivalence.py")])
    _log("Predicoes identicas entre os paradigmas")

    # Registrado antes do benchmark, conferido depois: as repeticoes
    # reexecutam as Etapas 3 a 5 e sobrescrevem estes mesmos arquivos.
    predictions_before = _prediction_digests()
    if not predictions_before:
        raise FileNotFoundError(
            "Nenhum artefato de predicao antes do benchmark; o gate de "
            "equivalencia nao teria o que verificar."
        )

    print("\nEtapa 6/7: Benchmark arquitetural")
    # Sem --repetitions/--warmup: o benchmark lê BENCHMARK_CONFIG, e repetir os
    # valores aqui criaria uma segunda fonte para o n do protocolo.
    run([py, os.path.join(root, "src/benchmarking/architectural_benchmark.py")])
    _assert_benchmark_left_predictions_intact(predictions_before)
    _log("Etapa 6 concluida")

    print("\nEtapa 7/7: Analise estatistica e tabelas derivadas")

    # Skipping the analysis on a missing benchmark would leave a run that
    # reports success while producing an incomplete set of artifacts. The
    # benchmark stage above runs with check=True, so its absence here means
    # something upstream is wrong.
    benchmark_csv = get_absolute_output_path(
        'outputs/benchmarks/architectural_benchmark_results.csv')
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
         'src/statistical_validation/significance_tests.py', []),
        ('b', 'Equivalencia (SESOI + IC)',
         'src/statistical_validation/equivalence_estimation.py', ['--latex']),
        ('c', 'Tamanhos de efeito e comparacoes multiplas',
         'src/statistical_validation/effect_analysis.py', []),
        ('d', 'Sensibilidade ao numero de resamples',
         'src/statistical_validation/bootstrap_sensitivity.py', []),
        ('e', 'Percentis de latencia',
         'src/benchmarking/derive_latency_percentiles.py', []),
        ('f', 'Percentis de throughput',
         'src/benchmarking/derive_throughput_percentiles.py', []),
        ('g', 'Uso de recursos',
         'src/benchmarking/derive_resource_usage_table.py', []),
        ('h', 'Painel operacional',
         'src/benchmarking/derive_operational_panel.py', []),
        ('i', 'Atribuicao do estagio (engine vs ajuste)',
         'src/benchmarking/derive_stage_attribution.py', []),
        ('j', 'Scorecard',
         'src/statistical_validation/make_scorecard.py', []),
    ]
    for suffix, description, script, script_args in ANALYSIS_STAGES:
        _log(f"Etapa 7{suffix}/7: {description}")
        run([py, os.path.join(root, script)] + script_args)

    _log("Etapa 7 concluida")

    print("\nPipeline concluido")
    print(f"Resultados em: {get_absolute_output_path('outputs')}")

if __name__ == "__main__":
    main()
