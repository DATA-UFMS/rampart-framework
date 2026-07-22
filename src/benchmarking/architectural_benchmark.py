#!/usr/bin/env python3
"""
Architectural Benchmarking - Data Warehouse vs Data Lake

Mede latências por fase do pipeline e throughput (registros/segundo) para
comparação científica entre as arquiteturas. Implementa múltiplas execuções
com warmup, gera relatório em CSV/JSON e figuras simples.

Fases suportadas:
    - collection: Coleta bruta com imputação hierárquica
    - processing: Processamento arquitetural (DL e DW)
    - setup: Preparação de dados para ML (DL e DW)

    - baseline: Modelos baseline (DL e DW)
    - hierarchical: Modelos hierárquicos (DL e DW)

Saídas:
    - outputs/benchmarks/architectural_benchmark_results.csv
    - outputs/benchmarks/architectural_benchmark_summary.json
    - outputs/benchmarks/fig_*.png

Este benchmark não altera a lógica do pipeline. Quando possível, obtém o
    número de registros processados a partir dos artefatos gerados para
    computar throughput. Nos passos de ML, a contagem é inferida da
    configuração de folds e dos datasets subjacentes (DW via SQL, DL via framework de processamento).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
import threading
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
import psutil


# ---------------------------------------------------------------------------
# Caminhos absolutos e utilitários
# ---------------------------------------------------------------------------
def _project_root() -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    while current != "/" and current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "README.md")):
            return current
        current = os.path.dirname(current)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


PROJECT_ROOT = _project_root()
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.config import (BENCHMARK_CONFIG, get_absolute_output_path,
                         write_environment_snapshot)
from core.paradigm_registry import COMPARABLE_PHASES, discover_paradigms


def _import_modules():
    """Importa dinamicamente todos os módulos de paradigma via registro do framework."""
    from core.paradigm_registry import discover_paradigms
    import importlib

    paradigms = discover_paradigms()
    modules = {}

    # Coleta (compartilhado, não específico por paradigma)
    from collection.raw_data_collector import RawDataCollector
    modules["RawDataCollector"] = RawDataCollector

    for name, meta in paradigms.items():
        # Módulo de setup (possui função main())
        setup_mod_path = meta['setup_script'].replace('/', '.')
        if setup_mod_path.endswith('.py'):
            setup_mod_path = setup_mod_path[:-3]
        if setup_mod_path.startswith('src.'):
            setup_mod_path = setup_mod_path[4:]
        modules[f"{name}_setup_module"] = importlib.import_module(setup_mod_path)

        # Processador
        proc_mod = importlib.import_module(meta['processor_module'])
        modules[f"{name}_processor_class"] = getattr(proc_mod, meta['processor_class'])
        modules[f"{name}_processor_run_method"] = meta['processor_run_method']

        # Baseline
        bl_mod = importlib.import_module(meta['baseline_module'])
        modules[f"{name}_baseline_class"] = getattr(bl_mod, meta['baseline_class'])

        # Hierárquico
        hier_mod = importlib.import_module(meta['hierarchical_module'])
        modules[f"{name}_hierarchical_class"] = getattr(hier_mod, meta['hierarchical_class'])

    modules["_paradigm_names"] = list(paradigms.keys())
    modules["_paradigm_metas"] = paradigms
    return modules


# ---------------------------------------------------------------------------
# Estruturas de medição
# ---------------------------------------------------------------------------
@dataclass
class PhaseResult:
    run_id: int
    phase: str
    architecture: str
    step: str
    duration_ns: int
    records: Optional[int]
    peak_rss_mb: Optional[float] = None

    @property
    def duration_s(self) -> float:
        # A failed phase used to reach the CSV as -1e-09, a value no consumer
        # filters: it would enter the paired vectors and the percentiles as a
        # negative latency.
        if self.duration_ns <= 0:
            raise ValueError(
                f"{self.phase}/{self.architecture}/{self.step}: non-positive "
                f"duration ({self.duration_ns} ns) has no latency to report"
            )
        return self.duration_ns / 1e9

    @property
    def throughput_rps(self) -> Optional[float]:
        if self.records is None or self.records <= 0:
            return None
        if self.duration_ns <= 0:
            return None
        return float(self.records) / self.duration_s


class BenchmarkRunner:
    """
    Orquestra execuções do benchmark por fase, medindo latências e throughput.

    Parâmetros de execução:
        - repetitions, warmup: obtidos de BENCHMARK_CONFIG ou CLI
        - phases: subconjunto de fases para execução
    """

    def __init__(
        self,
        repetitions: Optional[int] = None,
        warmup_runs: Optional[int] = None,
        phases: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ):
        self.repetitions = repetitions or int(BENCHMARK_CONFIG.get("repetitions", 10))
        # `or` trata 0 como ausente, então --warmup 0 caía no default e o
        # benchmark rodava aquecimentos que o operador pediu para não rodar.
        self.warmup_runs = (int(BENCHMARK_CONFIG.get("warmup_runs", 1))
                            if warmup_runs is None else int(warmup_runs))
        self.phases = phases or [
            "collection",
            "processing",
            "setup",
            "baseline",
            "hierarchical",
        ]
        self.modules = _import_modules()
        self.output_dir = output_dir or get_absolute_output_path("benchmarks")
        os.makedirs(self.output_dir, exist_ok=True)
        self.resource_log_path = os.path.join(
            self.output_dir, "architectural_benchmark_resource_log.jsonl"
        )

    # --------------------------- monitoramento de recursos -----------------
    class _ResourceMonitor:
        """Amostrador leve de recursos (CPU/Mem/IO) do processo e do sistema.

        Coleta amostras em segundo plano durante a execução de uma fase.
        Salva resumo agregado no arquivo JSONL do benchmark.
        """

        def __init__(
            self,
            log_path: str,
            run_id: int,
            phase: str,
            architecture: str,
            step_name: str,
            interval_s: float = 0.2,
            *,
            is_warmup: bool = False,
        ):
            self.log_path = log_path
            self.run_id = run_id
            self.is_warmup = is_warmup
            self.phase = phase
            self.architecture = architecture
            self.step_name = step_name
            self.interval_s = interval_s
            self._stop = threading.Event()
            self._thread: Optional[threading.Thread] = None
            self._proc = psutil.Process(os.getpid())
            self._samples = []  # lista de dicts com métricas por amostra
            self._io0 = None
            self._start_ts = None
            self._end_ts = None
            self.peak_rss_mb = None

        def __enter__(self):
            try:
                self._proc.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception:
                pass  # Inicializa contadores de CPU; falha é inofensiva
            # IO inicial
            try:
                self._io0 = self._proc.io_counters()
            except Exception:
                self._io0 = None
            self._start_ts = datetime.now(timezone.utc).isoformat()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return self

        def __exit__(self, exc_type, exc, tb):
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=2.0)
            self._end_ts = datetime.now(timezone.utc).isoformat()
            self._write_summary()

        def _run_loop(self):
            import time as _time

            while not self._stop.is_set():
                try:
                    cpu_p = self._proc.cpu_percent(interval=None)
                    cpu_s = psutil.cpu_percent(interval=None)
                    rss_mb = self._proc.memory_info().rss / (1024**2)
                    mem_s = psutil.virtual_memory().percent
                    # Agregar processos filhos (subprocessos, etc.)
                    child_cpu = 0.0
                    child_rss_mb = 0.0
                    try:
                        for ch in self._proc.children(recursive=True):
                            try:
                                child_cpu += ch.cpu_percent(interval=None)
                                child_rss_mb += ch.memory_info().rss / (1024**2)
                            except Exception:
                                continue
                    except Exception:
                        pass
                    sample = {
                        "cpu_proc_percent": float(cpu_p),
                        "cpu_sys_percent": float(cpu_s),
                        "rss_mb": float(rss_mb),
                        "mem_sys_percent": float(mem_s),
                        "cpu_children_percent": float(child_cpu),
                        "rss_children_mb": float(child_rss_mb),
                    }
                    self._samples.append(sample)
                except Exception:
                    pass
                self._stop.wait(self.interval_s)

        def _write_summary(self):
            import json as _json

            # Agregar amostras
            def agg(key):
                vals = [s.get(key) for s in self._samples if s.get(key) is not None]
                if not vals:
                    return {"mean": None, "max": None, "n": 0}
                arr = pd.Series(vals, dtype=float)
                return {
                    "mean": float(arr.mean()),
                    "max": float(arr.max()),
                    "n": int(arr.size),
                }

            cpu_proc = agg("cpu_proc_percent")
            cpu_sys = agg("cpu_sys_percent")
            rss = agg("rss_mb")
            mem_sys = agg("mem_sys_percent")
            cpu_children = agg("cpu_children_percent")
            rss_children = agg("rss_children_mb")
            rss_peak = rss.get("max") or 0.0
            children_peak = rss_children.get("max") or 0.0
            self.peak_rss_mb = round(rss_peak + children_peak, 1)
            # IO delta (processo)
            try:
                io1 = self._proc.io_counters()
                if self._io0:
                    read_mb = (io1.read_bytes - self._io0.read_bytes) / (1024**2)
                    write_mb = (io1.write_bytes - self._io0.write_bytes) / (1024**2)
                else:
                    read_mb = None
                    write_mb = None
            except Exception:
                read_mb = None
                write_mb = None

            rec = {
                "run_id": int(self.run_id),
                # Recorded so the table can drop them. The latency CSV already
                # excluded warmups; the resource log did not, and warmups are
                # exactly the runs with cold caches and unpaged memory -- the
                # atypical resource profile the repetitions exist to avoid.
                "is_warmup": bool(self.is_warmup),
                "phase": self.phase,
                "architecture": self.architecture,
                "step": self.step_name,
                "start_utc": self._start_ts,
                "end_utc": self._end_ts,
                "cpu_proc": cpu_proc,
                "cpu_sys": cpu_sys,
                "rss_mb": rss,
                "cpu_children": cpu_children,
                "rss_children_mb": rss_children,
                "mem_sys_percent": mem_sys,
                "io_read_mb": float(read_mb) if read_mb is not None else None,
                "io_write_mb": float(write_mb) if write_mb is not None else None,
            }
            try:
                with open(self.log_path, "a") as f:
                    f.write(_json.dumps(rec) + "\n")
            except Exception as exc:
                print(f"[WARN] Falha ao gravar log de recursos: {exc}")

    # --------------------------- utilitários de contagem -------------------
    def _count_rows_parquet(self, abs_path: str) -> Optional[int]:
        """Row count of a Parquet artifact, or None if it cannot be read.

        None and 0 are distinct: 0 means an empty artifact, None means the
        measurement is unavailable. Throughput is only derived from the former.
        """
        if not os.path.exists(abs_path):
            return None
        try:
            import pyarrow.parquet as pq
            return pq.read_metadata(abs_path).num_rows
        except Exception:
            try:
                df = pd.read_parquet(abs_path, columns=[])
                return int(len(df))
            except Exception:
                return None

    # --------------------------- fases medidas -----------------------------
    def _phase_collection(self) -> Tuple[int, Optional[int]]:
        dataset_name = os.environ.get("DATASET_NAME", "worldbank")
        raw_subdir = "collection/inep_raw" if dataset_name == "inep_censo" else "collection/raw_data"
        t0 = time.perf_counter_ns()
        if dataset_name == "inep_censo":
            from collection.inep_collector import collect_inep_data
            collect_inep_data(
                output_dir=get_absolute_output_path(raw_subdir)
            )
        else:
            Collector = self.modules["RawDataCollector"]
            collector = Collector()
            collector.run()
        t1 = time.perf_counter_ns()
        rows = self._count_rows_parquet(
            get_absolute_output_path(f"{raw_subdir}/complete_data.parquet")
        )
        return t1 - t0, rows

    def _phase_processing_generic(self, paradigm_name: str) -> Tuple[int, Optional[int]]:
        ProcessorClass = self.modules[f"{paradigm_name}_processor_class"]
        run_method_name = self.modules[f"{paradigm_name}_processor_run_method"]
        dataset_name = os.environ.get("DATASET_NAME", "worldbank")
        proc = ProcessorClass(dataset_name=dataset_name)
        t0 = time.perf_counter_ns()
        res = getattr(proc, run_method_name)()
        t1 = time.perf_counter_ns()
        # Um estágio que falhou é rápido: sem esta checagem a repetição entra
        # no CSV como uma latência curta e legítima, puxando a distribuição do
        # paradigma para baixo. measure() foi escrito para abortar, mas o status
        # de falha vem dentro do dict de retorno, não como exceção.
        if not (isinstance(res, dict) and res.get("status") == "success"):
            status = res.get("status") if isinstance(res, dict) else type(res).__name__
            raise RuntimeError(
                f"{paradigm_name}: {'processing'} retornou status {status!r}; "
                f"a repetição não pode entrar na comparação de latência"
            )
        rows = None
        if isinstance(res, dict) and res.get("status") == "success":
            out_path = res.get("output_path", "")
            if out_path:
                rows = self._count_rows_parquet(out_path)
        return t1 - t0, rows

    def _phase_setup_generic(self, paradigm_name: str) -> Tuple[int, Optional[int]]:
        setup_module = self.modules[f"{paradigm_name}_setup_module"]
        t0 = time.perf_counter_ns()
        res = setup_module.main()
        t1 = time.perf_counter_ns()
        # Um estágio que falhou é rápido: sem esta checagem a repetição entra
        # no CSV como uma latência curta e legítima, puxando a distribuição do
        # paradigma para baixo. measure() foi escrito para abortar, mas o status
        # de falha vem dentro do dict de retorno, não como exceção.
        if not (isinstance(res, dict) and res.get("status") == "success"):
            status = res.get("status") if isinstance(res, dict) else type(res).__name__
            raise RuntimeError(
                f"{paradigm_name}: {'setup'} retornou status {status!r}; "
                f"a repetição não pode entrar na comparação de latência"
            )
        rows = None
        if isinstance(res, dict) and res.get("status") == "success":
            # sql_engine keeps its data in DuckDB and materialises no master
            # Parquet, so its record count stays unavailable.
            master_path = get_absolute_output_path(
                f"ml_pipeline/architectures/{paradigm_name}/prep/master_data_{paradigm_name}.parquet"
            )
            rows = self._count_rows_parquet(master_path)
        return t1 - t0, rows

    def _phase_baseline_generic(self, paradigm_name: str) -> Tuple[int, Optional[int]]:
        AnalyzerClass = self.modules[f"{paradigm_name}_baseline_class"]
        analyzer = AnalyzerClass()
        start_ns = time.perf_counter_ns()
        analyzer.run_complete_analysis()
        end_ns = time.perf_counter_ns()
        records = self._count_fold_records(paradigm_name)
        if hasattr(analyzer, 'conn_manager'):
            analyzer.conn_manager.close_connection()
        del analyzer
        return end_ns - start_ns, records

    def _phase_hierarchical_generic(self, paradigm_name: str) -> Tuple[int, Optional[int]]:
        ModelClass = self.modules[f"{paradigm_name}_hierarchical_class"]
        model = ModelClass()
        start_ns = time.perf_counter_ns()
        model.run_hierarchical_analysis()
        end_ns = time.perf_counter_ns()
        records = self._count_fold_records(paradigm_name)
        if hasattr(model, 'conn_manager'):
            model.conn_manager.close_connection()
        del model
        return end_ns - start_ns, records

    def _fold_years(self, paradigm_name: str) -> Optional[List[int]]:
        """Anos presentes no artefato master do paradigma.

        A localização vem de PARADIGM_META, e não de um template: o engine SQL
        mantém os dados no próprio banco e não grava parquet master, então o
        template assumia um arquivo que nunca existiu para ele.
        """
        meta = discover_paradigms()[paradigm_name]
        artifact = meta.get('master_artifact')
        if artifact is None:
            raise KeyError(
                f"{paradigm_name} não declara 'master_artifact' em "
                f"PARADIGM_META, então seus registros não podem ser contados."
            )

        kind = artifact['kind']
        if kind == 'parquet':
            path = get_absolute_output_path(artifact['path'])
            if not os.path.exists(path):
                print(f"  [WARN] {paradigm_name}: master ausente em {path}")
                return None
            import pyarrow.parquet as pq
            table = pq.read_table(path, columns=["year"])
            years = table.column("year").to_pylist()
            del table
            return years

        if kind == 'duckdb_table':
            from core.config import get_dataset_name
            database = get_absolute_output_path(
                artifact['database'].format(dataset=get_dataset_name()))
            if not os.path.exists(database):
                print(f"  [WARN] {paradigm_name}: banco ausente em {database}")
                return None
            import duckdb
            conn = duckdb.connect(database, read_only=True)
            try:
                rows = conn.execute(
                    f"SELECT year FROM {artifact['table']}").fetchall()
            finally:
                conn.close()
            return [r[0] for r in rows]

        raise ValueError(
            f"{paradigm_name}: master_artifact de tipo desconhecido {kind!r}")

    def _count_fold_records(self, paradigm_name: str) -> Optional[int]:
        """Total de registros somados sobre os folds de um paradigma.

        Sem `except Exception: return None`: aquele bloco devolvia None para
        qualquer causa, e a ausência silenciosa deixou throughput sem medição nas
        fases de ML de todos os paradigmas -- o que fez a tabela de percentis de
        throughput nunca chegar a ser gerada.
        """
        folds_path = get_absolute_output_path(
            f"ml_pipeline/architectures/{paradigm_name}/prep/"
            f"temporal_folds_{paradigm_name}.json"
        )
        if not os.path.exists(folds_path):
            print(f"  [WARN] {paradigm_name}: folds ausentes em {folds_path}")
            return None
        with open(folds_path, "r") as handler:
            folds_cfg = json.load(handler)["folds"]

        years = self._fold_years(paradigm_name)
        if years is None:
            return None

        records = 0
        for fold in folds_cfg:
            for start, end in (("train_start", "train_end"),
                               ("val_start", "val_end"),
                               ("test_start", "test_end")):
                records += sum(1 for y in years if fold[start] <= y <= fold[end])
        return records

    # --------------------------- execução ----------------------------------
    # Fases "upstream" (coleta + processamento) são infraestrutura compartilhada
    # que produz os mesmos dados determinísticos em toda execução. Repetí-las
    # N vezes apenas desperdiça tempo com chamadas HTTP e I/O sem adicionar
    # informação estatística.
    #
    # Fases "downstream" (setup → baseline → hierarchical) contêm a lógica
    # arquitetural que diferencia DW e DL e são o alvo real do benchmark.
    #
    # Estratégia:
    #   1. Rodar coleta + processamento UMA vez (usando cache quando possível)
    #   2. Registrar seus tempos como run_id=-1 para referência
    #   3. Repetir apenas setup/baseline/hierarchical N vezes

    #: Uma definição só, em core.paradigm_registry: quatro arquivos
    #: enumeravam a mesma política e um deles já divergia.
    _DOWNSTREAM_PHASES = frozenset(COMPARABLE_PHASES)

    def run(self) -> List[PhaseResult]:
        results: List[PhaseResult] = []

        def measure(
            step_fn, phase: str, arch: str, step_name: str
        ) -> Optional[PhaseResult]:
            try:
                # Monitorar recursos durante a execução da fase
                mon = self._ResourceMonitor(
                    self.resource_log_path, run_id, phase, arch, step_name,
                    is_warmup=run_id < self.warmup_runs,
                )
                with mon:
                    duration_ns, records = step_fn()
                return PhaseResult(
                    run_id=run_id,
                    phase=phase,
                    architecture=arch,
                    step=step_name,
                    duration_ns=duration_ns,
                    records=records,
                    peak_rss_mb=mon.peak_rss_mb,
                )
            except Exception as exc:
                # Aborts rather than recording a sentinel. A latency comparison
                # with a missing phase is not a comparison: the paradigms stop
                # having the same paired observations, and a sentinel duration
                # would be averaged in as if it were a measurement.
                raise RuntimeError(
                    f"Benchmark phase failed and the run cannot be compared: "
                    f"{phase}/{arch}/{step_name} on repetition {run_id}"
                ) from exc

        # --- Fase 1: upstream (coleta) - executa UMA vez -------------------------
        run_id = -1
        print("Upstream: coleta (execução única)")

        if "collection" in self.phases:
            r = measure(
                self._phase_collection, "collection", "both", "raw_data_collector"
            )
            if r:
                results.append(r)

        # --- Fase 2: downstream - repetida N vezes --------------------------------
        downstream_phases = [p for p in self.phases if p in self._DOWNSTREAM_PHASES]
        if not downstream_phases:
            return results

        # Seed determinístico para randomização da ordem de execução.
        # Elimina viés sistemático de cache/page-fault ao permutar a
        # ordem das 3 arquiteturas em cada iteração.
        order_rng = random.Random(42)

        # Mapeamento dinâmico: phase → [(step_fn, arch, step_name), ...]
        paradigm_names = self.modules["_paradigm_names"]

        phase_fns = {
            "processing": [
                (lambda _pn=pn: self._phase_processing_generic(_pn), pn, "processor")
                for pn in paradigm_names
            ],
            "setup": [
                (lambda _pn=pn: self._phase_setup_generic(_pn), pn, "setup")
                for pn in paradigm_names
            ],
            "baseline": [
                (lambda _pn=pn: self._phase_baseline_generic(_pn), pn, "baseline_models")
                for pn in paradigm_names
            ],
            "hierarchical": [
                (lambda _pn=pn: self._phase_hierarchical_generic(_pn), pn, "hierarchical_models")
                for pn in paradigm_names
            ],
        }

        # Modo de append, e nada truncava: uma segunda execução do pipeline
        # empilhava sobre a primeira, e a tabela de recursos passava a
        # promediar execuções de versões diferentes do código.
        with open(self.resource_log_path, "w"):
            pass

        total_runs = self.warmup_runs + self.repetitions
        for run_id in range(total_runs):
            is_warmup = run_id < self.warmup_runs
            if is_warmup:
                print(f"Warmup run {run_id+1}/{self.warmup_runs}")
            else:
                print(
                    f"Benchmark run {run_id - self.warmup_runs + 1}/{self.repetitions}"
                )

            for phase_name in downstream_phases:
                archs = list(phase_fns[phase_name])
                order_rng.shuffle(archs)

                phase_results_run = []
                for fn, arch, step in archs:
                    r = measure(fn, phase_name, arch, step)
                    if r:
                        phase_results_run.append(r)
                    gc.collect()

                if not is_warmup:
                    results.extend(phase_results_run)

        return results

    def save_reports(self, results: List[PhaseResult]) -> Tuple[str, str]:
        if not results:
            raise RuntimeError("Sem resultados para salvar")

        # Registro de ambiente ao lado dos resultados. O orquestrador grava o
        # seu, mas uma execução do benchmark isolada não produzia nenhum -- e uma
        # latência sem o ambiente e o orçamento de núcleos que a produziram não é
        # comparável a outra execução.
        snapshot = write_environment_snapshot(
            self.output_dir,
            extra={'measured_phases': sorted(self.phases),
                   'repetitions': self.repetitions,
                   'warmup_runs': self.warmup_runs})
        print(f"  Ambiente registrado em {snapshot}")

        df = pd.DataFrame(
            [
                {
                    **asdict(r),
                    "duration_s": r.duration_s,
                    "throughput_rps": r.throughput_rps,
                }
                for r in results
            ]
        )

        csv_path = os.path.join(self.output_dir, "architectural_benchmark_results.csv")
        df.to_csv(csv_path, index=False)

        summary = (
            df.groupby(["phase", "architecture"])
            .agg(
                duration_s_mean=("duration_s", "mean"),
                duration_s_std=("duration_s", "std"),
                peak_rss_mb_mean=("peak_rss_mb", "mean"),
                peak_rss_mb_max=("peak_rss_mb", "max"),
                throughput_mean=("throughput_rps", "mean"),
                runs=("run_id", "count"),
            )
            .reset_index()
        )

        summary_path = os.path.join(
            self.output_dir, "architectural_benchmark_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)

        try:
            import matplotlib.pyplot as plt

            # Cor por paradigma, derivada do ciclo de propriedades do matplotlib
            # em ordem lexicográfica: estável entre execuções e independente da
            # ordem de sort, e um quarto paradigma recebe cor sem que ninguém
            # precise escolhê-la aqui.
            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            arch_color_map = {
                name: cycle[index % len(cycle)]
                for index, name in enumerate(sorted(discover_paradigms()))
            }

            for phase in df["phase"].unique():
                fig, ax = plt.subplots(figsize=(7, 4))
                sub = df[df["phase"] == phase]
                plot = (
                    sub.groupby("architecture")["duration_s"]
                    .mean()
                    .sort_values(ascending=False)
                )
                colors = [arch_color_map.get(a, "#999999") for a in plot.index]
                plot.plot(kind="bar", ax=ax, color=colors)
                ax.set_title(f"Tempo médio por arquitetura - {phase}")
                ax.set_ylabel("segundos")
                ax.set_xlabel("")
                plt.tight_layout()
                fig_path = os.path.join(self.output_dir, f"fig_{phase}_duration.png")
                plt.savefig(fig_path, dpi=120)
                plt.close(fig)
        except ImportError:
            pass

        return csv_path, summary_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Arquitetural — comparação entre paradigmas")
    p.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Número de repetições (exclui warmup)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Execuções de aquecimento (não registradas)",
    )
    p.add_argument(
        "--phases",
        type=str,
        default=None,
        help="Fases separadas por vírgula (collection,processing,setup,baseline,hierarchical)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    phases = [s.strip() for s in args.phases.split(",")] if args.phases else None
    runner = BenchmarkRunner(
        repetitions=args.repetitions,
        warmup_runs=args.warmup,
        phases=phases,
    )
    results = runner.run()
    csv_path, summary_path = runner.save_reports(results)
    print(f"Resultados salvos:\n- CSV: {csv_path}\n- Resumo: {summary_path}")
    # Pós-processamento: significância + relatórios
    try:
        from subprocess import run as subprocess_run

        # Significância e equivalência de baselines
        subprocess_run(
            [
                sys.executable,
                os.path.join(
                    PROJECT_ROOT,
                    "src",
                    "statistical_validation",
                    "significance_tests.py",
                ),
            ],
            check=True,
        )
        subprocess_run(
            [
                sys.executable,
                os.path.join(
                    PROJECT_ROOT, "src", "statistical_validation", "equivalence_estimation.py"
                ),
            ],
            check=True,
        )
        # Tamanhos de efeito e correções de múltiplas comparações
        subprocess_run(
            [
                sys.executable,
                os.path.join(
                    PROJECT_ROOT, "src", "statistical_validation", "effect_analysis.py"
                ),
            ],
            check=True,
        )
        # Scorecard consolidado (gera outputs/statistics/architectural_scorecard.tex)
        ms = os.path.join(
            PROJECT_ROOT, "src", "statistical_validation", "make_scorecard.py"
        )
        if os.path.exists(ms):
            subprocess_run([sys.executable, ms], check=True)
    except Exception as exc:
        print(f"[ERROR] Post-benchmark analysis failed: {exc}")
        raise
