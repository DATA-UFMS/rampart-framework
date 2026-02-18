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

Nota metodológica:
    Este benchmark não altera a lógica do pipeline. Quando possível, obtém o
    número de registros processados a partir dos artefatos gerados para
    computar throughput. Nos passos de ML, a contagem é inferida da
    configuração de folds e dos datasets subjacentes (DW via SQL, DL via Dask).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.config import get_absolute_output_path, BENCHMARK_CONFIG


# Importações tardias dos componentes do pipeline (evitar custo no import inicial)
def _import_modules():
    # Coleta
    from src.collection.raw_data_collector import RawDataCollector

    # Processamento
    from src.collection.data_lake.processor import DataLakeProcessor
    from src.collection.data_warehouse.processor import DataWarehouseProcessor
    # Setup - imports diretos dos módulos
    import src.architectures_ml.data_lake.setup as dl_setup
    import src.architectures_ml.data_warehouse.setup as dw_setup
    # Baseline
    from src.architectures_ml.data_lake.models.baseline_analysis import (
        BaselineModelAnalysisDataLake,
    )
    from src.architectures_ml.data_warehouse.models.baseline_analysis import (
        BaselineModelAnalysisDataWarehouse,
    )

    # Hierárquicos
    from src.architectures_ml.data_lake.models.hierarchical_model import (
        HierarchicalModelDataLake,
    )
    from src.architectures_ml.data_warehouse.models.hierarchical_model import (
        HierarchicalModelSQLFirst,
    )

    return {
        "RawDataCollector": RawDataCollector,
        "DataLakeProcessor": DataLakeProcessor,
        "DataWarehouseProcessor": DataWarehouseProcessor,
        "dl_setup_module": dl_setup,
        "dw_setup_module": dw_setup,
        "BaselineModelAnalysisDataLake": BaselineModelAnalysisDataLake,
        "BaselineModelAnalysisDataWarehouse": BaselineModelAnalysisDataWarehouse,
        "HierarchicalModelDataLake": HierarchicalModelDataLake,
        "HierarchicalModelSQLFirst": HierarchicalModelSQLFirst,
    }


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

    @property
    def duration_s(self) -> float:
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
        self.repetitions = repetitions or int(BENCHMARK_CONFIG.get("repetitions", 3))
        self.warmup_runs = warmup_runs or int(BENCHMARK_CONFIG.get("warmup_runs", 1))
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
        # Arquivo de log de recursos (JSONL)
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
        ):
            self.log_path = log_path
            self.run_id = run_id
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

        def __enter__(self):
            try:
                self._proc.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
            # IO inicial
            try:
                self._io0 = self._proc.io_counters()
            except Exception:
                self._io0 = None
            self._start_ts = datetime.utcnow().isoformat()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return self

        def __exit__(self, exc_type, exc, tb):
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=2.0)
            self._end_ts = datetime.utcnow().isoformat()
            self._write_summary()

        def _run_loop(self):
            import time as _time

            while not self._stop.is_set():
                try:
                    cpu_p = self._proc.cpu_percent(interval=None)
                    cpu_s = psutil.cpu_percent(interval=None)
                    rss_mb = self._proc.memory_info().rss / (1024**2)
                    mem_s = psutil.virtual_memory().percent
                    # Agregar processos filhos (Dask workers, subprocessos, etc.)
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
            except Exception:
                pass

    # --------------------------- utilitários de contagem -------------------
    def _count_rows_parquet(self, abs_path: str) -> int:
        if not os.path.exists(abs_path):
            return 0
        try:
            df = pd.read_parquet(abs_path)
            return int(len(df))
        except Exception:
            return 0

    def _dl_master_path(self) -> str:
        return get_absolute_output_path(
            "ml_pipeline/architectures/data_lake/prep/master_data_data_lake.parquet"
        )

    def _dl_folds_path(self) -> str:
        return get_absolute_output_path(
            "ml_pipeline/architectures/data_lake/prep/temporal_folds_data_lake.json"
        )

    def _dw_db_path(self) -> str:
        return get_absolute_output_path(
            "collection/data_warehouse/worldbank_data.duckdb"
        )

    def _count_dl_split_rows(self, start: int, end: int) -> int:
        import dask.dataframe as dd  # lazy import

        master = self._dl_master_path()
        if not os.path.exists(master):
            return 0
        ddf = dd.read_parquet(master)
        try:
            # Contar apenas registros com target não-nulo para comparabilidade com DW
            filt = (
                (ddf["year"] >= start)
                & (ddf["year"] <= end)
                & (
                    ~ddf["dropout_rate_data_lake"].isna()
                    if "dropout_rate_data_lake" in ddf.columns
                    else True
                )
            )
            return int(ddf[filt].map_partitions(len).sum().compute())
        except Exception:
            return 0

    def _count_dw_split_rows(
        self, start: int, end: int, only_target: bool = False
    ) -> int:
        # evita dependência direta; usa connection_manager do DW
        try:
            from src.collection.data_warehouse.connection_manager import DuckDBConnectionManager
        except Exception:
            return 0
        db = self._dw_db_path()
        if not os.path.exists(db):
            return 0
        conn = DuckDBConnectionManager(db)
        try:
            cond = f"year >= {start} AND year <= {end}"
            if only_target:
                cond += " AND dropout_rate_data_warehouse IS NOT NULL"
            return int(
                conn.execute_scalar(f"SELECT COUNT(*) FROM analytics_wide WHERE {cond}")
            )
        finally:
            try:
                conn.close_connection()
            except Exception:
                pass

    # --------------------------- fases medidas -----------------------------
    def _phase_collection(self) -> Tuple[int, Optional[int]]:
        Collector = self.modules["RawDataCollector"]
        collector = Collector()
        t0 = time.perf_counter_ns()
        ok = collector.run()
        t1 = time.perf_counter_ns()
        rows = None
        if ok:
            rows = self._count_rows_parquet(
                get_absolute_output_path("collection/raw_data/complete_data.parquet")
            )
        return t1 - t0, rows

    def _phase_processing_dl(self) -> Tuple[int, Optional[int]]:
        Processor = self.modules["DataLakeProcessor"]
        proc = Processor()
        t0 = time.perf_counter_ns()
        res = proc.run_data_lake_processing()
        t1 = time.perf_counter_ns()
        rows = None
        if isinstance(res, dict) and res.get("status") == "success":
            # Data Lake retorna artefatos em res['output']
            out = res.get("output", {}) if isinstance(res.get("output"), dict) else {}
            path = (
                out.get("primary_dataset")
                or out.get("partitioned_dataset")
                or res.get("output_path", "")
            )
            rows = self._count_rows_parquet(path)
        return t1 - t0, rows

    def _phase_processing_dw(self) -> Tuple[int, Optional[int]]:
        Processor = self.modules["DataWarehouseProcessor"]
        proc = Processor()
        t0 = time.perf_counter_ns()
        res = proc.run_data_warehouse_processing()
        t1 = time.perf_counter_ns()
        rows = None
        if isinstance(res, dict) and res.get("status") == "success":
            rows = self._count_rows_parquet(res.get("output_path", ""))
        return t1 - t0, rows

    def _phase_setup_dl(self) -> Tuple[int, Optional[int]]:
        setup_module = self.modules["dl_setup_module"]
        t0 = time.perf_counter_ns()
        # Chamar a função main() do módulo
        res = setup_module.main()
        t1 = time.perf_counter_ns()
        rows = None
        if isinstance(res, dict) and res.get("status") == "success":
            # Contar observações do master Data Lake preparado
            rows = self._count_rows_parquet(self._dl_master_path())
        return t1 - t0, rows

    def _phase_setup_dw(self) -> Tuple[int, Optional[int]]:
        setup_module = self.modules["dw_setup_module"]
        t0 = time.perf_counter_ns()
        # Chamar a função main() do módulo
        res = setup_module.main()
        t1 = time.perf_counter_ns()
        rows = None
        if isinstance(res, dict) and res.get("status") == "success":
            # Contar observações diretamente no banco (analytics_wide)
            try:
                from src.collection.data_warehouse.connection_manager import DuckDBConnectionManager

                db = self._dw_db_path()
                if os.path.exists(db):
                    conn = DuckDBConnectionManager(db)
                    try:
                        rows = int(
                            conn.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
                        )
                    finally:
                        try:
                            conn.close_connection()
                        except Exception:
                            pass
            except Exception:
                rows = None
        return t1 - t0, rows

    def _phase_baseline_dl(self) -> Tuple[int, Optional[int]]:
        # estimar registros usados pelos folds via master + folds
        start_ns = time.perf_counter_ns()
        Analyzer = self.modules["BaselineModelAnalysisDataLake"]
        analyzer = Analyzer()
        analyzer.run_complete_analysis()
        end_ns = time.perf_counter_ns()

        records = 0
        try:
            with open(self._dl_folds_path(), "r") as f:
                folds_cfg = json.load(f)["folds"]
            for fold in folds_cfg:
                records += self._count_dl_split_rows(
                    fold["train_start"], fold["train_end"]
                )
                records += self._count_dl_split_rows(fold["val_start"], fold["val_end"])
                records += self._count_dl_split_rows(
                    fold["test_start"], fold["test_end"]
                )
        except Exception:
            records = None

        return end_ns - start_ns, records

    def _phase_baseline_dw(self) -> Tuple[int, Optional[int]]:
        start_ns = time.perf_counter_ns()
        Analyzer = self.modules["BaselineModelAnalysisDataWarehouse"]
        analyzer = Analyzer()
        analyzer.run_complete_analysis()
        end_ns = time.perf_counter_ns()

        records = 0
        try:
            folds_path = get_absolute_output_path(
                "ml_pipeline/architectures/data_warehouse/prep/temporal_folds_data_warehouse.json"
            )
            with open(folds_path, "r") as f:
                folds_cfg = json.load(f)["folds"]
            for fold in folds_cfg:
                records += self._count_dw_split_rows(
                    fold["train_start"], fold["train_end"], only_target=True
                )
                records += self._count_dw_split_rows(
                    fold["val_start"], fold["val_end"], only_target=True
                )
                records += self._count_dw_split_rows(
                    fold["test_start"], fold["test_end"], only_target=True
                )
        except Exception:
            records = None

        return end_ns - start_ns, records

    def _phase_hierarchical_dl(self) -> Tuple[int, Optional[int]]:
        start_ns = time.perf_counter_ns()
        Model = self.modules["HierarchicalModelDataLake"]
        model = Model(use_enhanced_features=False)
        _ = model.run_hierarchical_analysis()
        end_ns = time.perf_counter_ns()
        # reusar mesma estimativa de contagem dos folds do baseline
        records = 0
        try:
            with open(self._dl_folds_path(), "r") as f:
                folds_cfg = json.load(f)["folds"]
            for fold in folds_cfg:
                records += self._count_dl_split_rows(
                    fold["train_start"], fold["train_end"]
                )
                records += self._count_dl_split_rows(fold["val_start"], fold["val_end"])
                records += self._count_dl_split_rows(
                    fold["test_start"], fold["test_end"]
                )
        except Exception:
            records = None
        return end_ns - start_ns, records

    def _phase_hierarchical_dw(self) -> Tuple[int, Optional[int]]:
        start_ns = time.perf_counter_ns()
        Model = self.modules["HierarchicalModelSQLFirst"]
        model = Model(use_enhanced_features=False)
        _ = model.run_hierarchical_analysis()
        end_ns = time.perf_counter_ns()
        records = 0
        try:
            folds_path = get_absolute_output_path(
                "ml_pipeline/architectures/data_warehouse/prep/temporal_folds_data_warehouse.json"
            )
            with open(folds_path, "r") as f:
                folds_cfg = json.load(f)["folds"]
            for fold in folds_cfg:
                records += self._count_dw_split_rows(
                    fold["train_start"], fold["train_end"], only_target=True
                )
                records += self._count_dw_split_rows(
                    fold["val_start"], fold["val_end"], only_target=True
                )
                records += self._count_dw_split_rows(
                    fold["test_start"], fold["test_end"], only_target=True
                )
        except Exception:
            records = None
        return end_ns - start_ns, records

    # --------------------------- execução ----------------------------------
    def run(self) -> List[PhaseResult]:
        results: List[PhaseResult] = []

        def measure(
            step_fn, phase: str, arch: str, step_name: str
        ) -> Optional[PhaseResult]:
            try:
                # Monitorar recursos durante a execução da fase
                with self._ResourceMonitor(
                    self.resource_log_path, run_id, phase, arch, step_name
                ):
                    duration_ns, records = step_fn()
                return PhaseResult(
                    run_id=run_id,
                    phase=phase,
                    architecture=arch,
                    step=step_name,
                    duration_ns=duration_ns,
                    records=records,
                )
            except Exception:
                # registrar falha com duração nula
                return PhaseResult(
                    run_id=run_id,
                    phase=phase,
                    architecture=arch,
                    step=step_name,
                    duration_ns=0,
                    records=None,
                )

        total_runs = self.warmup_runs + self.repetitions
        for run_id in range(total_runs):
            is_warmup = run_id < self.warmup_runs
            if is_warmup:
                print(f"Warmup run {run_id+1}/{self.warmup_runs}")
            else:
                print(
                    f"Benchmark run {run_id - self.warmup_runs + 1}/{self.repetitions}"
                )

            if "collection" in self.phases:
                r = measure(
                    self._phase_collection, "collection", "both", "raw_data_collector"
                )
                if r and not is_warmup:
                    results.append(r)

            if "processing" in self.phases:
                r1 = measure(
                    self._phase_processing_dl, "processing", "data_lake", "processor"
                )
                r2 = measure(
                    self._phase_processing_dw,
                    "processing",
                    "data_warehouse",
                    "processor",
                )
                if not is_warmup:
                    if r1:
                        results.append(r1)
                    if r2:
                        results.append(r2)

            if "setup" in self.phases:
                r1 = measure(self._phase_setup_dl, "setup", "data_lake", "setup")
                r2 = measure(self._phase_setup_dw, "setup", "data_warehouse", "setup")
                if not is_warmup:
                    if r1:
                        results.append(r1)
                    if r2:
                        results.append(r2)

            if "baseline" in self.phases:
                r1 = measure(
                    self._phase_baseline_dl, "baseline", "data_lake", "baseline_models"
                )
                r2 = measure(
                    self._phase_baseline_dw,
                    "baseline",
                    "data_warehouse",
                    "baseline_models",
                )
                if not is_warmup:
                    if r1:
                        results.append(r1)
                    if r2:
                        results.append(r2)

            if "hierarchical" in self.phases:
                r1 = measure(
                    self._phase_hierarchical_dl,
                    "hierarchical",
                    "data_lake",
                    "hierarchical_models",
                )
                r2 = measure(
                    self._phase_hierarchical_dw,
                    "hierarchical",
                    "data_warehouse",
                    "hierarchical_models",
                )
                if not is_warmup:
                    if r1:
                        results.append(r1)
                    if r2:
                        results.append(r2)

        return results

    def save_reports(self, results: List[PhaseResult]) -> Tuple[str, str]:
        if not results:
            raise RuntimeError("Sem resultados para salvar")

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

            for phase in df["phase"].unique():
                fig, ax = plt.subplots(figsize=(7, 4))
                sub = df[df["phase"] == phase]
                plot = (
                    sub.groupby("architecture")["duration_s"]
                    .mean()
                    .sort_values(ascending=False)
                )
                plot.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
                ax.set_title(f"Tempo médio por arquitetura - {phase}")
                ax.set_ylabel("segundos")
                ax.set_xlabel("")
                plt.tight_layout()
                fig_path = os.path.join(self.output_dir, f"fig_{phase}_duration.png")
                plt.savefig(fig_path, dpi=120)
                plt.close(fig)
        except Exception:
            # Apenas ignore falhas de plotting
            pass

        return csv_path, summary_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Architectural Benchmark - DW vs DL")
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
        from subprocess import run

        # Significância e equivalência de baselines
        run(
            [
                sys.executable,
                os.path.join(
                    PROJECT_ROOT,
                    "src",
                    "statistical_validation",
                    "significance_tests.py",
                ),
            ],
            check=False,
        )
        run(
            [
                sys.executable,
                os.path.join(
                    PROJECT_ROOT, "src", "statistical_validation", "tost_baseline.py"
                ),
            ],
            check=False,
        )
        # Tamanhos de efeito e correções de múltiplas comparações
        run(
            [
                sys.executable,
                os.path.join(
                    PROJECT_ROOT, "src", "statistical_validation", "effect_analysis.py"
                ),
            ],
            check=False,
        )
        # Scorecard consolidado (gera outputs/statistics/architectural_scorecard.tex)
        ms = os.path.join(
            PROJECT_ROOT, "src", "statistical_validation", "make_scorecard.py"
        )
        if os.path.exists(ms):
            run([sys.executable, ms], check=False)
    except Exception:
        pass
