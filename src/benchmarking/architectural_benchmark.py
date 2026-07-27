#!/usr/bin/env python3
"""
Architectural Benchmarking - Data Warehouse vs Data Lake

Measures per-phase pipeline latencies and throughput (records/second) for a
scientific comparison between the architectures. Implements multiple runs
with warmup, generates a CSV/JSON report and simple figures.

Supported phases:
    - collection: Raw collection with hierarchical imputation
    - processing: Architectural processing (DL and DW)
    - setup: Data preparation for ML (DL and DW)

    - baseline: Baseline models (DL and DW)
    - hierarchical: Hierarchical models (DL and DW)

Outputs:
    - outputs/benchmarks/architectural_benchmark_results.csv
    - outputs/benchmarks/architectural_benchmark_summary.json
    - outputs/benchmarks/fig_*.png

This benchmark does not change the pipeline logic. When possible, it obtains the
    number of processed records from the generated artifacts in order to
    compute throughput. In the ML steps, the count is inferred from the
    fold configuration and from the underlying datasets (DW via SQL, DL via the processing framework).
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
# Absolute paths and utilities
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
    """Dynamically import every paradigm module via the framework registry."""
    from core.paradigm_registry import discover_paradigms
    import importlib

    paradigms = discover_paradigms()
    modules = {}

    # Collection (shared, not paradigm-specific)
    from collection.raw_data_collector import RawDataCollector
    modules["RawDataCollector"] = RawDataCollector

    for name, meta in paradigms.items():
        # Setup module (has a main() function)
        setup_mod_path = meta['setup_script'].replace('/', '.')
        if setup_mod_path.endswith('.py'):
            setup_mod_path = setup_mod_path[:-3]
        if setup_mod_path.startswith('src.'):
            setup_mod_path = setup_mod_path[4:]
        modules[f"{name}_setup_module"] = importlib.import_module(setup_mod_path)

        # Processor
        proc_mod = importlib.import_module(meta['processor_module'])
        modules[f"{name}_processor_class"] = getattr(proc_mod, meta['processor_class'])
        modules[f"{name}_processor_run_method"] = meta['processor_run_method']

        # Baseline
        bl_mod = importlib.import_module(meta['baseline_module'])
        modules[f"{name}_baseline_class"] = getattr(bl_mod, meta['baseline_class'])

        # Hierarchical
        hier_mod = importlib.import_module(meta['hierarchical_module'])
        modules[f"{name}_hierarchical_class"] = getattr(hier_mod, meta['hierarchical_class'])

    modules["_paradigm_names"] = list(paradigms.keys())
    modules["_paradigm_metas"] = paradigms
    return modules


# ---------------------------------------------------------------------------
# Measurement structures
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
    Orchestrates benchmark runs per phase, measuring latencies and throughput.

    Run parameters:
        - repetitions, warmup: taken from BENCHMARK_CONFIG or the CLI
        - phases: subset of phases to run
    """

    def __init__(
        self,
        repetitions: Optional[int] = None,
        warmup_runs: Optional[int] = None,
        phases: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ):
        # `or` treats 0 as absent, the same defect already fixed in warmup.
        # Here zero is not a valid request -- there is nothing to measure -- so
        # instead of silently falling back to the default, it refuses.
        self.repetitions = (int(BENCHMARK_CONFIG.get("repetitions", 10))
                            if repetitions is None else int(repetitions))
        if self.repetitions < 1:
            raise ValueError(
                f"--repetitions={repetitions}: without repetitions there is no "
                f"measurement. Omit the parameter to use the BENCHMARK_CONFIG value."
            )
        # `or` treats 0 as absent, so --warmup 0 fell back to the default and the
        # benchmark ran warmups the operator asked it not to run.
        self.warmup_runs = (int(BENCHMARK_CONFIG.get("warmup_runs", 1))
                            if warmup_runs is None else int(warmup_runs))
        if self.warmup_runs < 0:
            raise ValueError(f"--warmup={warmup_runs}: there is no negative "
                             f"warmup.")
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

    # --------------------------- resource monitoring -----------------------
    class _ResourceMonitor:
        """Lightweight sampler of process and system resources (CPU/Mem/IO).

        Collects samples in the background while a phase runs.
        Saves an aggregated summary to the benchmark's JSONL file.
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
            self._samples = []  # list of dicts with per-sample metrics
            self._io0 = None
            self._start_ts = None
            self._end_ts = None
            self.peak_rss_mb = None

        def __enter__(self):
            try:
                self._proc.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception:
                pass  # Initializes CPU counters; failure is harmless
            # Initial IO
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
                    # Aggregate child processes (subprocesses, etc.)
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

            # Aggregate samples
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
            # IO delta (process)
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
                print(f"[WARN] Failed to write the resource log: {exc}")

    # --------------------------- counting utilities ------------------------
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

    # --------------------------- measured phases ---------------------------
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
        # A failed stage is fast: without this check the repetition enters
        # the CSV as a short, legitimate latency, pulling the paradigm's
        # distribution down. measure() was written to abort, but the failure
        # status comes inside the returned dict, not as an exception.
        if not (isinstance(res, dict) and res.get("status") == "success"):
            status = res.get("status") if isinstance(res, dict) else type(res).__name__
            raise RuntimeError(
                f"{paradigm_name}: {'processing'} returned status {status!r}; "
                f"the repetition cannot enter the latency comparison"
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
        # A failed stage is fast: without this check the repetition enters
        # the CSV as a short, legitimate latency, pulling the paradigm's
        # distribution down. measure() was written to abort, but the failure
        # status comes inside the returned dict, not as an exception.
        if not (isinstance(res, dict) and res.get("status") == "success"):
            status = res.get("status") if isinstance(res, dict) else type(res).__name__
            raise RuntimeError(
                f"{paradigm_name}: {'setup'} returned status {status!r}; "
                f"the repetition cannot enter the latency comparison"
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
        """Years present in the paradigm's master artifact.

        The location comes from PARADIGM_META, and not from a template: the SQL
        engine keeps the data in its own database and writes no master parquet, so
        the template assumed a file that never existed for it.
        """
        meta = discover_paradigms()[paradigm_name]
        artifact = meta.get('master_artifact')
        if artifact is None:
            raise KeyError(
                f"{paradigm_name} does not declare 'master_artifact' in "
                f"PARADIGM_META, so its records cannot be counted."
            )

        kind = artifact['kind']
        if kind == 'parquet':
            path = get_absolute_output_path(artifact['path'])
            if not os.path.exists(path):
                print(f"  [WARN] {paradigm_name}: master missing at {path}")
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
                print(f"  [WARN] {paradigm_name}: database missing at {database}")
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
            f"{paradigm_name}: master_artifact of unknown kind {kind!r}")

    def _count_fold_records(self, paradigm_name: str) -> Optional[int]:
        """Total records summed over a paradigm's folds.

        No `except Exception: return None`: that block returned None for any
        cause, and the silent absence left throughput unmeasured in the ML
        phases of every paradigm -- which meant the throughput percentile table
        was never generated at all.
        """
        folds_path = get_absolute_output_path(
            f"ml_pipeline/architectures/{paradigm_name}/prep/"
            f"temporal_folds_{paradigm_name}.json"
        )
        if not os.path.exists(folds_path):
            print(f"  [WARN] {paradigm_name}: folds missing at {folds_path}")
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

    # --------------------------- execution ---------------------------------
    # "Upstream" phases (collection + processing) are shared infrastructure
    # that produces the same deterministic data on every run. Repeating them
    # N times only wastes time on HTTP calls and I/O without adding
    # statistical information.
    #
    # "Downstream" phases (setup → baseline → hierarchical) contain the
    # architectural logic that differentiates DW and DL and are the benchmark's
    # real target.
    #
    # Strategy:
    #   1. Run collection + processing ONCE (using cache when possible)
    #   2. Record their times as run_id=-1 for reference
    #   3. Repeat only setup/baseline/hierarchical N times

    #: A single definition, in core.paradigm_registry: four files
    #: enumerated the same policy and one of them had already diverged.
    _DOWNSTREAM_PHASES = frozenset(COMPARABLE_PHASES)

    def run(self) -> List[PhaseResult]:
        results: List[PhaseResult] = []

        def measure(
            step_fn, phase: str, arch: str, step_name: str
        ) -> Optional[PhaseResult]:
            try:
                # Monitor resources while the phase runs
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

        # --- Phase 1: upstream (collection) - runs ONCE --------------------------
        run_id = -1
        print("Upstream: collection (single run)")

        if "collection" in self.phases:
            r = measure(
                self._phase_collection, "collection", "both", "raw_data_collector"
            )
            if r:
                results.append(r)

        # --- Phase 2: downstream - repeated N times -------------------------------
        downstream_phases = [p for p in self.phases if p in self._DOWNSTREAM_PHASES]
        if not downstream_phases:
            return results

        # Deterministic seed for randomizing the execution order.
        # Eliminates systematic cache/page-fault bias by permuting the
        # order of the 3 architectures in each iteration.
        order_rng = random.Random(42)

        # Dynamic mapping: phase → [(step_fn, arch, step_name), ...]
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

        # Append mode, and nothing truncated: a second run of the pipeline
        # stacked on top of the first, and the resource table then averaged
        # runs from different versions of the code.
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
            raise RuntimeError("No results to save")

        # Environment record alongside the results. The orchestrator writes its
        # own, but an isolated benchmark run produced none -- and a latency
        # without the environment and the core budget that produced it is not
        # comparable to another run.
        snapshot = write_environment_snapshot(
            self.output_dir,
            extra={'measured_phases': sorted(self.phases),
                   'repetitions': self.repetitions,
                   'warmup_runs': self.warmup_runs})
        print(f"  Environment recorded at {snapshot}")

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

            # Color per paradigm, derived from matplotlib's property cycle
            # in lexicographic order: stable across runs and independent of the
            # sort order, and a fourth paradigm gets a color without anyone
            # having to choose it here.
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
                ax.set_title(f"Mean time per architecture - {phase}")
                ax.set_ylabel("seconds")
                ax.set_xlabel("")
                plt.tight_layout()
                fig_path = os.path.join(self.output_dir, f"fig_{phase}_duration.png")
                plt.savefig(fig_path, dpi=120)
                plt.close(fig)
        except ImportError:
            pass

        return csv_path, summary_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Architectural Benchmark — comparison across paradigms")
    p.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Number of repetitions (excludes warmup)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup runs (not recorded)",
    )
    p.add_argument(
        "--phases",
        type=str,
        default=None,
        help="Comma-separated phases (collection,processing,setup,baseline,hierarchical)",
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
    print(f"Results saved:\n- CSV: {csv_path}\n- Summary: {summary_path}")
    # Post-processing: significance + reports
    try:
        from subprocess import run as subprocess_run

        # Significance and equivalence of baselines
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
        # Effect sizes and multiple-comparison corrections
        subprocess_run(
            [
                sys.executable,
                os.path.join(
                    PROJECT_ROOT, "src", "statistical_validation", "effect_analysis.py"
                ),
            ],
            check=True,
        )
        # Consolidated scorecard (generates outputs/statistics/architectural_scorecard.tex)
        ms = os.path.join(
            PROJECT_ROOT, "src", "statistical_validation", "make_scorecard.py"
        )
        if os.path.exists(ms):
            subprocess_run([sys.executable, ms], check=True)
    except Exception as exc:
        print(f"[ERROR] Post-benchmark analysis failed: {exc}")
        raise
