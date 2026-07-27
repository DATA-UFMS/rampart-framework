# Usage Guide

Instructions to run, verify and adapt the framework. Complements the [README](README.md).

## Environment Setup

Requirements: Python 3.12, 8 GB RAM, 10 GB disk, internet access.

```bash
git clone https://github.com/DATA-UFMS/rampart-framework.git
cd rampart-framework
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Execution

**Full pipeline** (hours; the benchmark re-runs the phases `warmup + n` times):

```bash
python pipeline.py
```

Runs, in order: reproducibility snapshot, collection, processing (3 paradigms),
ML setup (3 paradigms), the setup provenance and anti-leakage gates, baselines
(3 paradigms), hierarchical models (3 paradigms), the protocol receipts and
prediction equivalence gates, the architectural benchmark, and the statistical
analysis with its derived tables.

**Individual components:**

```bash
# Benchmark with the experimental protocol (n=10, 2 warmups)
python src/benchmarking/architectural_benchmark.py --repetitions 10 --warmup 2

# Shorter, exploratory run (does NOT reproduce the latency table)
python src/benchmarking/architectural_benchmark.py --repetitions 5 --warmup 1

# Equivalence (generates JSON + LaTeX)
python src/statistical_validation/equivalence_estimation.py --latex

# Tests
pytest tests/

# Negative validation of the gate (scenarios S1-S4)
python scripts/validation/leakage_injection.py
```

**Post-processing (LaTeX tables from the benchmark CSVs):**

```bash
python src/benchmarking/derive_latency_percentiles.py
python src/benchmarking/derive_throughput_percentiles.py
python src/benchmarking/derive_resource_usage_table.py
python src/statistical_validation/bootstrap_sensitivity.py --latex
```

## Verifying the Results

Artifacts generated under `outputs/`:

| Artifact | Path | Contents |
|----------|------|----------|
| Temporal folds | `ml_pipeline/architectures/<arch>/prep/temporal_folds_<arch>.json` | Train/val/test intervals, gaps |
| Target statistics | `ml_pipeline/architectures/<arch>/prep/target_statistics.json` | Distribution, consistency |
| Equivalence | `statistics/equivalence_estimation.json` | Decision + 95% CI + Wilcoxon |
| Latency | `benchmarks/architectural_benchmark_results.csv` | Time per phase and repetition |
| Resources | `benchmarks/architectural_benchmark_resource_log.jsonl` | CPU/RAM/IO per sample |
| Scorecard | `statistics/architectural_scorecard.tex` | Consolidated panel |
| Predictions | `ml_pipeline/architectures/<arch>/predictions/predictions_<stage>_<arch>.parquet` | Vectors over which bitwise equivalence is asserted |
| Target coverage | `collection/raw_data/target_coverage.json` | Observed and imputed fraction per column, and the declared carry limit |
| Per-fold imputation | `ml_pipeline/architectures/<arch>/prep/fold_imputation_<arch>.json` | Cells filled with the training-window median, per split and per fold |
| Feature audit | `ml_pipeline/architectures/<arch>/prep/feature_audit_<arch>.json` | P3 re-audit of the matrix each model fits, per fold, with the result of every check |
| Provenance | `scientific_config_snapshot.json` | Commit, timestamp, core budget and the entire configuration |

The first three rows on imputation answer different questions and none of them replaces the others: how much of the input panel was observed, how far the temporal carry reached, and how much the training-window median filled in afterwards — the last one is the part with no limit on its reach.

## Customization

### Parameters

Edit `src/core/scientific_config.py`:

- `temporal_gap_years`, `folds_min_train_years`, `folds_step_years`, `embargo_years`
- `sesoi_r2`, `sesoi_mase`, `sesoi_wape` (equivalence thresholds)
- `bootstrap_iters`

### New Architecture

1. Create `src/architectures_ml/<new>/setup.py` with a subclass of `BaseArchitectureML` defining `PARADIGM_META` and implementing the abstract methods. P1, P2 and P4 are inherited from the base class. P5's imputation and P3's post-lag re-audit are not: they run over the materialised fold, which is what the paradigm implements, so the paradigm's own model has to call them. Forgetting does not pass in silence — the receipt gate demands evidence of both at the end of the hierarchical stage and halts without it.
2. Create the processing, baseline and hierarchical modules at the paths declared in `PARADIGM_META`. The framework discovers them automatically via `__init_subclass__`.

### New Dataset

Implement a `DatasetConfig` in `src/datasets/` and a collector in `src/collection/`. Use the adapter pattern to convert to the internal schema. Existing example: INEP Censo Escolar (`python pipeline.py --dataset inep_censo`).

### Extra Metrics

Add modules under `src/benchmarking/` or `src/statistical_validation/` following the JSON → LaTeX pattern.

## FAQ

**How long does it take?** About an hour and a half for the World Bank and more than a day for
the INEP Censo Escolar on the reference machine. That machine has to accommodate the core
budget: `pipeline.py` refuses to run below 8. The cost is dominated by the benchmark step,
which re-runs processing, setup, baseline and hierarchical of the three paradigms
`warmup + n` times (12 by default, so 144 measured phase executions). Collection is
the one phase it does not repeat, so the collection cache shortens that single pass
and nothing else. To explore, use a smaller `--repetitions` —
aware that this does not reproduce the latency table.

**Do I need an API key?** No. The World Bank API is open.

**Should the results match exactly across runs?** Yes. Centralized seeds and `n_jobs=1` guarantee determinism. Divergences indicate a different environment.

**Does it work on Windows?** The pipeline was tested on Linux. DuckDB and Polars work on Windows; Dask distributed may have limitations.

---

If you have questions, open an issue in the repository.
