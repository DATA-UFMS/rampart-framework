#!/usr/bin/env python3
"""
Research orchestrator pipeline: runs the phases in order with absolute paths.

Phases:
  1) Raw collection (World Bank or INEP Censo Escolar)
  2) Processing per paradigm (sql_engine, task_graph, dataframe_lib)
  3) ML setup (identical folds with 2-year gaps; feature selection)
  4) Baselines (one per paradigm)
  5) Hierarchical (one per paradigm)
  6) Architectural benchmark (one per paradigm)
  7) Statistical analysis and derived tables

Every published artifact is produced by a stage from here. An analysis script
outside this orchestrator means that reproducing the results requires knowing a
sequence that is written down nowhere.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

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
from core.validation import AntiLeakageViolation, TemporalValidator

def _log(msg: str) -> None:
    print(f"  {msg}")

def deterministic_environment() -> dict:
    """Variables that need to exist before NumPy is imported.

    The numerical libraries size their thread pools at load time, and there is
    no way to shrink them afterwards from inside the process — which is why they
    are exported to the subprocess rather than set in the orchestrator.

    PYTHONHASHSEED fixes the iteration order of sets and dictionaries with text
    keys, which some aggregation paths depend on.
    """
    blas = str(int(SCIENTIFIC_CONFIG['blas_threads']))
    engine = str(int(SCIENTIFIC_CONFIG['engine_threads']))
    return {
        # Component common to the paradigms: all of them materialize in pandas
        # before scikit-learn, so the BLAS pool is no paradigm's property.
        'OMP_NUM_THREADS': blas,
        'OPENBLAS_NUM_THREADS': blas,
        'MKL_NUM_THREADS': blas,
        'NUMEXPR_NUM_THREADS': blas,
        'VECLIB_MAXIMUM_THREADS': blas,
        # Paradigm component: Polars sizes its Rayon pool at import, so only an
        # environment variable reaches it. Dask reads DASK_NUM_WORKERS the same
        # way, which reaches it in the baseline and hierarchical stages too --
        # they run as separate processes and did not inherit the dask.config.set
        # made in the processing stage, so they measured with the host's core
        # count.
        'POLARS_MAX_THREADS': engine,
        'DASK_NUM_WORKERS': engine,
        'PYTHONHASHSEED': str(int(SCIENTIFIC_CONFIG['random_seed'])),
    }


def _validate_core_budget() -> None:
    """The declared budget has to fit on the machine.

    Oversubscribing the cores would make the latency reflect scheduling
    contention rather than the paradigm, and would do so silently.
    """
    import multiprocessing
    available = multiprocessing.cpu_count()
    engine = int(SCIENTIFIC_CONFIG['engine_threads'])
    blas = int(SCIENTIFIC_CONFIG['blas_threads'])
    if engine + blas - 1 > available:
        raise RuntimeError(
            f"Core budget does not fit on this machine: engine_threads="
            f"{engine} and blas_threads={blas}, with {available} cores "
            f"available. Adjust scientific_config instead of oversubscribing "
            f"the cores: the measured latency would come to reflect contention."
        )


def run(argv: list) -> None:
    """Runs a subprocess with PYTHONPATH pointing at src/.

    Arguments as a list, without a shell: a repository path with spaces would
    break the string form, and there is no reason to interpret metacharacters in
    paths that this module itself builds.
    """
    print(f"\n$ {' '.join(argv)}")
    env = os.environ.copy()
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    env.update(deterministic_environment())
    subprocess.run(argv, check=True, env=env)

def _snapshot_scientific_config(root: str) -> None:
    """Records the configuration and environment of the run.

    Delegates to core.config: the benchmark writes the same record, and two
    copies would diverge in precisely the file that exists to say how the run
    was made.
    """
    path = write_environment_snapshot(get_absolute_output_path("outputs"))
    print(f"\nScientific snapshot recorded at {path}")


def _discover():
    """Lazy paradigm discovery — importing triggers the registration."""
    from core.paradigm_registry import discover_paradigms
    paradigms = discover_paradigms()
    if not paradigms:
        raise RuntimeError("No paradigm discovered — check src/architectures_ml/*/setup.py")
    return paradigms


def _validate_anti_leakage_gate(root: str, started_at: datetime) -> None:
    """Validates the temporal integrity of every fold before moving on to the benchmark."""
    # Indexed, not .get with a default: a silent default here would let the gate
    # validate a gap different from the configured one.
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
            raise FileNotFoundError(f"Folds not found: {folds_path}")

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
        _log(f"  {arch}: {len(folds)} folds — temporal integrity verified")
        per_paradigm[arch] = [
            (f['train_start'], f['train_end'], f['val_start'], f['val_end'],
             f['test_start'], f['test_end']) for f in folds
        ]

    # Each paradigm used to be validated in isolation, and nothing required the
    # three to have the same folds. Different splits turn the comparison between
    # paradigms into a comparison between different problems -- and Δ=0 would be
    # falsified for that reason, not by the implementation.
    distinct = {arch: tuple(windows) for arch, windows in per_paradigm.items()}
    if len(set(distinct.values())) > 1:
        divergent = {arch: len(windows) for arch, windows in distinct.items()}
        raise ValueError(
            f"The paradigms do not share the same temporal folds "
            f"{divergent}. The comparison between them presupposes identical "
            f"splits; otherwise it measures different problems."
        )



#: Artifacts setup produces and the models consume. Both decide what the model
#: trains on, and neither was checked for which run it came from.
_SETUP_ARTIFACTS = ('feature_selection', 'temporal_folds')


def _validate_setup_provenance(run_id: str) -> None:
    """The setup artifacts belong to this run rather than to an earlier one.

    The temporal gate below checks the *content* of the folds; this checks whose
    they are. The distinction matters for feature_selection, which no gate was
    looking at: it is where the models read their feature list, so a file left
    behind by another execution would train all three on a set this run never
    selected -- and the three would agree with each other, being sat on the same
    stale file, so not even the prediction equivalence gate would notice.

    Runs right after setup rather than alongside the protocol receipts: finding
    this out costs minutes here and hours there.
    """
    prep_root = get_absolute_output_path('outputs/ml_pipeline/architectures')

    for arch in _discover():
        for stem in _SETUP_ARTIFACTS:
            path = os.path.join(prep_root, arch, 'prep', f'{stem}_{arch}.json')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{arch}: setup left no {stem}_{arch}.json at {path}"
                )

            with open(path, 'r') as handle:
                stamped = json.load(handle).get('run_id')

            if stamped is None:
                raise ValueError(
                    f"{arch}: {stem}_{arch}.json carries no run_id, so it "
                    f"cannot be shown to belong to this run: {path}"
                )
            if stamped != run_id:
                raise ValueError(
                    f"{arch}: {stem}_{arch}.json belongs to another run "
                    f"(file {stamped}, run {run_id}). This is the file the "
                    f"models read their feature set from."
                )

        _log(f"  {arch}: setup artifacts belong to this run")


#: Receipts each paradigm must leave to show it ran the two protocols the base
#: class cannot impose on it. Each entry names the artifact stem, the protocol,
#: and the field whose emptiness means the protocol did not actually run --
#: presence of the file is not enough, since one of the two writers runs
#: unconditionally and would happily record that nothing was imputed.
_PROTOCOL_RECEIPTS = (
    ('fold_imputation', 'P5', 'folds'),
    ('feature_audit', 'P3', 'folds'),
)


def _validate_protocol_receipts(run_id: str) -> None:
    """Every paradigm left evidence that it ran P5 and P3's post-lag re-audit.

    P1, P2 and P4 are enforced by the base class: they live inside concrete
    methods that the setup skeleton calls, so a paradigm cannot reach the
    models without passing through them. The other two are not. P5 and the
    re-audit of the feature set the models actually train on both run inside
    each paradigm's model code, because they need the materialised fold -- the
    one thing the paradigms exist to build differently. What the core can
    guarantee about them today is that their author remembered to call them.

    Moving the calls into the core would mean the core materialising the fold
    itself, which erases the difference the experiment measures. So the gap is
    closed from the other side: each call leaves a receipt, and this refuses to
    let the run continue when a receipt is missing, empty, or older than the
    run that is supposed to have produced it.

    A fourth paradigm that omits either call now halts the pipeline instead of
    silently reporting results under a protocol it never followed.

    Belonging to this run is established by a nonce and not by comparing
    timestamps. The temporal gate can compare clocks because it runs minutes
    after the run starts; this one runs hours later, on the INEP panel, and
    over that window a backward step of the wall clock -- an NTP correction
    after a suspend is the ordinary cause -- would make a fresh receipt look
    stale and abort the run. The nonce is also strictly the stronger test: a
    leftover receipt carrying a *newer* timestamp would pass a clock comparison
    and fails this one.
    """
    prep_root = get_absolute_output_path('outputs/ml_pipeline/architectures')

    for arch in _discover():
        for stem, protocol, evidence_field in _PROTOCOL_RECEIPTS:
            path = os.path.join(prep_root, arch, 'prep', f'{stem}_{arch}.json')
            if not os.path.exists(path):
                raise AntiLeakageViolation(
                    f"{arch}: {protocol} left no receipt at {path}. The "
                    f"protocol runs inside the paradigm's own code, and its "
                    f"absence here is indistinguishable from never running."
                )

            with open(path, 'r') as handle:
                receipt = json.load(handle)

            stamped = receipt.get('run_id')
            if stamped is None:
                raise AntiLeakageViolation(
                    f"{arch}: the {protocol} receipt carries no run_id, so "
                    f"it cannot be shown to belong to this run: {path}"
                )
            if stamped != run_id:
                raise AntiLeakageViolation(
                    f"{arch}: the {protocol} receipt belongs to another run "
                    f"(receipt {stamped}, run {run_id}). Someone else's receipt "
                    f"would attest to a protocol this execution never followed."
                )

            if not receipt.get(evidence_field):
                raise AntiLeakageViolation(
                    f"{arch}: the {protocol} receipt exists but its "
                    f"'{evidence_field}' field is empty, which is the record of "
                    f"the protocol reaching no fold at all: {path}"
                )

            # Presence is not conformance. A check whose statistic could not be
            # computed -- too few complete rows after listwise deletion, an
            # empty feature subset -- used to return None and leave a report
            # that reads exactly like one where the check had passed.
            unresolved = sorted(
                check for check, outcome
                in (receipt.get('checks_across_folds') or {}).items()
                if outcome == 'indeterminate')
            if unresolved:
                raise AntiLeakageViolation(
                    f"{arch}: the {protocol} receipt records checks that could "
                    f"not be evaluated in at least one fold: {unresolved}. A "
                    f"check that did not run is not a check that passed: {path}"
                )

        _log(f"  {arch}: P3 and P5 receipts present and from this run")


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
            f"The benchmark removed prediction artifacts that the gate had "
            f"verified: {missing}"
        )

    appeared = sorted(set(after) - set(before))
    if appeared:
        raise ValueError(
            f"The benchmark created prediction artifacts that the gate did not "
            f"see: {appeared}. What will be published has not been verified."
        )

    changed = sorted(path for path, digest in after.items()
                     if before[path] != digest)
    if changed:
        raise ValueError(
            f"The benchmark repetitions produced predictions different from the "
            f"ones the gate verified: {changed}. Either the run is not "
            f"deterministic, or the published artifacts are not the ones that "
            f"were attested -- in either case the equivalence claim does not "
            f"cover what is in the package."
        )

    _log(f"  {len(after)} prediction artifacts intact after the benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description="Research pipeline - architectural benchmarking")
    parser.add_argument(
        '--dataset', default='worldbank',
        choices=['worldbank', 'inep_censo'],
        help='Dataset to process (default: worldbank)'
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    os.environ['DATASET_NAME'] = dataset_name  # propagated to subprocesses via run()
    # Identifies this execution for the receipt gates. It travels in the
    # environment rather than as an argument because the receipts are written by
    # the models, three subprocesses below here.
    run_id = uuid4().hex
    os.environ['RAMPART_RUN_ID'] = run_id
    root = os.path.abspath(os.path.dirname(__file__))
    py = sys.executable
    started_at = datetime.now()

    _validate_core_budget()
    print(f"\nPipeline started (dataset: {dataset_name})")
    _log(f"Budget: {SCIENTIFIC_CONFIG['engine_threads']} cores per engine, "
         f"{SCIENTIFIC_CONFIG['blas_threads']} BLAS thread(s)")
    _snapshot_scientific_config(root)
    paradigms = _discover()
    print("\nStage 0: Reproducibility snapshot")
    _log(f"Snapshot saved at {get_absolute_output_path('outputs')}")

    if dataset_name == 'worldbank':
        print("\nStage 1/7: Collection")
        _log("Source: World Bank")
        run([py, os.path.join(root, "src/collection/raw_data_collector.py")])
    else:
        print("\nStage 1/7: Collection")
        _log("Source: INEP Censo Escolar")
        run([py, os.path.join(root, "src/collection/inep_collector.py")])
    _log("Stage 1 completed")

    n_paradigms = len(paradigms)
    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print(f"\nStage 2{chr(96+i)}/7: Processing {arch}")
        _log(f"Architecture: {info['label']}")
        run([py, os.path.join(root, info["processor_script"])])
    _log(f"Stage 2 completed ({n_paradigms} paradigms)")

    print("\nStage 3: ML setup")
    _log("Temporal gaps: 2 years (P1-P3)")
    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print(f"\nStage 3{chr(96+i)}/7: ML setup {arch}")
        _log(f"Architecture: {info['label']}")
        run([py, os.path.join(root, info["setup_script"])])
    _log(f"Stage 3 completed ({n_paradigms} paradigms)")

    print("\nSetup provenance gate")
    _validate_setup_provenance(run_id)
    _log("feature_selection and temporal_folds belong to this run")

    print("\nAnti-leakage gate")
    _validate_anti_leakage_gate(root, started_at)
    _log("All folds passed temporal validation")

    for i, (arch, info) in enumerate(paradigms.items(), 1):
        print(f"\nStage 4{chr(96+i)}/7: Baselines {arch}")
        run([py, os.path.join(root, info["baseline_script"])])
    _log(f"Stage 4 completed ({n_paradigms} paradigms)")

    print("\nStage 5/7: Hierarchical")
    for arch, info in paradigms.items():
        run([py, os.path.join(root, info["hierarchical_script"])])
    _log(f"Stage 5 completed ({n_paradigms} paradigms)")

    # Here and not alongside the anti-leakage gate: the P3 and P5 receipts are
    # written by the models, in the stage above. Checking them earlier would be
    # checking files that do not exist yet.
    print("\nProtocol receipts gate")
    _validate_protocol_receipts(run_id)
    # Names what the receipt covers. Each paradigm's scaler is fitted on the
    # training window right after imputation, but emits no report, so no receipt
    # attests to it -- and saying "P5 verified" would cover both.
    _log("P3 re-audit and P5 imputation evidenced in every paradigm "
         "(the scaler fit leaves no receipt)")

    # Precedes the benchmark: a latency comparison between paradigms is only
    # meaningful once they are established to predict the same values for the
    # same rows. Running it afterwards could report a timing difference between
    # paradigms that were not doing the same work.
    print("\nPrediction equivalence gate")
    run([py, os.path.join(root, "src/statistical_validation/prediction_equivalence.py")])
    _log("Predictions identical across the paradigms")

    # Recorded before the benchmark, checked afterwards: the repetitions re-run
    # Stages 3 to 5 and overwrite these same files.
    predictions_before = _prediction_digests()
    if not predictions_before:
        raise FileNotFoundError(
            "No prediction artifact before the benchmark; the equivalence gate "
            "would have nothing to verify."
        )

    print("\nStage 6/7: Architectural benchmark")
    # Without --repetitions/--warmup: the benchmark reads BENCHMARK_CONFIG, and
    # repeating the values here would create a second source for the protocol's n.
    run([py, os.path.join(root, "src/benchmarking/architectural_benchmark.py")])
    _assert_benchmark_left_predictions_intact(predictions_before)
    _log("Stage 6 completed")

    print("\nStage 7/7: Statistical analysis and derived tables")

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
        ('a', 'Significance (bootstrap)',
         'src/statistical_validation/significance_tests.py', []),
        ('b', 'Equivalence (SESOI + CI)',
         'src/statistical_validation/equivalence_estimation.py', ['--latex']),
        ('c', 'Effect sizes and multiple comparisons',
         'src/statistical_validation/effect_analysis.py', []),
        ('d', 'Sensitivity to the number of resamples',
         'src/statistical_validation/bootstrap_sensitivity.py', []),
        # After (a)-(c) because it reads the prediction vectors, and before the
        # tables because its result goes into the info sheet.
        ('d2', 'Model against the best baseline',
         'src/statistical_validation/baseline_comparison.py', []),
        ('e', 'Latency percentiles',
         'src/benchmarking/derive_latency_percentiles.py', []),
        ('f', 'Throughput percentiles',
         'src/benchmarking/derive_throughput_percentiles.py', []),
        ('g', 'Resource usage',
         'src/benchmarking/derive_resource_usage_table.py', []),
        ('h', 'Operational panel',
         'src/benchmarking/derive_operational_panel.py', []),
        ('i', 'Stage attribution (engine vs fitting)',
         'src/benchmarking/derive_stage_attribution.py', []),
        ('j', 'Scorecard',
         'src/statistical_validation/make_scorecard.py', []),
    ]
    for suffix, description, script, script_args in ANALYSIS_STAGES:
        _log(f"Stage 7{suffix}/7: {description}")
        run([py, os.path.join(root, script)] + script_args)

    _log("Stage 7 completed")

    print("\nPipeline completed")
    print(f"Results at: {get_absolute_output_path('outputs')}")

if __name__ == "__main__":
    main()
