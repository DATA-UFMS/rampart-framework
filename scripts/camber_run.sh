#!/usr/bin/env bash
# Submit the runs that do not finish on a laptop to Camber.
#
# Everything the paper claims was measured by probes on one CPU. What is here
# needs a bigger machine for one reason each, and three of the four are
# full-panel versions of something already measured on a subsample -- so each
# closes a confound that is currently declared rather than resolved.
#
# The `base` engine is an astronomy image -- astropy, galpy, emcee, mpi4py, and
# numpy/pandas/matplotlib. Its description claims sklearn and pytorch and it has
# neither, which is how the first three submissions died in 36 seconds. Both are
# pip-installable and pip works, so each job installs what it needs.
#
# The versions are pinned to the lockfile rather than left to resolve. These runs
# exist to be compared against measurements made locally -- the whole point of
# the full INEP run is to separate a panel effect from an n effect against the
# subsample -- and a different scikit-learn would put estimator changes into that
# comparison. Pinning is not tidiness here, it is the comparison.
#
# pyarrow is on the list because pandas reads parquet through it and does not
# depend on it: the panels are parquet, so without it a run dies at the first
# read a minute in rather than at import. dask and polars are deliberately
# absent -- a static scan of the import graph reaches them through the paradigm
# modules, but the probes never execute those, and the run that got as far as
# reading a file proved it.
#
#   STASH        where camber_setup.sh put things
#   CAMBER_SIZE  node size for the CPU runs (default small = 16 cores)
#
# Usage:  scripts/camber_run.sh [r1 r2 r3 r4]      (default: r1 r2)
set -uo pipefail

USER_NAME="${CAMBER_USER:-$(camber me 2>/dev/null | awk '/^Username:/ {print $2}')}"
STASH="${STASH:-stash://${USER_NAME}/rampart}"
SIZE="${CAMBER_SIZE:-small}"
RUNS=("$@"); [ $# -eq 0 ] && RUNS=(r1 r2)

# Every job starts from the Stash path as its working directory, so the probes
# need src/ importable and the panels findable. Both are relative to it.
PRELUDE='export PYTHONPATH=$PWD/src:$PWD/scripts/validation:$PYTHONPATH;
  export RAMPART_PANEL_DIR=$PWD/panels;
  python3 -m pip install --quiet numpy==2.2.1 pandas==2.3.1 scikit-learn==1.5.2 scipy==1.14.1 pyarrow==18.1.0 psutil &&'

submit () {
  local tag="$1" gpu="$2"; shift 2
  local flags=(--engine base --size "$SIZE" --path "$STASH/")
  [ "$gpu" = gpu ] && flags=(--engine base --size xsmall --gpu --path "$STASH/")
  echo "--- $tag"
  camber job create "${flags[@]}" --cmd "$PRELUDE $*" || {
    echo "$tag: submission failed" >&2; return 1; }
}

for what in "${RUNS[@]}"; do
  case "$what" in
    # Separates "the decay slope is panel-specific" from "it is a function of n".
    # The subsampled replication cannot: it moves n_train from 41k to 5k, and
    # absorption depends on n.
    r1) submit r1_routes_inep cpu \
        'python3 scripts/validation/probe_global_routes.py inep_censo' ;;
    # Same confound for the attenuation ratio: 0.0725 on World Bank against
    # 0.249-0.316 on the subsample, predicted 0.1055.
    r2) submit r2_label_inep cpu \
        'python3 scripts/validation/probe_label_channel.py inep_censo' ;;
    # In-context models, and note `python3 -m pip` rather than `pip`. The GPU
    # node runs a barer image than the CPU one -- no numpy at all -- and its bare
    # `pip` belongs to a different interpreter than its `python3`, so a first
    # attempt installed the packages somewhere the script could not see and died
    # importing numpy after a successful install.
    #
    # The extra is not in the base image, and TabPFN must
    # resolve to v2 -- the newer weights need a browser licence and a personal
    # token, which a batch job does not have. The adapter pins it and aborts if
    # it resolves otherwise, so a wrong image fails loudly here rather than
    # producing numbers from a model nobody chose.
    r3) submit r3_icl_wb gpu \
        'python3 -m pip install "tabpfn>=8,<9" "tabicl>=2,<3" &&
         python3 scripts/validation/probe_leakage_channels.py' ;;
    # The published artifact, for the latency intervals. Wants the node to
    # itself, because it measures time.
    r4) submit r4_artifact cpu 'python3 pipeline.py' ;;
    *) echo "unknown run: $what" >&2 ;;
  esac
done

echo
echo "camber job list          # ids and status"
echo "camber job log ID ~/Downloads/"
