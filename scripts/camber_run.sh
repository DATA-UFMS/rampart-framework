#!/usr/bin/env bash
# Submit the runs that do not finish on a laptop to Camber.
#
# Everything the paper claims was measured by probes on one CPU. What is here
# needs a bigger machine for one reason each, and three of the four are
# full-panel versions of something already measured on a subsample -- so each
# closes a confound that is currently declared rather than resolved.
#
# The `base` engine already carries sklearn, pandas and pytorch. Only the
# in-context extra is missing, so r3 installs it inside its own command rather
# than everything paying for it.
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
PRELUDE='export PYTHONPATH=$PWD/src:$PWD/scripts/validation:$PYTHONPATH; export RAMPART_PANEL_DIR=$PWD/panels;'

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
    # In-context models. The extra is not in the base image, and TabPFN must
    # resolve to v2 -- the newer weights need a browser licence and a personal
    # token, which a batch job does not have. The adapter pins it and aborts if
    # it resolves otherwise, so a wrong image fails loudly here rather than
    # producing numbers from a model nobody chose.
    r3) submit r3_icl_wb gpu \
        'pip install --quiet "tabpfn>=8,<9" "tabicl>=2,<3" &&
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
