#!/usr/bin/env bash
# The runs that do not finish on a laptop, in the order their answers are worth.
#
# Everything the paper claims was measured by probes on one CPU. What is here
# needs a bigger machine for one reason each, stated per run, and three of the
# four are full-panel versions of something already measured on a subsample --
# so each closes a confound that is currently declared rather than resolved.
#
#   RAMPART_PANEL_DIR   directory holding azure_results_v7_wb/ and
#                       azure_results_v7_inep/. The container does not carry
#                       them and the copies under outputs/ are eight-byte
#                       fixtures.
#   RAMPART_CLOUD_OUT   where logs go. Defaults to ./cloud-logs.
#
# Usage:  scripts/cloud_run.sh [r1 r2 r4 ...]     (default: r1 r2)
set -uo pipefail

OUT="${RAMPART_CLOUD_OUT:-cloud-logs}"
mkdir -p "$OUT"
PY="${RAMPART_PYTHON:-python3}"
RUNS=("${@:-r1 r2}")
[ $# -eq 0 ] && RUNS=(r1 r2)

if [ -z "${RAMPART_PANEL_DIR:-}" ]; then
  echo "RAMPART_PANEL_DIR is not set; the probes will look next to the repo." >&2
fi

# Fail before spending an hour: both panels must load.
if ! "$PY" -c "
import sys; sys.path.insert(0, 'scripts/validation')
from probe_harness import panel
for name in ('worldbank', 'inep_censo'):
    panel(name)
" 2>&1 | tail -5; then
  echo "panels do not load -- fix RAMPART_PANEL_DIR before starting" >&2
  exit 1
fi
echo "panels load."

run () {
  local tag="$1"; shift
  local log="$OUT/${tag}.log"
  echo "=== $tag: $* ===" | tee "$log"
  local started=$SECONDS
  if "$PY" "$@" >>"$log" 2>&1; then
    echo "$tag finished in $(( (SECONDS - started) / 60 )) min" | tee -a "$log"
  else
    # Not fatal: a later run may still be worth its time, and a partial log is
    # more useful than a halted queue with no record of why.
    echo "$tag FAILED after $(( (SECONDS - started) / 60 )) min" | tee -a "$log"
  fi
}

for what in ${RUNS[@]}; do
  case "$what" in
    # Separates "the decay slope is panel-specific" from "it is a function of n".
    # The subsampled replication cannot: it moves n_train from 41k to 5k, and
    # absorption depends on n.
    r1) run r1_routes_inep scripts/validation/probe_global_routes.py inep_censo ;;
    # Same confound for the attenuation ratio: 0.0725 on World Bank against
    # 0.249-0.316 on the subsample, predicted 0.1055.
    r2) run r2_label_inep scripts/validation/probe_label_channel.py inep_censo ;;
    # In-context models on the large panel. Do not start before the windowing
    # rule is pre-registered -- TabPFN caps context at 10k of 41k training rows,
    # and choosing which rows after seeing results is a free parameter.
    r3) run r3_icl_inep scripts/validation/probe_leakage_channels.py ;;
    # The published artifact, for the latency intervals. Needs the machine to
    # itself: it measures time.
    r4) run r4_artifact pipeline.py ;;
    *) echo "unknown run: $what" >&2 ;;
  esac
done

echo; echo "logs in $OUT/"
